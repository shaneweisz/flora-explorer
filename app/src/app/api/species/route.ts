import { NextRequest, NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";
import { getTaxonConfig, TaxonConfig } from "@/config/taxa";

interface SpeciesRecord {
  species_key: number;
  occurrence_count: number;
  scientific_name?: string;
  redlist_category?: string | null;
}

interface RedListSpecies {
  scientific_name: string;
  category: string;
}

// Cache per taxon (for CSV data)
const dataCache: Record<string, SpeciesRecord[]> = {};

// Valid species keys cache (for filtering live queries)
const validSpeciesKeysCache: Record<string, Set<number>> = {};

// Red List lookup cache: scientific_name (lowercase) -> category
const redListCache: Record<string, Map<string, string>> = {};

// Data source keys for live queries
const DATA_SOURCES: Record<string, { type: "dataset" | "publishingOrg"; key: string }> = {
  iNaturalist: { type: "dataset", key: "50c9509d-22c7-4a22-a47d-8c48425ef4a7" },
  iRecord: { type: "publishingOrg", key: "32f1b389-5871-4da3-832f-9a89132520c5" },
  BSBI: { type: "publishingOrg", key: "aa569acf-991d-4467-b327-8442f30ddbd2" },
};

async function loadRedListLookup(taxon: TaxonConfig): Promise<Map<string, string>> {
  const cacheKey = taxon.id;
  if (redListCache[cacheKey]) return redListCache[cacheKey];

  const lookup = new Map<string, string>();

  // Load from primary dataFile or multiple dataFiles
  const files = taxon.dataFiles || [taxon.dataFile];

  for (const file of files) {
    const filePath = path.join(process.cwd(), "data", file);
    try {
      const content = await fs.readFile(filePath, "utf-8");
      const data = JSON.parse(content);
      const species: RedListSpecies[] = data.species || [];

      for (const sp of species) {
        if (sp.scientific_name && sp.category) {
          // Normalize name for matching (lowercase, trim)
          const normalizedName = sp.scientific_name.toLowerCase().trim();
          lookup.set(normalizedName, sp.category);
        }
      }
    } catch {
      // File not found or invalid, skip
    }
  }

  redListCache[cacheKey] = lookup;
  return lookup;
}

async function loadCsvData(taxonId: string): Promise<SpeciesRecord[]> {
  if (dataCache[taxonId]) return dataCache[taxonId];

  const taxon = getTaxonConfig(taxonId);
  const filePath = path.join(process.cwd(), "data", taxon.gbifDataFile);

  try {
    const fileContent = await fs.readFile(filePath, "utf-8");
    const lines = fileContent.trim().split("\n");
    const header = lines[0];
    const hasScientificName = header.includes("scientific_name");

    // Load Red List lookup for this taxon
    const redListLookup = await loadRedListLookup(taxon);

    // Skip header
    dataCache[taxonId] = lines.slice(1).map((line) => {
      const parts = line.split(",");
      const species_key = parseInt(parts[0], 10);
      const occurrence_count = parseInt(parts[1], 10);
      const scientific_name = hasScientificName ? parts[2] || undefined : undefined;

      // Look up Red List category by scientific name
      let redlist_category: string | null = null;
      if (scientific_name) {
        const normalizedName = scientific_name.toLowerCase().trim();
        redlist_category = redListLookup.get(normalizedName) || null;
      }

      return {
        species_key,
        occurrence_count,
        scientific_name,
        redlist_category,
      };
    });

    // Also build the valid species keys set for filtering live queries
    validSpeciesKeysCache[taxonId] = new Set(
      dataCache[taxonId].map(r => r.species_key)
    );

    return dataCache[taxonId];
  } catch {
    // File doesn't exist for this taxon yet
    return [];
  }
}

// Get valid species keys from CSV (for filtering live query results)
async function getValidSpeciesKeys(taxonId: string): Promise<Set<number>> {
  if (validSpeciesKeysCache[taxonId]) return validSpeciesKeysCache[taxonId];

  // Load CSV data which will populate the cache
  await loadCsvData(taxonId);

  return validSpeciesKeysCache[taxonId] || new Set();
}

// Handle unfiltered requests using CSV data (accurate, pre-validated species)
async function handleCsvRequest(
  taxonId: string,
  page: number,
  limit: number,
  minCount: number,
  maxCount: number,
  sortOrder: string,
  redlistFilter: string | null
) {
  const data = await loadCsvData(taxonId);

  // Filter by occurrence count range
  let filtered = data.filter(
    (d) => d.occurrence_count >= minCount && d.occurrence_count <= maxCount
  );

  // Filter by Red List category
  if (redlistFilter && redlistFilter !== "all") {
    if (redlistFilter === "NE") {
      filtered = filtered.filter((d) => !d.redlist_category);
    } else {
      filtered = filtered.filter((d) => d.redlist_category === redlistFilter);
    }
  }

  // Sort
  if (sortOrder === "asc") {
    filtered = [...filtered].sort((a, b) => a.occurrence_count - b.occurrence_count);
  }
  // Default is already sorted desc from the CSV

  // Paginate
  const start = (page - 1) * limit;
  const end = start + limit;
  const paginated = filtered.slice(start, end);

  // Calculate stats
  const assessed = data.filter((d) => d.redlist_category);
  const notAssessed = data.filter((d) => !d.redlist_category);

  const stats = {
    total: data.length,
    filtered: filtered.length,
    totalOccurrences: data.reduce((sum, d) => sum + d.occurrence_count, 0),
    median: data[Math.floor(data.length / 2)]?.occurrence_count || 0,
    distribution: {
      eq1: data.filter((d) => d.occurrence_count === 1).length,
      gt1_lte10: data.filter((d) => d.occurrence_count > 1 && d.occurrence_count <= 10).length,
      gt10_lte100: data.filter((d) => d.occurrence_count > 10 && d.occurrence_count <= 100).length,
      gt100_lte1000: data.filter((d) => d.occurrence_count > 100 && d.occurrence_count <= 1000).length,
      gt1000_lte10000: data.filter((d) => d.occurrence_count > 1000 && d.occurrence_count <= 10000).length,
      gt10000: data.filter((d) => d.occurrence_count > 10000).length,
    },
    redlist: {
      assessed: assessed.length,
      notAssessed: notAssessed.length,
      assessedOccurrences: assessed.reduce((sum, d) => sum + d.occurrence_count, 0),
      notAssessedOccurrences: notAssessed.reduce((sum, d) => sum + d.occurrence_count, 0),
    },
  };

  return NextResponse.json({
    data: paginated,
    pagination: {
      page,
      limit,
      total: filtered.length,
      totalPages: Math.ceil(filtered.length / limit),
    },
    stats,
  });
}

// Handle filtered requests using live GBIF queries
async function handleLiveRequest(
  taxon: TaxonConfig,
  taxonId: string,
  page: number,
  limit: number,
  minCount: number,
  maxCount: number,
  sortOrder: string,
  redlistFilter: string | null,
  basisOfRecord: string | null,
  maxUncertainty: string | null,
  dataSource: string | null
) {
  const redListLookup = await loadRedListLookup(taxon);

  // Get valid species keys from CSV to filter out subspecies/synonyms
  const validSpeciesKeys = await getValidSpeciesKeys(taxonId);

  // Use GBIF occurrence search with facets
  const gbifParams = new URLSearchParams({
    facet: "speciesKey",
    facetLimit: "500000",
    limit: "0",
    hasCoordinate: "true",
    hasGeospatialIssue: "false",
  });

  // Add basisOfRecord filter
  if (basisOfRecord) {
    if (basisOfRecord === "OTHER") {
      ["OBSERVATION", "MATERIAL_CITATION", "OCCURRENCE", "LIVING_SPECIMEN", "FOSSIL_SPECIMEN"].forEach(type => {
        gbifParams.append("basisOfRecord", type);
      });
    } else {
      gbifParams.set("basisOfRecord", basisOfRecord);
    }
  }

  // Add coordinate uncertainty filter
  if (maxUncertainty) {
    gbifParams.set("coordinateUncertaintyInMeters", `*,${maxUncertainty}`);
  }

  // Add data source filter
  if (dataSource && DATA_SOURCES[dataSource]) {
    const source = DATA_SOURCES[dataSource];
    if (source.type === "dataset") {
      gbifParams.set("datasetKey", source.key);
    } else {
      gbifParams.set("publishingOrg", source.key);
    }
  }

  // Add taxon filter
  if (taxon.gbifClassKey) {
    gbifParams.set("classKey", taxon.gbifClassKey.toString());
  } else if (taxon.gbifClassKeys && taxon.gbifClassKeys.length > 0) {
    taxon.gbifClassKeys.forEach(key => {
      gbifParams.append("classKey", key.toString());
    });
  } else if (taxon.gbifOrderKeys && taxon.gbifOrderKeys.length > 0) {
    taxon.gbifOrderKeys.forEach(key => {
      gbifParams.append("orderKey", key.toString());
    });
  } else if (taxon.gbifKingdomKey) {
    gbifParams.set("kingdomKey", taxon.gbifKingdomKey.toString());
  }

  const response = await fetch(
    `https://api.gbif.org/v1/occurrence/search?${gbifParams}`
  );

  if (!response.ok) {
    throw new Error(`GBIF API error: ${response.statusText}`);
  }

  const data = await response.json();

  const speciesFacets = data.facets?.find(
    (f: { field: string }) => f.field === "SPECIES_KEY"
  );

  if (!speciesFacets?.counts) {
    return NextResponse.json({
      data: [],
      pagination: { page: 1, limit, total: 0, totalPages: 0 },
      stats: {
        total: 0,
        filtered: 0,
        totalOccurrences: 0,
        median: 0,
        distribution: { eq1: 0, gt1_lte10: 0, gt10_lte100: 0, gt100_lte1000: 0, gt1000_lte10000: 0, gt10000: 0 },
      },
    });
  }

  // Convert facets to species records, filtering to only valid species from CSV
  const allSpecies = speciesFacets.counts
    .map((facet: { name: string; count: number }) => ({
      speciesKey: parseInt(facet.name),
      count: facet.count,
    }))
    .filter((sp: { speciesKey: number }) => validSpeciesKeys.has(sp.speciesKey));

  // Calculate stats
  const totalOccurrences = allSpecies.reduce((sum: number, s: { count: number }) => sum + s.count, 0);
  const counts = allSpecies.map((s: { count: number }) => s.count).sort((a: number, b: number) => a - b);
  const median = counts.length > 0 ? counts[Math.floor(counts.length / 2)] : 0;

  const distribution = {
    eq1: allSpecies.filter((s: { count: number }) => s.count === 1).length,
    gt1_lte10: allSpecies.filter((s: { count: number }) => s.count > 1 && s.count <= 10).length,
    gt10_lte100: allSpecies.filter((s: { count: number }) => s.count > 10 && s.count <= 100).length,
    gt100_lte1000: allSpecies.filter((s: { count: number }) => s.count > 100 && s.count <= 1000).length,
    gt1000_lte10000: allSpecies.filter((s: { count: number }) => s.count > 1000 && s.count <= 10000).length,
    gt10000: allSpecies.filter((s: { count: number }) => s.count > 10000).length,
  };

  // Filter by count range
  let filteredSpecies = allSpecies.filter(
    (s: { count: number }) => s.count >= minCount && s.count <= maxCount
  );

  // Sort
  if (sortOrder === "asc") {
    filteredSpecies.sort((a: { count: number }, b: { count: number }) => a.count - b.count);
  } else {
    filteredSpecies.sort((a: { count: number }, b: { count: number }) => b.count - a.count);
  }

  // Paginate
  const total = filteredSpecies.length;
  const totalPages = Math.ceil(total / limit);
  const startIdx = (page - 1) * limit;
  const pageSpecies = filteredSpecies.slice(startIdx, startIdx + limit);

  // Fetch species details for the page
  const speciesWithNames = await Promise.all(
    pageSpecies.map(async (sp: { speciesKey: number; count: number }) => {
      try {
        const speciesResponse = await fetch(
          `https://api.gbif.org/v1/species/${sp.speciesKey}`
        );
        const speciesData = await speciesResponse.json();
        const canonicalName = speciesData.canonicalName || speciesData.scientificName;

        const normalizedName = canonicalName?.toLowerCase().trim();
        const redlist_category = normalizedName ? redListLookup.get(normalizedName) || null : null;

        return {
          species_key: sp.speciesKey,
          occurrence_count: sp.count,
          scientific_name: canonicalName,
          vernacularName: speciesData.vernacularName,
          redlist_category,
        };
      } catch {
        return {
          species_key: sp.speciesKey,
          occurrence_count: sp.count,
          scientific_name: `Species ${sp.speciesKey}`,
          redlist_category: null,
        };
      }
    })
  );

  // Apply Red List filter (only works on current page for live queries)
  let finalData = speciesWithNames;
  if (redlistFilter && redlistFilter !== "all") {
    if (redlistFilter === "NE") {
      finalData = speciesWithNames.filter(s => !s.redlist_category);
    } else {
      finalData = speciesWithNames.filter(s => s.redlist_category === redlistFilter);
    }
  }

  return NextResponse.json({
    data: finalData,
    pagination: {
      page,
      limit,
      total,
      totalPages,
    },
    stats: {
      total: allSpecies.length,
      filtered: total,
      totalOccurrences,
      median,
      distribution,
    },
    // Flag to indicate this is live data (unvalidated species counts)
    isLiveQuery: true,
  });
}

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const taxonId = searchParams.get("taxon") || "plantae";
  const page = parseInt(searchParams.get("page") || "1", 10);
  const limit = Math.min(parseInt(searchParams.get("limit") || "100", 10), 1000);
  const minCount = parseInt(searchParams.get("minCount") || "0", 10);
  const maxCount = parseInt(searchParams.get("maxCount") || "999999999", 10);
  const sortOrder = searchParams.get("sort") || "desc";
  const redlistFilter = searchParams.get("redlist");

  // GBIF filter params
  const basisOfRecord = searchParams.get("basisOfRecord");
  const maxUncertainty = searchParams.get("maxUncertainty");
  const dataSource = searchParams.get("dataSource");

  // Check if any GBIF filters are applied
  const hasGbifFilters = basisOfRecord || maxUncertainty || dataSource;

  const taxon = getTaxonConfig(taxonId);

  try {
    if (hasGbifFilters) {
      // Use live GBIF queries when filters are applied
      return await handleLiveRequest(
        taxon,
        taxonId,
        page,
        limit,
        minCount,
        maxCount,
        sortOrder,
        redlistFilter,
        basisOfRecord,
        maxUncertainty,
        dataSource
      );
    } else {
      // Use pre-computed CSV data for accurate species counts
      return await handleCsvRequest(
        taxonId,
        page,
        limit,
        minCount,
        maxCount,
        sortOrder,
        redlistFilter
      );
    }
  } catch (error) {
    console.error("Error fetching species:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}

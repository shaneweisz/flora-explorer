import { NextRequest, NextResponse } from "next/server";
import { getTaxonConfig, TaxonConfig } from "@/config/taxa";
import { promises as fs } from "fs";
import path from "path";

interface RedListSpecies {
  scientific_name: string;
  category: string;
}

// Red List lookup cache: taxon -> (scientific_name lowercase -> category)
const redListCache: Record<string, Map<string, string>> = {};

// GBIF species key -> scientific name cache (from our pre-computed CSV files)
const gbifNameCache: Record<string, Map<number, string>> = {};

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

// Load species key -> name mapping from GBIF CSV
async function loadGbifNameLookup(taxon: TaxonConfig): Promise<Map<number, string>> {
  const cacheKey = taxon.id;
  if (gbifNameCache[cacheKey]) return gbifNameCache[cacheKey];

  const lookup = new Map<number, string>();
  const filePath = path.join(process.cwd(), "data", taxon.gbifDataFile);

  try {
    const content = await fs.readFile(filePath, "utf-8");
    const lines = content.trim().split("\n");
    const header = lines[0];
    const hasScientificName = header.includes("scientific_name");

    if (hasScientificName) {
      for (let i = 1; i < lines.length; i++) {
        const parts = lines[i].split(",");
        const speciesKey = parseInt(parts[0], 10);
        const scientificName = parts[2];
        if (speciesKey && scientificName) {
          lookup.set(speciesKey, scientificName);
        }
      }
    }
  } catch {
    // File not found, skip
  }

  gbifNameCache[cacheKey] = lookup;
  return lookup;
}

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ code: string }> }
) {
  const { code } = await params;
  const countryCode = code.toUpperCase();

  const searchParams = request.nextUrl.searchParams;
  const taxonId = searchParams.get("taxon") || "plantae";
  const page = parseInt(searchParams.get("page") || "1");
  const limit = parseInt(searchParams.get("limit") || "10");
  const minCount = parseInt(searchParams.get("minCount") || "0");
  const maxCount = parseInt(searchParams.get("maxCount") || "999999999");
  const sort = searchParams.get("sort") || "desc";
  const redlistFilter = searchParams.get("redlist"); // "all", "NE", or specific category (CR, EN, VU, etc.)

  // New filters for observation type, uncertainty, and data source
  const basisOfRecord = searchParams.get("basisOfRecord"); // HUMAN_OBSERVATION, PRESERVED_SPECIMEN, etc.
  const maxUncertainty = searchParams.get("maxUncertainty"); // max coordinate uncertainty in meters
  const dataSource = searchParams.get("dataSource"); // iNaturalist, iRecord, BSBI, or null for all

  // Data source keys
  const DATA_SOURCES: Record<string, { type: "dataset" | "publishingOrg"; key: string }> = {
    iNaturalist: { type: "dataset", key: "50c9509d-22c7-4a22-a47d-8c48425ef4a7" },
    iRecord: { type: "publishingOrg", key: "32f1b389-5871-4da3-832f-9a89132520c5" }, // BRC
    BSBI: { type: "publishingOrg", key: "aa569acf-991d-4467-b327-8442f30ddbd2" },
  };

  const taxon = getTaxonConfig(taxonId);

  // Load lookups early so we can filter by Red List status
  const redListLookup = await loadRedListLookup(taxon);
  const gbifNameLookup = await loadGbifNameLookup(taxon);

  try {
    // Use GBIF occurrence search with facets to get species counts for the country
    const gbifParams = new URLSearchParams({
      country: countryCode,
      facet: "speciesKey",
      facetLimit: "500000",
      limit: "0", // We only want facets, not actual occurrences
      hasCoordinate: "true",
      hasGeospatialIssue: "false",
    });

    // Add basisOfRecord filter if specified
    if (basisOfRecord) {
      if (basisOfRecord === "OTHER") {
        // "OTHER" includes: OBSERVATION, MATERIAL_CITATION, OCCURRENCE, LIVING_SPECIMEN, FOSSIL_SPECIMEN
        ["OBSERVATION", "MATERIAL_CITATION", "OCCURRENCE", "LIVING_SPECIMEN", "FOSSIL_SPECIMEN"].forEach(type => {
          gbifParams.append("basisOfRecord", type);
        });
      } else {
        gbifParams.set("basisOfRecord", basisOfRecord);
      }
    }

    // Add coordinate uncertainty filter if specified
    if (maxUncertainty) {
      gbifParams.set("coordinateUncertaintyInMeters", `*,${maxUncertainty}`);
    }

    // Add data source filter if specified
    if (dataSource && DATA_SOURCES[dataSource]) {
      const source = DATA_SOURCES[dataSource];
      if (source.type === "dataset") {
        gbifParams.set("datasetKey", source.key);
      } else {
        gbifParams.set("publishingOrg", source.key);
      }
    }

    // Add taxon filter - use classKey(s) if available, otherwise kingdomKey
    if (taxon.gbifClassKey) {
      gbifParams.set("classKey", taxon.gbifClassKey.toString());
    } else if (taxon.gbifClassKeys && taxon.gbifClassKeys.length > 0) {
      // Multiple class keys - add each as a separate param
      taxon.gbifClassKeys.forEach(key => {
        gbifParams.append("classKey", key.toString());
      });
    } else if (taxon.gbifOrderKeys && taxon.gbifOrderKeys.length > 0) {
      // Multiple order keys (e.g., fishes)
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

    // Extract species facets
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
        country: countryCode,
      });
    }

    // Convert facets to species records with counts and Red List status
    // Filter to only include valid species from our CSV (excludes subspecies, synonyms, etc.)
    const allSpecies: { speciesKey: number; count: number; scientificName: string | undefined; redlistCategory: string | null }[] = speciesFacets.counts
      .map((facet: { name: string; count: number }) => {
        const speciesKey = parseInt(facet.name);
        // Look up name from our cached GBIF data
        const scientificName = gbifNameLookup.get(speciesKey);
        // Look up Red List category
        const normalizedName = scientificName?.toLowerCase().trim();
        const redlistCategory = normalizedName ? redListLookup.get(normalizedName) || null : null;

        return {
          speciesKey,
          count: facet.count,
          scientificName,
          redlistCategory,
        };
      })
      .filter((sp: { speciesKey: number }) => gbifNameLookup.has(sp.speciesKey));

    // Calculate stats from all species (before filtering)
    const totalOccurrences = allSpecies.reduce((sum, s) => sum + s.count, 0);
    const counts = allSpecies.map(s => s.count).sort((a, b) => a - b);
    const median = counts.length > 0 ? counts[Math.floor(counts.length / 2)] : 0;

    const distribution = {
      // Histogram buckets (non-cumulative)
      eq1: allSpecies.filter(s => s.count === 1).length,
      gt1_lte10: allSpecies.filter(s => s.count > 1 && s.count <= 10).length,
      gt10_lte100: allSpecies.filter(s => s.count > 10 && s.count <= 100).length,
      gt100_lte1000: allSpecies.filter(s => s.count > 100 && s.count <= 1000).length,
      gt1000_lte10000: allSpecies.filter(s => s.count > 1000 && s.count <= 10000).length,
      gt10000: allSpecies.filter(s => s.count > 10000).length,
    };

    // Red List stats (before filtering)
    const assessed = allSpecies.filter(s => s.redlistCategory);
    const notAssessed = allSpecies.filter(s => !s.redlistCategory);
    const redlistStats = {
      assessed: assessed.length,
      notAssessed: notAssessed.length,
      assessedOccurrences: assessed.reduce((sum, s) => sum + s.count, 0),
      notAssessedOccurrences: notAssessed.reduce((sum, s) => sum + s.count, 0),
    };

    // Filter by count range
    let filteredSpecies = allSpecies.filter(
      s => s.count >= minCount && s.count <= maxCount
    );

    // Filter by Red List category
    if (redlistFilter && redlistFilter !== "all") {
      if (redlistFilter === "NE") {
        // Not Evaluated = no redlistCategory
        filteredSpecies = filteredSpecies.filter(s => !s.redlistCategory);
      } else {
        // Specific category (CR, EN, VU, NT, LC, DD, EW, EX)
        filteredSpecies = filteredSpecies.filter(s => s.redlistCategory === redlistFilter);
      }
    }

    // Sort
    if (sort === "asc") {
      filteredSpecies.sort((a, b) => a.count - b.count);
    } else {
      filteredSpecies.sort((a, b) => b.count - a.count);
    }

    // Paginate
    const total = filteredSpecies.length;
    const totalPages = Math.ceil(total / limit);
    const startIdx = (page - 1) * limit;
    const pageSpecies = filteredSpecies.slice(startIdx, startIdx + limit);

    // Fetch additional species details (vernacular names) for the page
    // Use cached scientific name if available, fall back to GBIF API
    const speciesWithNames = await Promise.all(
      pageSpecies.map(async (sp) => {
        // If we already have the name from our cache, just fetch vernacular name
        if (sp.scientificName) {
          try {
            const speciesResponse = await fetch(
              `https://api.gbif.org/v1/species/${sp.speciesKey}`
            );
            const speciesData = await speciesResponse.json();
            return {
              species_key: sp.speciesKey,
              occurrence_count: sp.count,
              canonicalName: sp.scientificName,
              vernacularName: speciesData.vernacularName,
              redlist_category: sp.redlistCategory,
            };
          } catch {
            return {
              species_key: sp.speciesKey,
              occurrence_count: sp.count,
              canonicalName: sp.scientificName,
              redlist_category: sp.redlistCategory,
            };
          }
        }

        // Fall back to full GBIF API lookup if not in our cache
        try {
          const speciesResponse = await fetch(
            `https://api.gbif.org/v1/species/${sp.speciesKey}`
          );
          const speciesData = await speciesResponse.json();
          const canonicalName = speciesData.canonicalName || speciesData.scientificName;

          // Look up Red List category
          const normalizedName = canonicalName?.toLowerCase().trim();
          const redlist_category = normalizedName ? redListLookup.get(normalizedName) || null : null;

          return {
            species_key: sp.speciesKey,
            occurrence_count: sp.count,
            canonicalName,
            vernacularName: speciesData.vernacularName,
            redlist_category,
          };
        } catch {
          return {
            species_key: sp.speciesKey,
            occurrence_count: sp.count,
            canonicalName: `Species ${sp.speciesKey}`,
            redlist_category: null,
          };
        }
      })
    );

    return NextResponse.json({
      data: speciesWithNames,
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
        redlist: redlistStats,
      },
      country: countryCode,
    });
  } catch (error) {
    console.error("Error fetching country species:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}

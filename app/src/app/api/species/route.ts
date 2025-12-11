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

// Cache per taxon
const dataCache: Record<string, SpeciesRecord[]> = {};

// Red List lookup cache: scientific_name (lowercase) -> category
const redListCache: Record<string, Map<string, string>> = {};

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

async function loadData(taxonId: string): Promise<SpeciesRecord[]> {
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

    return dataCache[taxonId];
  } catch {
    // File doesn't exist for this taxon yet
    return [];
  }
}

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const taxonId = searchParams.get("taxon") || "plantae";
  const page = parseInt(searchParams.get("page") || "1", 10);
  const limit = Math.min(parseInt(searchParams.get("limit") || "100", 10), 1000);
  const minCount = parseInt(searchParams.get("minCount") || "0", 10);
  const maxCount = parseInt(searchParams.get("maxCount") || "999999999", 10);
  const sortOrder = searchParams.get("sort") || "desc";
  const redlistFilter = searchParams.get("redlist"); // "none" = not assessed, "assessed" = has assessment

  const data = await loadData(taxonId);

  // Filter by occurrence count range
  let filtered = data.filter(
    (d) => d.occurrence_count >= minCount && d.occurrence_count <= maxCount
  );

  // Filter by Red List status
  if (redlistFilter === "none") {
    filtered = filtered.filter((d) => !d.redlist_category);
  } else if (redlistFilter === "assessed") {
    filtered = filtered.filter((d) => d.redlist_category);
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

  // Calculate stats - cumulative thresholds to show data deficiency
  const assessed = data.filter((d) => d.redlist_category);
  const notAssessed = data.filter((d) => !d.redlist_category);

  const stats = {
    total: data.length,
    filtered: filtered.length,
    totalOccurrences: data.reduce((sum, d) => sum + d.occurrence_count, 0),
    median: data[Math.floor(data.length / 2)]?.occurrence_count || 0,
    distribution: {
      lte1: data.filter((d) => d.occurrence_count <= 1).length,
      lte10: data.filter((d) => d.occurrence_count <= 10).length,
      lte100: data.filter((d) => d.occurrence_count <= 100).length,
      lte1000: data.filter((d) => d.occurrence_count <= 1000).length,
      lte10000: data.filter((d) => d.occurrence_count <= 10000).length,
      lte100000: data.filter((d) => d.occurrence_count <= 100000).length,
      lte1000000: data.filter((d) => d.occurrence_count <= 1000000).length,
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

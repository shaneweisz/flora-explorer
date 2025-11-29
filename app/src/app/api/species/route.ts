import { NextRequest, NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";

interface SpeciesRecord {
  species_key: number;
  occurrence_count: number;
}

let cachedData: SpeciesRecord[] | null = null;

async function loadData(): Promise<SpeciesRecord[]> {
  if (cachedData) return cachedData;

  const filePath = path.join(process.cwd(), "public", "plant_species_counts.csv");
  const fileContent = await fs.readFile(filePath, "utf-8");
  const lines = fileContent.trim().split("\n");

  // Skip header
  cachedData = lines.slice(1).map((line) => {
    const [species_key, occurrence_count] = line.split(",");
    return {
      species_key: parseInt(species_key, 10),
      occurrence_count: parseInt(occurrence_count, 10),
    };
  });

  return cachedData;
}

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const page = parseInt(searchParams.get("page") || "1", 10);
  const limit = Math.min(parseInt(searchParams.get("limit") || "100", 10), 1000);
  const minCount = parseInt(searchParams.get("minCount") || "0", 10);
  const maxCount = parseInt(searchParams.get("maxCount") || "999999999", 10);
  const sortOrder = searchParams.get("sort") || "desc";

  const data = await loadData();

  // Filter by occurrence count range
  let filtered = data.filter(
    (d) => d.occurrence_count >= minCount && d.occurrence_count <= maxCount
  );

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

import { NextRequest, NextResponse } from "next/server";
import * as fs from "fs";
import * as path from "path";
import { getTaxonConfig } from "@/config/taxa";

interface YearRange {
  range: string;
  count: number;
  minYear: number;
  maxYear: number;
}

interface Species {
  assessment_date: string | null;
}

interface PrecomputedData {
  species: Species[];
  metadata: {
    totalSpecies: number;
    fetchedAt: string;
    taxonId?: string;
  };
}

// In-memory cache (keyed by taxon ID)
const cachedData: Map<string, PrecomputedData | null> = new Map();
const cacheLoadTimes: Map<string, number> = new Map();
const CACHE_RELOAD_INTERVAL = 60 * 60 * 1000; // Reload file every hour

function loadPrecomputedData(taxonId: string): PrecomputedData | null {
  const taxon = getTaxonConfig(taxonId);
  const dataPath = path.join(process.cwd(), "data", taxon.dataFile);

  try {
    if (!fs.existsSync(dataPath)) {
      console.warn(`Pre-computed data file not found: ${dataPath}`);
      return null;
    }

    const fileContent = fs.readFileSync(dataPath, "utf-8");
    return JSON.parse(fileContent) as PrecomputedData;
  } catch (error) {
    console.error(`Error loading pre-computed data for ${taxonId}:`, error);
    return null;
  }
}

function getSpeciesData(taxonId: string): PrecomputedData | null {
  const cacheTime = cacheLoadTimes.get(taxonId) || 0;
  if (!cachedData.has(taxonId) || Date.now() - cacheTime > CACHE_RELOAD_INTERVAL) {
    cachedData.set(taxonId, loadPrecomputedData(taxonId));
    cacheLoadTimes.set(taxonId, Date.now());
  }
  return cachedData.get(taxonId) || null;
}

function getYearRange(yearsSince: number): string {
  if (yearsSince <= 1) return "0-1 years";
  if (yearsSince <= 5) return "2-5 years";
  if (yearsSince <= 10) return "6-10 years";
  if (yearsSince <= 20) return "11-20 years";
  return "20+ years";
}

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const taxonId = searchParams.get("taxon") || "plantae";
  const taxon = getTaxonConfig(taxonId);

  const data = getSpeciesData(taxonId);

  if (!data) {
    return NextResponse.json(
      {
        error: `Species data not available for ${taxon.name}. Run: npx tsx scripts/fetch-redlist-species.ts ${taxonId}`,
      },
      { status: 503 }
    );
  }

  const currentYear = new Date().getFullYear();
  const yearCounts: Record<string, number> = {
    "0-1 years": 0,
    "2-5 years": 0,
    "6-10 years": 0,
    "11-20 years": 0,
    "20+ years": 0,
  };

  // Count species by year range (based on assessment_date)
  for (const species of data.species) {
    if (species.assessment_date) {
      const assessmentYear = new Date(species.assessment_date).getFullYear();
      if (!isNaN(assessmentYear)) {
        const yearsSince = currentYear - assessmentYear;
        const range = getYearRange(yearsSince);
        yearCounts[range]++;
      }
    }
  }

  // Build year ranges array in order
  const yearRanges: YearRange[] = [
    { range: "0-1 years", count: yearCounts["0-1 years"], minYear: 0, maxYear: 1 },
    { range: "2-5 years", count: yearCounts["2-5 years"], minYear: 2, maxYear: 5 },
    { range: "6-10 years", count: yearCounts["6-10 years"], minYear: 6, maxYear: 10 },
    { range: "11-20 years", count: yearCounts["11-20 years"], minYear: 11, maxYear: 20 },
    { range: "20+ years", count: yearCounts["20+ years"], minYear: 21, maxYear: 999 },
  ];

  return NextResponse.json({
    yearsSinceAssessment: yearRanges,
    sampleSize: data.metadata.totalSpecies,
    lastUpdated: data.metadata.fetchedAt,
    cached: true,
  });
}

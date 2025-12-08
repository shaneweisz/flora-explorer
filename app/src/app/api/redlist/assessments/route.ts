import { NextResponse } from "next/server";

interface YearRange {
  range: string;
  count: number;
  minYear: number;
  maxYear: number;
}

interface CachedAssessments {
  yearsSinceAssessment: YearRange[];
  sampleSize: number;
  lastUpdated: string;
}

// Cache for 24 hours
let cachedAssessments: CachedAssessments | null = null;
let cacheTime: number = 0;
const CACHE_DURATION = 24 * 60 * 60 * 1000; // 24 hours

// Delay helper for rate limiting
const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

async function fetchWithAuth(url: string): Promise<Response> {
  const apiKey = process.env.RED_LIST_API_KEY;
  if (!apiKey) {
    throw new Error("RED_LIST_API_KEY environment variable not set");
  }

  return fetch(url, {
    headers: {
      Authorization: `Bearer ${apiKey}`,
    },
  });
}

interface Assessment {
  year_published: string;
  sis_taxon_id: number;
}

interface ApiResponse {
  assessments: Assessment[];
}

function getYearRange(yearsSince: number): string {
  if (yearsSince <= 1) return "0-1 years";
  if (yearsSince <= 5) return "2-5 years";
  if (yearsSince <= 10) return "6-10 years";
  if (yearsSince <= 20) return "11-20 years";
  return "20+ years";
}

export async function GET() {
  // Return cached data if still valid
  if (cachedAssessments && Date.now() - cacheTime < CACHE_DURATION) {
    return NextResponse.json({ ...cachedAssessments, cached: true });
  }

  try {
    const currentYear = new Date().getFullYear();
    const yearCounts: Record<string, number> = {
      "0-1 years": 0,
      "2-5 years": 0,
      "6-10 years": 0,
      "11-20 years": 0,
      "20+ years": 0,
    };

    let totalSampled = 0;

    // Just 1 page = 100 species for fast loading
    const pagesToSample = [400]; // Middle of dataset for less bias

    for (let i = 0; i < pagesToSample.length; i++) {
      const page = pagesToSample[i];
      const response = await fetchWithAuth(
        `https://api.iucnredlist.org/api/v4/taxa/kingdom/Plantae?latest=true&page=${page}`
      );

      if (!response.ok) {
        throw new Error(`IUCN API error: ${response.statusText}`);
      }

      const data: ApiResponse = await response.json();
      const assessments = data.assessments || [];

      if (assessments.length === 0) {
        break; // No more data
      }

      for (const assessment of assessments) {
        const yearPublished = parseInt(assessment.year_published, 10);
        if (!isNaN(yearPublished)) {
          const yearsSince = currentYear - yearPublished;
          const range = getYearRange(yearsSince);
          yearCounts[range]++;
        }
        totalSampled++;
      }

      // Rate limiting: 500ms delay between requests
      if (i < pagesToSample.length - 1) {
        await delay(500);
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

    const result: CachedAssessments = {
      yearsSinceAssessment: yearRanges,
      sampleSize: totalSampled,
      lastUpdated: new Date().toISOString(),
    };

    // Cache the results
    cachedAssessments = result;
    cacheTime = Date.now();

    return NextResponse.json({ ...result, cached: false });
  } catch (error) {
    console.error("Error fetching assessment data:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}

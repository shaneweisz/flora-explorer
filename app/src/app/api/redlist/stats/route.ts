import { NextResponse } from "next/server";

// IUCN Red List category colors (official)
const IUCN_COLORS: Record<string, string> = {
  EX: "#000000", // Extinct - Black
  EW: "#542344", // Extinct in Wild - Purple
  CR: "#d81e05", // Critically Endangered - Red
  EN: "#fc7f3f", // Endangered - Orange
  VU: "#f9e814", // Vulnerable - Yellow
  NT: "#cce226", // Near Threatened - Yellow-green
  LC: "#60c659", // Least Concern - Green
  DD: "#d1d1c6", // Data Deficient - Gray
};

const IUCN_CATEGORY_NAMES: Record<string, string> = {
  EX: "Extinct",
  EW: "Extinct in the Wild",
  CR: "Critically Endangered",
  EN: "Endangered",
  VU: "Vulnerable",
  NT: "Near Threatened",
  LC: "Least Concern",
  DD: "Data Deficient",
};

// Category order for display (most threatened first)
const CATEGORY_ORDER = ["EX", "EW", "CR", "EN", "VU", "NT", "LC", "DD"];

interface CategoryStats {
  code: string;
  name: string;
  count: number;
  color: string;
}

interface CachedStats {
  totalAssessed: number;
  byCategory: CategoryStats[];
  sampleSize: number;
  lastUpdated: string;
}

// Cache for 24 hours
let cachedStats: CachedStats | null = null;
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
  red_list_category_code: string;
  year_published: string;
  sis_taxon_id: number;
}

interface ApiResponse {
  assessments: Assessment[];
}

export async function GET() {
  // Return cached data if still valid
  if (cachedStats && Date.now() - cacheTime < CACHE_DURATION) {
    return NextResponse.json({ ...cachedStats, cached: true });
  }

  try {
    // Fetch plant assessments from IUCN API
    // We'll sample pages to build category distribution
    const categoryCounts: Record<string, number> = {};
    CATEGORY_ORDER.forEach((cat) => (categoryCounts[cat] = 0));

    let totalSampled = 0;
    const pagesToFetch = 20; // Sample 20 pages = 2000 species

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
        const cat = assessment.red_list_category_code;
        if (cat && categoryCounts[cat] !== undefined) {
          categoryCounts[cat]++;
        }
        totalSampled++;
      }

      // Rate limiting: 500ms delay between requests
      if (i < pagesToSample.length - 1) {
        await delay(500);
      }
    }

    // Build category stats array in display order
    const byCategory: CategoryStats[] = CATEGORY_ORDER.map((code) => ({
      code,
      name: IUCN_CATEGORY_NAMES[code],
      count: categoryCounts[code],
      color: IUCN_COLORS[code],
    }));

    // We know the exact total from testing: 807 pages * 100 = ~80,650 species
    const estimatedTotal = 80650;

    const stats: CachedStats = {
      totalAssessed: estimatedTotal,
      byCategory,
      sampleSize: totalSampled,
      lastUpdated: new Date().toISOString(),
    };

    // Cache the results
    cachedStats = stats;
    cacheTime = Date.now();

    return NextResponse.json({ ...stats, cached: false });
  } catch (error) {
    console.error("Error fetching Red List stats:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}

import { NextRequest, NextResponse } from "next/server";

/**
 * Literature Since Assessment API
 *
 * Automatically finds literature published AFTER a species' last assessment date.
 * Combines results from:
 * - OpenAlex (primary): Scientific papers with DOIs, citations, abstracts
 * - Nosible (supplementary): Grey literature, news, NGO reports
 *
 * See: Weeknotes/Subpages/Nosible API Evaluation.md for comparison details
 */

interface OpenAlexWork {
  id: string;
  doi: string | null;
  title: string;
  publication_year: number | null;
  publication_date: string | null;
  cited_by_count: number;
  type: string;
  primary_location?: {
    source?: {
      display_name: string;
    };
  };
  abstract_inverted_index?: Record<string, number[]>;
  authorships?: Array<{
    author: {
      display_name: string;
    };
  }>;
}

interface OpenAlexResponse {
  meta: {
    count: number;
  };
  results: OpenAlexWork[];
}

interface LiteratureResult {
  title: string;
  url: string;
  doi: string | null;
  year: number | null;
  date: string | null;
  citations: number | null;
  source: string;
  sourceType: "academic" | "grey";
  abstract: string | null;
  authors: string | null;
}

// Reconstruct abstract from OpenAlex inverted index
function reconstructAbstract(invertedIndex: Record<string, number[]> | undefined, maxWords = 100): string | null {
  if (!invertedIndex) return null;

  const words: [number, string][] = [];
  for (const [word, positions] of Object.entries(invertedIndex)) {
    for (const pos of positions) {
      words.push([pos, word]);
    }
  }
  words.sort((a, b) => a[0] - b[0]);

  const reconstructed = words.slice(0, maxWords).map(([, word]) => word).join(" ");
  return reconstructed + (words.length > maxWords ? "..." : "");
}

// Search OpenAlex for papers published after a given year
async function searchOpenAlexSinceYear(
  scientificName: string,
  sinceYear: number,
  limit: number = 5
): Promise<{ count: number; results: LiteratureResult[] }> {
  // OpenAlex filter: publication_year > sinceYear, exclude datasets (GBIF occurrence downloads)
  // Sorted by most recent first
  // Note: per_page must be >= 1 for the API to work, even for count-only requests
  // Build URL manually to ensure proper encoding of negation filter
  // Exclude datasets (which includes GBIF Occurrence Downloads)
  const filter = encodeURIComponent(`publication_year:>${sinceYear},type:!dataset`);
  const search = encodeURIComponent(scientificName);
  const url = `https://api.openalex.org/works?search=${search}&filter=${filter}&sort=publication_date:desc&per_page=${Math.max(1, limit)}&mailto=red-list-dashboard@example.com`;

  const response = await fetch(url);
  if (!response.ok) {
    console.error("OpenAlex API error:", response.status);
    return { count: 0, results: [] };
  }

  const data: OpenAlexResponse = await response.json();

  const results = data.results.map((work) => ({
    title: work.title,
    url: work.doi ? `https://doi.org/${work.doi.replace("https://doi.org/", "")}` : work.id,
    doi: work.doi,
    year: work.publication_year,
    date: work.publication_date,
    citations: work.cited_by_count,
    source: work.primary_location?.source?.display_name || "Unknown",
    sourceType: "academic" as const,
    abstract: reconstructAbstract(work.abstract_inverted_index),
    authors: work.authorships?.slice(0, 3).map(a => a.author.display_name).join(", ") || null,
  }));

  return { count: data.meta.count, results };
}

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const scientificName = searchParams.get("scientificName");
  const assessmentYear = searchParams.get("assessmentYear");
  const limit = Math.min(parseInt(searchParams.get("limit") || "5"), 20);

  if (!scientificName) {
    return NextResponse.json(
      { error: "Query parameter 'scientificName' is required" },
      { status: 400 }
    );
  }

  if (!assessmentYear) {
    return NextResponse.json(
      { error: "Query parameter 'assessmentYear' is required" },
      { status: 400 }
    );
  }

  const sinceYear = parseInt(assessmentYear);
  if (isNaN(sinceYear)) {
    return NextResponse.json(
      { error: "Invalid assessmentYear" },
      { status: 400 }
    );
  }

  try {
    const { count, results } = await searchOpenAlexSinceYear(scientificName, sinceYear, limit);

    return NextResponse.json({
      scientificName,
      assessmentYear: sinceYear,
      totalPapersSinceAssessment: count,
      topPapers: results,
    });
  } catch (error) {
    console.error("Literature search error:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Search failed" },
      { status: 500 }
    );
  }
}

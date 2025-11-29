import { NextRequest, NextResponse } from "next/server";

// Cambridge bbox (same as test_classifier.py)
const CAMBRIDGE_BBOX = {
  minLon: -0.003,
  maxLon: 0.250,
  minLat: 52.092,
  maxLat: 52.318,
};

// GBIF kingdom key for Plantae
const PLANTAE_KINGDOM_KEY = 6;

interface SpeciesFacet {
  speciesKey: number;
  species: string;
  count: number;
}

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const page = parseInt(searchParams.get("page") || "1");
  const limit = parseInt(searchParams.get("limit") || "10");
  const minCount = parseInt(searchParams.get("minCount") || "0");
  const maxCount = parseInt(searchParams.get("maxCount") || "999999999");
  const sort = searchParams.get("sort") || "desc";

  try {
    // Use GBIF occurrence search with facets to get species counts
    // Fetch more than we need to allow filtering and pagination
    const params = new URLSearchParams({
      kingdomKey: PLANTAE_KINGDOM_KEY.toString(),
      geometry: `POLYGON((${CAMBRIDGE_BBOX.minLon} ${CAMBRIDGE_BBOX.minLat}, ${CAMBRIDGE_BBOX.maxLon} ${CAMBRIDGE_BBOX.minLat}, ${CAMBRIDGE_BBOX.maxLon} ${CAMBRIDGE_BBOX.maxLat}, ${CAMBRIDGE_BBOX.minLon} ${CAMBRIDGE_BBOX.maxLat}, ${CAMBRIDGE_BBOX.minLon} ${CAMBRIDGE_BBOX.minLat}))`,
      facet: "speciesKey",
      facetLimit: "500", // Get top 500 species
      limit: "0", // We only want facets, not actual occurrences
      hasCoordinate: "true",
      hasGeospatialIssue: "false",
    });

    const response = await fetch(
      `https://api.gbif.org/v1/occurrence/search?${params}`
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
          distribution: { lte1: 0, lte10: 0, lte100: 0, lte1000: 0, lte10000: 0 },
        },
        bbox: CAMBRIDGE_BBOX,
      });
    }

    // Convert facets to species records with counts
    const allSpecies: { speciesKey: number; count: number }[] = speciesFacets.counts.map(
      (facet: { name: string; count: number }) => ({
        speciesKey: parseInt(facet.name),
        count: facet.count,
      })
    );

    // Calculate stats from all species
    const totalOccurrences = allSpecies.reduce((sum, s) => sum + s.count, 0);
    const counts = allSpecies.map(s => s.count).sort((a, b) => a - b);
    const median = counts.length > 0 ? counts[Math.floor(counts.length / 2)] : 0;

    const distribution = {
      lte1: allSpecies.filter(s => s.count <= 1).length,
      lte10: allSpecies.filter(s => s.count <= 10).length,
      lte100: allSpecies.filter(s => s.count <= 100).length,
      lte1000: allSpecies.filter(s => s.count <= 1000).length,
      lte10000: allSpecies.filter(s => s.count <= 10000).length,
    };

    // Filter by count range
    let filteredSpecies = allSpecies.filter(
      s => s.count >= minCount && s.count <= maxCount
    );

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

    // Fetch species names for the page
    const speciesWithNames = await Promise.all(
      pageSpecies.map(async (sp) => {
        try {
          const speciesResponse = await fetch(
            `https://api.gbif.org/v1/species/${sp.speciesKey}`
          );
          const speciesData = await speciesResponse.json();
          return {
            species_key: sp.speciesKey,
            occurrence_count: sp.count,
            canonicalName: speciesData.canonicalName || speciesData.scientificName,
            vernacularName: speciesData.vernacularName,
          };
        } catch {
          return {
            species_key: sp.speciesKey,
            occurrence_count: sp.count,
            canonicalName: `Species ${sp.speciesKey}`,
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
      },
      bbox: CAMBRIDGE_BBOX,
    });
  } catch (error) {
    console.error("Error fetching Cambridge species:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}

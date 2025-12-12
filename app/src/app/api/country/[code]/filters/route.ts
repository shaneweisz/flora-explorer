import { NextRequest, NextResponse } from "next/server";
import { getTaxonConfig } from "@/config/taxa";

// Data source definitions
const DATA_SOURCES = {
  iNaturalist: { type: "dataset" as const, key: "50c9509d-22c7-4a22-a47d-8c48425ef4a7", label: "iNaturalist" },
  iRecord: { type: "publishingOrg" as const, key: "32f1b389-5871-4da3-832f-9a89132520c5", label: "iRecord" },
  BSBI: { type: "publishingOrg" as const, key: "aa569acf-991d-4467-b327-8442f30ddbd2", label: "BSBI" },
};

// Uncertainty bands in meters
const UNCERTAINTY_BANDS = [
  { key: "10", label: "≤ 10m", max: 10 },
  { key: "100", label: "≤ 100m", max: 100 },
  { key: "1000", label: "≤ 1km", max: 1000 },
  { key: "10000", label: "≤ 10km", max: 10000 },
];

// Basis of record types (main ones we show)
const BASIS_OF_RECORD_MAIN = [
  { key: "HUMAN_OBSERVATION", label: "Human Observation" },
  { key: "PRESERVED_SPECIMEN", label: "Preserved Specimen" },
  { key: "MACHINE_OBSERVATION", label: "Machine Observation" },
  { key: "MATERIAL_SAMPLE", label: "Material Sample" },
];

// Other basis of record types (grouped into "Other")
const BASIS_OF_RECORD_OTHER = [
  "OBSERVATION",
  "MATERIAL_CITATION",
  "OCCURRENCE",
  "LIVING_SPECIMEN",
  "FOSSIL_SPECIMEN",
];

interface FilterStats {
  basisOfRecord: { key: string; label: string; count: number }[];
  uncertainty: { key: string; label: string; count: number }[];
  dataSources: { key: string; label: string; count: number }[];
  total: number;
}

// Simple cache to avoid redundant API calls
const cache: Record<string, { data: FilterStats; timestamp: number }> = {};
const CACHE_DURATION = 5 * 60 * 1000; // 5 minutes

function buildTaxonParams(taxon: ReturnType<typeof getTaxonConfig>): URLSearchParams {
  const params = new URLSearchParams();

  if (taxon.gbifClassKey) {
    params.set("classKey", taxon.gbifClassKey.toString());
  } else if (taxon.gbifClassKeys && taxon.gbifClassKeys.length > 0) {
    taxon.gbifClassKeys.forEach(key => {
      params.append("classKey", key.toString());
    });
  } else if (taxon.gbifOrderKeys && taxon.gbifOrderKeys.length > 0) {
    taxon.gbifOrderKeys.forEach(key => {
      params.append("orderKey", key.toString());
    });
  } else if (taxon.gbifKingdomKey) {
    params.set("kingdomKey", taxon.gbifKingdomKey.toString());
  }

  return params;
}

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ code: string }> }
) {
  const { code } = await params;
  const countryCode = code.toUpperCase();

  const searchParams = request.nextUrl.searchParams;
  const taxonId = searchParams.get("taxon") || "plantae";

  const cacheKey = `${countryCode}-${taxonId}`;

  // Return cached data if valid
  if (cache[cacheKey] && Date.now() - cache[cacheKey].timestamp < CACHE_DURATION) {
    return NextResponse.json(cache[cacheKey].data);
  }

  const taxon = getTaxonConfig(taxonId);
  const taxonParams = buildTaxonParams(taxon);

  try {
    // Base params for all queries
    const baseParams = new URLSearchParams({
      country: countryCode,
      hasCoordinate: "true",
      hasGeospatialIssue: "false",
      limit: "0",
    });

    // Add taxon params
    taxonParams.forEach((value, key) => {
      baseParams.append(key, value);
    });

    // Fetch all stats in parallel
    const [
      totalResp,
      basisOfRecordResp,
      ...uncertaintyAndSourceResponses
    ] = await Promise.all([
      // Total count
      fetch(`https://api.gbif.org/v1/occurrence/search?${baseParams}`),
      // Basis of record facets
      fetch(`https://api.gbif.org/v1/occurrence/search?${baseParams}&facet=basisOfRecord&facetLimit=10`),
      // Uncertainty bands
      ...UNCERTAINTY_BANDS.map(band =>
        fetch(`https://api.gbif.org/v1/occurrence/search?${baseParams}&coordinateUncertaintyInMeters=*,${band.max}`)
      ),
      // Data sources
      ...Object.entries(DATA_SOURCES).map(([, source]) => {
        const sourceParams = new URLSearchParams(baseParams);
        if (source.type === "dataset") {
          sourceParams.set("datasetKey", source.key);
        } else {
          sourceParams.set("publishingOrg", source.key);
        }
        return fetch(`https://api.gbif.org/v1/occurrence/search?${sourceParams}`);
      }),
    ]);

    // Parse responses
    const totalData = await totalResp.json();
    const total = totalData.count || 0;

    const basisData = await basisOfRecordResp.json();
    const basisFacets = basisData.facets?.find((f: { field: string }) => f.field === "BASIS_OF_RECORD");

    // Map basis of record counts for main types
    const basisOfRecordStats = BASIS_OF_RECORD_MAIN.map(basis => {
      const facet = basisFacets?.counts?.find((c: { name: string }) => c.name === basis.key);
      return {
        key: basis.key,
        label: basis.label,
        count: facet?.count || 0,
      };
    });

    // Calculate "Other" count (sum of remaining basisOfRecord types)
    const otherCount = BASIS_OF_RECORD_OTHER.reduce((sum, type) => {
      const facet = basisFacets?.counts?.find((c: { name: string }) => c.name === type);
      return sum + (facet?.count || 0);
    }, 0);

    // Add "Other" if there are any
    if (otherCount > 0) {
      basisOfRecordStats.push({
        key: "OTHER",
        label: "Other",
        count: otherCount,
      });
    }

    // Parse uncertainty responses
    const uncertaintyStats = await Promise.all(
      UNCERTAINTY_BANDS.map(async (band, index) => {
        const resp = uncertaintyAndSourceResponses[index];
        const data = await resp.json();
        return {
          key: band.key,
          label: band.label,
          count: data.count || 0,
        };
      })
    );

    // Parse data source responses
    const sourceKeys = Object.keys(DATA_SOURCES);
    const dataSourceStats = await Promise.all(
      sourceKeys.map(async (sourceKey, index) => {
        const resp = uncertaintyAndSourceResponses[UNCERTAINTY_BANDS.length + index];
        const data = await resp.json();
        return {
          key: sourceKey,
          label: DATA_SOURCES[sourceKey as keyof typeof DATA_SOURCES].label,
          count: data.count || 0,
        };
      })
    );

    const stats: FilterStats = {
      basisOfRecord: basisOfRecordStats,
      uncertainty: uncertaintyStats,
      dataSources: dataSourceStats,
      total,
    };

    // Cache the result
    cache[cacheKey] = { data: stats, timestamp: Date.now() };

    return NextResponse.json(stats);
  } catch (error) {
    console.error("Error fetching filter stats:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}

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

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const taxonId = searchParams.get("taxon") || "plantae";
  const country = searchParams.get("country"); // Optional - if not provided, returns global stats

  // Current filter selections (for dynamic/linked charts)
  const currentBasisOfRecord = searchParams.get("basisOfRecord");
  const currentMaxUncertainty = searchParams.get("maxUncertainty");
  const currentDataSource = searchParams.get("dataSource");

  // Build cache key including current filters
  const cacheKey = `${country || 'global'}-${taxonId}-${currentBasisOfRecord || ''}-${currentMaxUncertainty || ''}-${currentDataSource || ''}`;

  // Return cached data if valid
  if (cache[cacheKey] && Date.now() - cache[cacheKey].timestamp < CACHE_DURATION) {
    return NextResponse.json(cache[cacheKey].data);
  }

  const taxon = getTaxonConfig(taxonId);
  const taxonParams = buildTaxonParams(taxon);

  try {
    // Base params for all queries
    const baseParams = new URLSearchParams({
      hasCoordinate: "true",
      hasGeospatialIssue: "false",
      limit: "0",
    });

    // Add country if specified
    if (country) {
      baseParams.set("country", country.toUpperCase());
    }

    // Add taxon params
    taxonParams.forEach((value, key) => {
      baseParams.append(key, value);
    });

    // Build params with current filters applied (for dynamic counts)
    const buildFilteredParams = (excludeFilter: "basisOfRecord" | "uncertainty" | "dataSource") => {
      const params = new URLSearchParams(baseParams);

      // Add basisOfRecord filter if set and not excluded
      if (currentBasisOfRecord && excludeFilter !== "basisOfRecord") {
        params.set("basisOfRecord", currentBasisOfRecord);
      }

      // Add uncertainty filter if set and not excluded
      if (currentMaxUncertainty && excludeFilter !== "uncertainty") {
        params.set("coordinateUncertaintyInMeters", `*,${currentMaxUncertainty}`);
      }

      // Add data source filter if set and not excluded
      if (currentDataSource && excludeFilter !== "dataSource") {
        const source = DATA_SOURCES[currentDataSource as keyof typeof DATA_SOURCES];
        if (source) {
          if (source.type === "dataset") {
            params.set("datasetKey", source.key);
          } else {
            params.set("publishingOrg", source.key);
          }
        }
      }

      return params;
    };

    // Params for basisOfRecord counts (apply uncertainty + dataSource filters)
    const basisParams = buildFilteredParams("basisOfRecord");

    // Params for uncertainty counts (apply basisOfRecord + dataSource filters)
    const uncertaintyBaseParams = buildFilteredParams("uncertainty");

    // Params for dataSource counts (apply basisOfRecord + uncertainty filters)
    const dataSourceBaseParams = buildFilteredParams("dataSource");

    // Params for total (apply all current filters)
    const totalParams = new URLSearchParams(baseParams);
    if (currentBasisOfRecord) totalParams.set("basisOfRecord", currentBasisOfRecord);
    if (currentMaxUncertainty) totalParams.set("coordinateUncertaintyInMeters", `*,${currentMaxUncertainty}`);
    if (currentDataSource) {
      const source = DATA_SOURCES[currentDataSource as keyof typeof DATA_SOURCES];
      if (source) {
        if (source.type === "dataset") {
          totalParams.set("datasetKey", source.key);
        } else {
          totalParams.set("publishingOrg", source.key);
        }
      }
    }

    // Fetch all stats in parallel
    const [
      totalResp,
      basisOfRecordResp,
      ...uncertaintyAndSourceResponses
    ] = await Promise.all([
      // Total count (with all current filters)
      fetch(`https://api.gbif.org/v1/occurrence/search?${totalParams}`),
      // Basis of record facets (with uncertainty + dataSource filters)
      fetch(`https://api.gbif.org/v1/occurrence/search?${basisParams}&facet=basisOfRecord&facetLimit=10`),
      // Uncertainty bands (with basisOfRecord + dataSource filters)
      ...UNCERTAINTY_BANDS.map(band => {
        const params = new URLSearchParams(uncertaintyBaseParams);
        params.set("coordinateUncertaintyInMeters", `*,${band.max}`);
        return fetch(`https://api.gbif.org/v1/occurrence/search?${params}`);
      }),
      // Data sources (with basisOfRecord + uncertainty filters)
      ...Object.entries(DATA_SOURCES).map(([, source]) => {
        const params = new URLSearchParams(dataSourceBaseParams);
        if (source.type === "dataset") {
          params.set("datasetKey", source.key);
        } else {
          params.set("publishingOrg", source.key);
        }
        return fetch(`https://api.gbif.org/v1/occurrence/search?${params}`);
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

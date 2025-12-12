import { NextRequest, NextResponse } from "next/server";

interface InatObservation {
  url: string;
  date: string | null;
  imageUrl: string | null;
  location: string | null;
  observer: string | null;
}

interface RecordTypeBreakdown {
  humanObservation: number;
  preservedSpecimen: number;
  machineObservation: number;
  other: number;
  iNaturalist: number;
  recentInatObservations: InatObservation[];
  inatTotalCount: number;
  total: number;
}

// Cache breakdown results for 1 hour
const cache: Record<string, { data: RecordTypeBreakdown; timestamp: number }> = {};
const CACHE_DURATION = 60 * 60 * 1000; // 1 hour

// iNaturalist dataset key in GBIF
const INAT_DATASET_KEY = "50c9509d-22c7-4a22-a47d-8c48425ef4a7";

// Data source keys
const DATA_SOURCES: Record<string, { type: "dataset" | "publishingOrg"; key: string }> = {
  iNaturalist: { type: "dataset", key: "50c9509d-22c7-4a22-a47d-8c48425ef4a7" },
  iRecord: { type: "publishingOrg", key: "32f1b389-5871-4da3-832f-9a89132520c5" },
  BSBI: { type: "publishingOrg", key: "aa569acf-991d-4467-b327-8442f30ddbd2" },
};

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ key: string }> }
) {
  const { key } = await params;
  const speciesKey = key;
  const searchParams = request.nextUrl.searchParams;
  const country = searchParams.get("country");

  // Get current filters to match the species list query
  const maxUncertainty = searchParams.get("maxUncertainty");
  const dataSource = searchParams.get("dataSource");

  // Build cache key including filters
  const cacheKey = `${speciesKey}-${country || "global"}-${maxUncertainty || ""}-${dataSource || ""}`;

  // Return cached data if valid
  if (cache[cacheKey] && Date.now() - cache[cacheKey].timestamp < CACHE_DURATION) {
    return NextResponse.json(cache[cacheKey].data);
  }

  try {
    // Build base params using occurrence/search with limit=0 to get counts
    // This supports hasCoordinate and hasGeospatialIssue unlike /count endpoint
    const buildParams = (extraParams: Record<string, string> = {}) => {
      const params = new URLSearchParams({
        taxonKey: speciesKey,
        hasCoordinate: "true",
        hasGeospatialIssue: "false",
        limit: "0",
        ...extraParams,
      });

      if (country) {
        params.set("country", country.toUpperCase());
      }

      // Add uncertainty filter if specified
      if (maxUncertainty) {
        params.set("coordinateUncertaintyInMeters", `*,${maxUncertainty}`);
      }

      // Add data source filter if specified
      if (dataSource && DATA_SOURCES[dataSource]) {
        const source = DATA_SOURCES[dataSource];
        if (source.type === "dataset") {
          params.set("datasetKey", source.key);
        } else {
          params.set("publishingOrg", source.key);
        }
      }

      return params;
    };

    // Fetch counts for each basisOfRecord type in parallel
    const [humanResp, specimenResp, machineResp, inatResp, inatRecentResp, totalResp] = await Promise.all([
      fetch(`https://api.gbif.org/v1/occurrence/search?${buildParams({ basisOfRecord: "HUMAN_OBSERVATION" })}`),
      fetch(`https://api.gbif.org/v1/occurrence/search?${buildParams({ basisOfRecord: "PRESERVED_SPECIMEN" })}`),
      fetch(`https://api.gbif.org/v1/occurrence/search?${buildParams({ basisOfRecord: "MACHINE_OBSERVATION" })}`),
      // iNaturalist count (with current filters)
      fetch(`https://api.gbif.org/v1/occurrence/search?${buildParams({ datasetKey: INAT_DATASET_KEY })}`),
      // Recent iNaturalist observations (up to 10 for preview)
      fetch(`https://api.gbif.org/v1/occurrence/search?${new URLSearchParams({
        taxonKey: speciesKey,
        datasetKey: INAT_DATASET_KEY,
        hasCoordinate: "true",
        limit: "10",
        ...(country ? { country: country.toUpperCase() } : {}),
      })}`),
      // Total count (with all filters)
      fetch(`https://api.gbif.org/v1/occurrence/search?${buildParams()}`),
    ]);

    const [humanData, specimenData, machineData, inatData, totalData] = await Promise.all([
      humanResp.json(),
      specimenResp.json(),
      machineResp.json(),
      inatResp.json(),
      totalResp.json(),
    ]);

    const humanCount = humanData.count || 0;
    const specimenCount = specimenData.count || 0;
    const machineCount = machineData.count || 0;
    const inatCount = inatData.count || 0;
    const totalCount = totalData.count || 0;

    const otherCount = Math.max(0, totalCount - humanCount - specimenCount - machineCount);

    // Parse recent iNaturalist observations
    let recentInatObservations: InatObservation[] = [];
    if (inatRecentResp.ok) {
      const inatRecentData = await inatRecentResp.json();
      if (inatRecentData.results && inatRecentData.results.length > 0) {
        recentInatObservations = inatRecentData.results
          .filter((obs: { references?: string }) => obs.references)
          .map((obs: {
            references: string;
            eventDate?: string;
            media?: { identifier?: string }[];
            verbatimLocality?: string;
            stateProvince?: string;
            country?: string;
            recordedBy?: string;
          }) => {
            const imageUrl = obs.media?.[0]?.identifier || null;
            const locationParts = [obs.verbatimLocality, obs.stateProvince, obs.country].filter(Boolean);
            const location = locationParts.length > 0 ? locationParts.join(', ') : null;
            return {
              url: obs.references,
              date: obs.eventDate ? obs.eventDate.split('T')[0] : null,
              imageUrl,
              location,
              observer: obs.recordedBy || null,
            };
          });
      }
    }

    const breakdown: RecordTypeBreakdown = {
      humanObservation: humanCount,
      preservedSpecimen: specimenCount,
      machineObservation: machineCount,
      other: otherCount,
      iNaturalist: inatCount,
      recentInatObservations,
      inatTotalCount: inatCount,
      total: totalCount,
    };

    // Cache the result
    cache[cacheKey] = { data: breakdown, timestamp: Date.now() };

    return NextResponse.json(breakdown);
  } catch (error) {
    console.error("Error fetching record breakdown:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}

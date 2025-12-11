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
}

// Cache breakdown results for 1 hour
const cache: Record<string, { data: RecordTypeBreakdown; timestamp: number }> = {};
const CACHE_DURATION = 60 * 60 * 1000; // 1 hour

// iNaturalist dataset key in GBIF
const INAT_DATASET_KEY = "50c9509d-22c7-4a22-a47d-8c48425ef4a7";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ key: string }> }
) {
  const { key } = await params;
  const speciesKey = key;
  const searchParams = request.nextUrl.searchParams;
  const country = searchParams.get("country");

  const cacheKey = country ? `${speciesKey}-${country}` : speciesKey;

  // Return cached data if valid
  if (cache[cacheKey] && Date.now() - cache[cacheKey].timestamp < CACHE_DURATION) {
    return NextResponse.json(cache[cacheKey].data);
  }

  try {
    // Build base params - note: the count endpoint only supports taxonKey, country, basisOfRecord, datasetKey
    // It does NOT support hasCoordinate or hasGeospatialIssue
    const baseParams: Record<string, string> = {
      taxonKey: speciesKey,
    };

    if (country) {
      baseParams.country = country.toUpperCase();
    }

    // Fetch counts for each basisOfRecord type in parallel, plus iNat observations
    const [humanResp, specimenResp, machineResp, inatResp, inatRecentResp, totalResp] = await Promise.all([
      fetch(`https://api.gbif.org/v1/occurrence/count?${new URLSearchParams({
        ...baseParams,
        basisOfRecord: "HUMAN_OBSERVATION",
      })}`),
      fetch(`https://api.gbif.org/v1/occurrence/count?${new URLSearchParams({
        ...baseParams,
        basisOfRecord: "PRESERVED_SPECIMEN",
      })}`),
      fetch(`https://api.gbif.org/v1/occurrence/count?${new URLSearchParams({
        ...baseParams,
        basisOfRecord: "MACHINE_OBSERVATION",
      })}`),
      // iNaturalist count
      fetch(`https://api.gbif.org/v1/occurrence/count?${new URLSearchParams({
        ...baseParams,
        datasetKey: INAT_DATASET_KEY,
      })}`),
      // Recent iNaturalist observations (up to 5 for navigation)
      fetch(`https://api.gbif.org/v1/occurrence/search?${new URLSearchParams({
        ...baseParams,
        datasetKey: INAT_DATASET_KEY,
        limit: "5",
      })}`),
      // Total count
      fetch(`https://api.gbif.org/v1/occurrence/count?${new URLSearchParams(baseParams)}`),
    ]);

    const [humanCount, specimenCount, machineCount, inatCount, totalCount] = await Promise.all([
      humanResp.ok ? parseInt(await humanResp.text(), 10) || 0 : 0,
      specimenResp.ok ? parseInt(await specimenResp.text(), 10) || 0 : 0,
      machineResp.ok ? parseInt(await machineResp.text(), 10) || 0 : 0,
      inatResp.ok ? parseInt(await inatResp.text(), 10) || 0 : 0,
      totalResp.ok ? parseInt(await totalResp.text(), 10) || 0 : 0,
    ]);

    const otherCount = Math.max(0, totalCount - humanCount - specimenCount - machineCount);

    // Parse recent iNaturalist observations
    let recentInatObservations: InatObservation[] = [];
    if (inatRecentResp.ok) {
      const inatData = await inatRecentResp.json();
      if (inatData.results && inatData.results.length > 0) {
        recentInatObservations = inatData.results
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

import { NextRequest, NextResponse } from "next/server";

interface RecordTypeBreakdown {
  humanObservation: number;
  preservedSpecimen: number;
  machineObservation: number;
  other: number;
  iNaturalist: number;
}

// Cache breakdown results for 1 hour
const cache: Record<string, { data: RecordTypeBreakdown; timestamp: number }> = {};
const CACHE_DURATION = 60 * 60 * 1000; // 1 hour

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

    // Fetch counts for each basisOfRecord type in parallel
    const [humanResp, specimenResp, machineResp, inatResp] = await Promise.all([
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
      // iNaturalist dataset key
      fetch(`https://api.gbif.org/v1/occurrence/count?${new URLSearchParams({
        ...baseParams,
        datasetKey: "50c9509d-22c7-4a22-a47d-8c48425ef4a7",
      })}`),
    ]);

    const [humanCount, specimenCount, machineCount, inatCount] = await Promise.all([
      humanResp.ok ? parseInt(await humanResp.text(), 10) || 0 : 0,
      specimenResp.ok ? parseInt(await specimenResp.text(), 10) || 0 : 0,
      machineResp.ok ? parseInt(await machineResp.text(), 10) || 0 : 0,
      inatResp.ok ? parseInt(await inatResp.text(), 10) || 0 : 0,
    ]);

    // Get total to calculate "other"
    const totalResp = await fetch(`https://api.gbif.org/v1/occurrence/count?${new URLSearchParams(baseParams)}`);
    const totalCount = totalResp.ok ? parseInt(await totalResp.text(), 10) || 0 : 0;
    const otherCount = Math.max(0, totalCount - humanCount - specimenCount - machineCount);

    const breakdown: RecordTypeBreakdown = {
      humanObservation: humanCount,
      preservedSpecimen: specimenCount,
      machineObservation: machineCount,
      other: otherCount,
      iNaturalist: inatCount,
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

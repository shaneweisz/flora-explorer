import { NextRequest, NextResponse } from "next/server";

// Cambridge bbox
const CAMBRIDGE_BBOX = {
  minLon: -0.003,
  maxLon: 0.250,
  minLat: 52.092,
  maxLat: 52.318,
};

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const speciesKey = searchParams.get("speciesKey");
  const limit = parseInt(searchParams.get("limit") || "300");

  if (!speciesKey) {
    return NextResponse.json(
      { error: "speciesKey parameter is required" },
      { status: 400 }
    );
  }

  try {
    const params = new URLSearchParams({
      speciesKey,
      geometry: `POLYGON((${CAMBRIDGE_BBOX.minLon} ${CAMBRIDGE_BBOX.minLat}, ${CAMBRIDGE_BBOX.maxLon} ${CAMBRIDGE_BBOX.minLat}, ${CAMBRIDGE_BBOX.maxLon} ${CAMBRIDGE_BBOX.maxLat}, ${CAMBRIDGE_BBOX.minLon} ${CAMBRIDGE_BBOX.maxLat}, ${CAMBRIDGE_BBOX.minLon} ${CAMBRIDGE_BBOX.minLat}))`,
      hasCoordinate: "true",
      hasGeospatialIssue: "false",
      limit: limit.toString(),
    });

    const response = await fetch(
      `https://api.gbif.org/v1/occurrence/search?${params}`
    );

    if (!response.ok) {
      throw new Error(`GBIF API error: ${response.statusText}`);
    }

    const data = await response.json();

    // Convert to GeoJSON
    const features = data.results
      .filter((r: { decimalLatitude?: number; decimalLongitude?: number }) =>
        r.decimalLatitude && r.decimalLongitude
      )
      .map((r: {
        key: number;
        species?: string;
        scientificName?: string;
        eventDate?: string;
        recordedBy?: string;
        decimalLongitude: number;
        decimalLatitude: number;
      }) => ({
        type: "Feature",
        properties: {
          gbifID: r.key,
          species: r.species || r.scientificName,
          eventDate: r.eventDate,
          recordedBy: r.recordedBy,
        },
        geometry: {
          type: "Point",
          coordinates: [r.decimalLongitude, r.decimalLatitude],
        },
      }));

    return NextResponse.json({
      type: "FeatureCollection",
      features,
      metadata: {
        speciesKey: parseInt(speciesKey),
        count: features.length,
        total: data.count,
        bbox: [CAMBRIDGE_BBOX.minLon, CAMBRIDGE_BBOX.minLat, CAMBRIDGE_BBOX.maxLon, CAMBRIDGE_BBOX.maxLat],
      },
    });
  } catch (error) {
    console.error("Error fetching occurrences:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}

import { NextRequest, NextResponse } from "next/server";

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const speciesKey = searchParams.get("speciesKey");
  const limit = parseInt(searchParams.get("limit") || "500");

  if (!speciesKey) {
    return NextResponse.json(
      { error: "speciesKey parameter is required" },
      { status: 400 }
    );
  }

  try {
    const params = new URLSearchParams({
      speciesKey,
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
        country?: string;
      }) => ({
        type: "Feature",
        properties: {
          gbifID: r.key,
          species: r.species || r.scientificName,
          eventDate: r.eventDate,
          recordedBy: r.recordedBy,
          country: r.country,
        },
        geometry: {
          type: "Point",
          coordinates: [r.decimalLongitude, r.decimalLatitude],
        },
      }));

    // Calculate bbox from features
    let minLon = Infinity, maxLon = -Infinity;
    let minLat = Infinity, maxLat = -Infinity;

    for (const feature of features) {
      const [lon, lat] = feature.geometry.coordinates;
      minLon = Math.min(minLon, lon);
      maxLon = Math.max(maxLon, lon);
      minLat = Math.min(minLat, lat);
      maxLat = Math.max(maxLat, lat);
    }

    return NextResponse.json({
      type: "FeatureCollection",
      features,
      metadata: {
        speciesKey: parseInt(speciesKey),
        count: features.length,
        total: data.count,
        bbox: features.length > 0 ? [minLon, minLat, maxLon, maxLat] : null,
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

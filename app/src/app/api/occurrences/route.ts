import { NextRequest, NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";

// Cache for loaded occurrence files
const occurrenceCache: Record<string, object> = {};

async function loadOccurrencesFromFile(species: string): Promise<object | null> {
  const cacheKey = species.toLowerCase().replace(/\s+/g, "_");

  if (occurrenceCache[cacheKey]) {
    return occurrenceCache[cacheKey];
  }

  // Look for occurrence file in data/occurrences directory
  const dataDir = path.join(process.cwd(), "..", "data", "occurrences");
  const filename = `${cacheKey}_cambridge.geojson`;
  const filePath = path.join(dataDir, filename);

  try {
    const content = await fs.readFile(filePath, "utf-8");
    const geojson = JSON.parse(content);
    occurrenceCache[cacheKey] = geojson;
    return geojson;
  } catch {
    return null;
  }
}

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const speciesKey = searchParams.get("speciesKey");
  const species = searchParams.get("species");
  const country = searchParams.get("country");
  const limit = parseInt(searchParams.get("limit") || "500");

  // If species name is provided, try to load from local file first
  if (species) {
    const localData = await loadOccurrencesFromFile(species);
    if (localData) {
      return NextResponse.json(localData);
    }
  }

  if (!speciesKey) {
    return NextResponse.json(
      { error: "speciesKey or species parameter is required" },
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

    // Add country filter if provided
    if (country) {
      params.set("country", country.toUpperCase());
    }

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
        basisOfRecord?: string;
      }) => ({
        type: "Feature",
        properties: {
          gbifID: r.key,
          species: r.species || r.scientificName,
          eventDate: r.eventDate,
          recordedBy: r.recordedBy,
          country: r.country,
          basisOfRecord: r.basisOfRecord,
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

import { NextRequest, NextResponse } from "next/server";
import { spawn } from "child_process";
import path from "path";

interface PredictionPoint {
  lon: number;
  lat: number;
  score: number;
}

interface PredictionResult {
  predictions: PredictionPoint[];
  species: string;
  species_key: number;
  center: { lon: number; lat: number };
  grid_size_m: number;
  n_pixels: number;
}

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const lat = parseFloat(searchParams.get("lat") || "");
  const lon = parseFloat(searchParams.get("lon") || "");
  const speciesKey = parseInt(searchParams.get("speciesKey") || "");
  const gridSize = parseInt(searchParams.get("gridSize") || "100"); // meters

  if (isNaN(lat) || isNaN(lon) || isNaN(speciesKey)) {
    return NextResponse.json(
      { error: "Missing required parameters: lat, lon, speciesKey" },
      { status: 400 }
    );
  }

  // Call the Python script to get predictions
  const projectRoot = path.join(process.cwd(), "..");
  const scriptPath = path.join(projectRoot, "predict_local.py");

  try {
    const result = await new Promise<PredictionResult>((resolve, reject) => {
      const proc = spawn("python3", [
        scriptPath,
        "--lat", lat.toString(),
        "--lon", lon.toString(),
        "--species-key", speciesKey.toString(),
        "--grid-size", gridSize.toString(),
      ], {
        cwd: projectRoot,
      });

      let stdout = "";
      let stderr = "";

      proc.stdout.on("data", (data) => {
        stdout += data.toString();
      });

      proc.stderr.on("data", (data) => {
        stderr += data.toString();
      });

      proc.on("close", (code) => {
        if (code !== 0) {
          reject(new Error(`Python script failed: ${stderr}`));
        } else {
          try {
            const result = JSON.parse(stdout);
            resolve(result);
          } catch (e) {
            reject(new Error(`Failed to parse output: ${stdout}`));
          }
        }
      });

      proc.on("error", (err) => {
        reject(err);
      });
    });

    return NextResponse.json(result);
  } catch (error) {
    console.error("Prediction error:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Prediction failed" },
      { status: 500 }
    );
  }
}

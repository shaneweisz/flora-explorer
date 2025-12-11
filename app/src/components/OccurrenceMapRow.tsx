"use client";

import { useState, useEffect } from "react";
import dynamic from "next/dynamic";

// Dynamically import Leaflet components
const MapContainer = dynamic(
  () => import("react-leaflet").then((mod) => mod.MapContainer),
  { ssr: false }
);
const TileLayer = dynamic(
  () => import("react-leaflet").then((mod) => mod.TileLayer),
  { ssr: false }
);
const CircleMarker = dynamic(
  () => import("react-leaflet").then((mod) => mod.CircleMarker),
  { ssr: false }
);
const Popup = dynamic(
  () => import("react-leaflet").then((mod) => mod.Popup),
  { ssr: false }
);
const LocateControl = dynamic(
  () => import("./LocateControl"),
  { ssr: false }
);

interface OccurrenceFeature {
  type: "Feature";
  properties: {
    gbifID: number;
    species: string;
    eventDate?: string;
    basisOfRecord?: string;
  };
  geometry: {
    type: "Point";
    coordinates: [number, number];
  };
}

// Format basisOfRecord to human-readable string
function formatBasisOfRecord(basis?: string): string {
  if (!basis) return "";
  const labels: Record<string, string> = {
    HUMAN_OBSERVATION: "Human observation",
    PRESERVED_SPECIMEN: "Preserved specimen",
    MACHINE_OBSERVATION: "Machine observation",
    FOSSIL_SPECIMEN: "Fossil specimen",
    LIVING_SPECIMEN: "Living specimen",
    MATERIAL_SAMPLE: "Material sample",
    OCCURRENCE: "Occurrence",
    MATERIAL_CITATION: "Material citation",
  };
  return labels[basis] || basis.replace(/_/g, " ").toLowerCase();
}

interface CandidateFeature {
  type: "Feature";
  properties: {
    probability: number;
  };
  geometry: {
    type: "Point";
    coordinates: [number, number];
  };
}

interface OccurrenceMapRowProps {
  speciesKey: number;
  speciesName?: string;
  countryCode?: string | null;
  mounted: boolean;
  colSpan: number;
  hasCandidates?: boolean;
}

// Color scale from grey (low prob) to red (high prob)
function getProbabilityColor(probability: number): string {
  const r = Math.round(180 + (220 - 180) * probability);
  const g = Math.round(180 - (180 - 38) * probability);
  const b = Math.round(180 - (180 - 38) * probability);
  return `rgb(${r}, ${g}, ${b})`;
}

export default function OccurrenceMapRow({
  speciesKey,
  speciesName,
  countryCode,
  mounted,
  colSpan,
  hasCandidates,
}: OccurrenceMapRowProps) {
  const [occurrences, setOccurrences] = useState<OccurrenceFeature[]>([]);
  const [candidates, setCandidates] = useState<CandidateFeature[]>([]);
  const [loadingOccurrences, setLoadingOccurrences] = useState(true);
  const [loadingCandidates, setLoadingCandidates] = useState(false);
  const [showCandidates, setShowCandidates] = useState(true);
  const [heatmapOpacity, setHeatmapOpacity] = useState(0.7);

  // Fetch occurrences immediately
  useEffect(() => {
    setLoadingOccurrences(true);
    const params = new URLSearchParams({
      speciesKey: speciesKey.toString(),
      limit: "500",
    });
    if (countryCode) {
      params.set("country", countryCode);
    }
    fetch(`/api/occurrences?${params}`)
      .then((res) => res.json())
      .then((data) => setOccurrences(data.features || []))
      .catch(console.error)
      .finally(() => setLoadingOccurrences(false));
  }, [speciesKey, countryCode]);

  // Fetch ALL candidates (no threshold) for heatmap
  useEffect(() => {
    if (!hasCandidates || !speciesName) return;

    setLoadingCandidates(true);
    fetch(`/api/candidates?species=${encodeURIComponent(speciesName)}&minProb=0`)
      .then((res) => res.json())
      .then((data) => {
        if (!data.error) {
          setCandidates(data.features || []);
        }
      })
      .catch(console.error)
      .finally(() => setLoadingCandidates(false));
  }, [speciesName, hasCandidates]);

  // Sort candidates by probability (low first so high prob renders on top)
  const sortedCandidates = [...candidates].sort(
    (a, b) => a.properties.probability - b.properties.probability
  );

  return (
    <tr>
      <td colSpan={colSpan} className="p-0">
        <div className="bg-zinc-50 dark:bg-zinc-800/50 border-t border-zinc-200 dark:border-zinc-700">
          <div className="p-2">
            {/* Controls for candidates */}
            {hasCandidates && (
              <div className="flex flex-wrap items-center gap-4 mb-2 p-2 bg-white dark:bg-zinc-900 rounded-lg border border-zinc-200 dark:border-zinc-700">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={showCandidates}
                    onChange={(e) => setShowCandidates(e.target.checked)}
                    className="w-4 h-4 rounded accent-orange-500"
                  />
                  <span className="text-sm text-zinc-700 dark:text-zinc-300">
                    Show heatmap ({loadingCandidates ? "..." : candidates.length}{" "}
                    points)
                  </span>
                </label>
                {showCandidates && (
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-zinc-500">Opacity:</span>
                    <input
                      type="range"
                      min="0.1"
                      max="1"
                      step="0.1"
                      value={heatmapOpacity}
                      onChange={(e) => setHeatmapOpacity(parseFloat(e.target.value))}
                      className="w-20 accent-orange-500"
                    />
                  </div>
                )}
                <div className="flex items-center gap-3 ml-auto text-xs">
                  <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded-full bg-blue-500 border-2 border-blue-700" />
                    <span className="text-zinc-500">GBIF record</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div
                      className="w-12 h-3 rounded"
                      style={{
                        background:
                          "linear-gradient(to right, rgb(180,180,180), rgb(220,38,38))",
                      }}
                    />
                    <span className="text-zinc-500">Low → High prob</span>
                  </div>
                </div>
              </div>
            )}

            {/* Map */}
            <div className="h-[300px] md:h-[400px] rounded-lg overflow-hidden border border-zinc-200 dark:border-zinc-700 relative">
              {loadingOccurrences ? (
                <div className="flex items-center justify-center h-full bg-zinc-100 dark:bg-zinc-800">
                  <div className="text-zinc-400">Loading occurrences...</div>
                </div>
              ) : mounted ? (
                <MapContainer
                  center={[20, 0]}
                  zoom={2}
                  style={{ height: "100%", width: "100%" }}
                >
                  <TileLayer
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                  />
                  <LocateControl />
                  {/* Render candidates as heatmap (sorted low-to-high so high prob on top) */}
                  {showCandidates &&
                    sortedCandidates.map((feature, idx) => {
                      const [lon, lat] = feature.geometry.coordinates;
                      const prob = feature.properties.probability;
                      const color = getProbabilityColor(prob);
                      const opacity = 0.2 + prob * 0.8 * heatmapOpacity;
                      return (
                        <CircleMarker
                          key={`candidate-${idx}`}
                          center={[lat, lon]}
                          radius={6}
                          pathOptions={{
                            color: "transparent",
                            fillColor: color,
                            fillOpacity: opacity,
                            weight: 0,
                          }}
                        >
                          <Popup>
                            <div className="text-sm">
                              <div className="font-medium text-orange-600">
                                Predicted Location
                              </div>
                              <div>Probability: {(prob * 100).toFixed(1)}%</div>
                              <div className="text-xs text-gray-500">
                                {lat.toFixed(4)}, {lon.toFixed(4)}
                              </div>
                            </div>
                          </Popup>
                        </CircleMarker>
                      );
                    })}
                  {/* Render occurrences on top with distinct style */}
                  {occurrences.map((feature, idx) => {
                    const [lon, lat] = feature.geometry.coordinates;
                    return (
                      <CircleMarker
                        key={feature.properties.gbifID || idx}
                        center={[lat, lon]}
                        radius={5}
                        pathOptions={{
                          color: "#1d4ed8",
                          fillColor: "#3b82f6",
                          fillOpacity: 0.9,
                          weight: 2,
                        }}
                      >
                        <Popup>
                          <div className="text-sm">
                            <div className="font-medium italic">
                              {feature.properties.species}
                            </div>
                            {feature.properties.basisOfRecord && (
                              <div className="text-xs text-gray-600">
                                {formatBasisOfRecord(feature.properties.basisOfRecord)}
                              </div>
                            )}
                            {feature.properties.eventDate && (
                              <div className="text-xs">
                                {feature.properties.eventDate}
                              </div>
                            )}
                            <div className="text-xs text-gray-500">
                              {lat.toFixed(4)}, {lon.toFixed(4)}
                            </div>
                            <a
                              href={`https://www.gbif.org/occurrence/${feature.properties.gbifID}`}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-xs text-blue-600 hover:text-blue-800 hover:underline mt-1 inline-block"
                            >
                              View on GBIF →
                            </a>
                          </div>
                        </Popup>
                      </CircleMarker>
                    );
                  })}
                </MapContainer>
              ) : null}
              {!loadingOccurrences && (
                <div className="absolute bottom-2 left-2 bg-white dark:bg-zinc-800 px-2 py-1 rounded text-xs text-zinc-600 dark:text-zinc-300 shadow">
                  {occurrences.length} occurrences
                  {hasCandidates &&
                    showCandidates &&
                    ` • ${candidates.length} predictions`}
                </div>
              )}
            </div>
          </div>
        </div>
      </td>
    </tr>
  );
}

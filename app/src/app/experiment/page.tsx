"use client";

import React, { useState, useEffect, useMemo } from "react";
import dynamic from "next/dynamic";

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
const Rectangle = dynamic(
  () => import("react-leaflet").then((mod) => mod.Rectangle),
  { ssr: false }
);

interface Point {
  lon: number;
  lat: number;
  score?: number;
}

interface Trial {
  test_block: number;
  n_blocks: number;
  auc: number;
  auc_vs_random: number;
  optimal_threshold: number;
  precision: number;
  recall: number;
  f1: number;
  n_train_positive: number;
  n_train_negative: number;
  n_test_positive: number;
  n_test_negative: number;
  train_positive: { lon: number; lat: number }[];
  train_negative: { lon: number; lat: number }[];
  test_positive: Point[];
  test_negative: Point[];
}

interface SpeciesData {
  species: string;
  species_key: number;
  region: string;
  model_type: string;
  n_blocks: number;
  n_valid_folds: number;
  n_occurrences_raw: number;
  n_occurrences_thinned: number;
  thinning_resolution_deg: number;
  negative_ratio: number;
  block_counts: Record<string, number>;
  auc_mean: number;
  auc_std: number;
  auc_vs_random_mean: number;
  f1_mean: number;
  f1_std: number;
  precision_mean: number;
  recall_mean: number;
  optimal_threshold_mean: number;
  trials: Trial[];
}

interface Summary {
  species: {
    species: string;
    n_occurrences_raw: number;
    n_occurrences_thinned: number;
    n_valid_folds: number;
    auc_mean: number;
    auc_std: number;
    auc_vs_random: number;
  }[];
}

type ModelType = "logistic" | "mlp";

const SPECIES_FILES = [
  "quercus_robur",
  "fraxinus_excelsior",
  "urtica_dioica",
  "alnus_glutinosa",
  "salix_caprea",
];

// Cambridge bounding box
const BBOX = { minLon: 0.03, minLat: 52.13, maxLon: 0.22, maxLat: 52.29 };

// Block colors for visualization
const BLOCK_COLORS = [
  "#ef4444", "#f97316", "#eab308",
  "#22c55e", "#14b8a6", "#3b82f6",
  "#8b5cf6", "#ec4899", "#6b7280",
];

export default function ExperimentPage() {
  const [speciesDataByModel, setSpeciesDataByModel] = useState<Record<ModelType, Record<string, SpeciesData>>>({
    logistic: {},
    mlp: {},
  });
  const [summaryByModel, setSummaryByModel] = useState<Record<ModelType, Summary | null>>({
    logistic: null,
    mlp: null,
  });
  const [selectedSpecies, setSelectedSpecies] = useState<string>("quercus_robur");
  const [selectedFoldIdx, setSelectedFoldIdx] = useState<number>(0);
  const [mounted, setMounted] = useState(false);
  const [loading, setLoading] = useState(true);
  const [modelType, setModelType] = useState<ModelType>("logistic");
  const [showExplanation, setShowExplanation] = useState(true);
  const [mapView, setMapView] = useState<"blocks" | "predictions">("blocks");

  useEffect(() => {
    setMounted(true);

    const loadModelData = async (mt: ModelType) => {
      const results = await Promise.all(
        SPECIES_FILES.map(async (slug) => {
          try {
            const res = await fetch(`/experiments/${mt}/${slug}.json`);
            if (res.ok) {
              const data = await res.json();
              return [slug, data] as [string, SpeciesData];
            }
          } catch (e) {
            console.error(`Failed to load ${mt}/${slug}:`, e);
          }
          return null;
        })
      );

      const data: Record<string, SpeciesData> = {};
      results.forEach((r) => {
        if (r) data[r[0]] = r[1];
      });
      return data;
    };

    const loadSummary = async (mt: ModelType) => {
      try {
        const res = await fetch(`/experiments/${mt}/summary.json`);
        if (res.ok) {
          return await res.json() as Summary;
        }
      } catch (e) {
        console.error(`Failed to load ${mt}/summary.json:`, e);
      }
      return null;
    };

    Promise.all([
      loadModelData("logistic"),
      loadModelData("mlp"),
      loadSummary("logistic"),
      loadSummary("mlp"),
    ]).then(([logisticData, mlpData, logisticSummary, mlpSummary]) => {
      setSpeciesDataByModel({
        logistic: logisticData,
        mlp: mlpData,
      });
      setSummaryByModel({
        logistic: logisticSummary,
        mlp: mlpSummary,
      });
      setLoading(false);
    });
  }, []);

  const speciesData = speciesDataByModel[modelType];
  const summary = summaryByModel[modelType];

  const currentData = speciesData[selectedSpecies];
  const currentTrial = currentData?.trials[selectedFoldIdx];

  // Calculate block boundaries for visualization
  const blockBounds = useMemo(() => {
    if (!currentData) return [];
    const n = Math.sqrt(currentData.n_blocks); // 3 for 3x3 grid
    const lonStep = (BBOX.maxLon - BBOX.minLon) / n;
    const latStep = (BBOX.maxLat - BBOX.minLat) / n;

    const bounds: { id: number; bounds: [[number, number], [number, number]]; count: number }[] = [];
    for (let row = 0; row < n; row++) {
      for (let col = 0; col < n; col++) {
        const id = row * n + col;
        bounds.push({
          id,
          bounds: [
            [BBOX.minLat + row * latStep, BBOX.minLon + col * lonStep],
            [BBOX.minLat + (row + 1) * latStep, BBOX.minLon + (col + 1) * lonStep],
          ],
          count: currentData.block_counts[id.toString()] || 0,
        });
      }
    }
    return bounds;
  }, [currentData]);

  // Reset fold when species changes
  useEffect(() => {
    setSelectedFoldIdx(0);
  }, [selectedSpecies]);

  if (loading) {
    return (
      <div className="min-h-screen bg-zinc-50 dark:bg-zinc-950 flex items-center justify-center">
        <div className="text-zinc-500">Loading experiment data...</div>
      </div>
    );
  }

  if (!currentData || !currentTrial) {
    return (
      <div className="min-h-screen bg-zinc-50 dark:bg-zinc-950 flex items-center justify-center">
        <div className="text-zinc-500">No experiment data found</div>
      </div>
    );
  }

  const testBlockId = currentTrial.test_block;
  const trainBlockIds = blockBounds.filter(b => b.id !== testBlockId).map(b => b.id);

  // Calculate metrics for current threshold
  const threshold = currentData.optimal_threshold_mean;
  const tp = currentTrial.test_positive.filter(p => (p.score ?? 0) >= threshold).length;
  const fn = currentTrial.test_positive.filter(p => (p.score ?? 0) < threshold).length;
  const fp = currentTrial.test_negative.filter(p => (p.score ?? 0) >= threshold).length;
  const tn = currentTrial.test_negative.filter(p => (p.score ?? 0) < threshold).length;

  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-zinc-950 p-4 md:p-8">
      <main className="max-w-7xl mx-auto">

        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-2xl font-bold text-zinc-900 dark:text-zinc-100">
              Model Validation
            </h1>
            <p className="text-zinc-500 text-sm mt-1">
              How well can we predict where a species occurs?
            </p>
          </div>
          <a
            href="/experiment/sample-size"
            className="px-4 py-2 bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 rounded-lg hover:bg-amber-200 text-sm font-medium"
          >
            How much data do we need? →
          </a>
        </div>

        {/* Model Type Toggle */}
        <div className="bg-white dark:bg-zinc-900 rounded-xl border border-zinc-200 dark:border-zinc-800 p-4 mb-6">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-medium text-zinc-900 dark:text-zinc-100">Model Type</h3>
              <p className="text-xs text-zinc-500 mt-1">Compare logistic regression vs neural network</p>
            </div>
            <div className="flex rounded-lg overflow-hidden border border-zinc-200 dark:border-zinc-700">
              <button
                onClick={() => setModelType("logistic")}
                className={`px-4 py-2 text-sm font-medium transition-colors ${
                  modelType === "logistic"
                    ? "bg-green-600 text-white"
                    : "bg-white dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400 hover:bg-zinc-50"
                }`}
              >
                Logistic Regression
              </button>
              <button
                onClick={() => setModelType("mlp")}
                className={`px-4 py-2 text-sm font-medium transition-colors ${
                  modelType === "mlp"
                    ? "bg-purple-600 text-white"
                    : "bg-white dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400 hover:bg-zinc-50"
                }`}
              >
                MLP + Uncertainty
              </button>
            </div>
          </div>

          {/* Model comparison summary */}
          {summaryByModel.logistic && summaryByModel.mlp && (
            <div className="mt-4 pt-4 border-t border-zinc-200 dark:border-zinc-700">
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div className={`p-3 rounded-lg ${modelType === "logistic" ? "bg-green-50 dark:bg-green-900/20 ring-2 ring-green-500" : "bg-zinc-50 dark:bg-zinc-800"}`}>
                  <div className="font-medium text-zinc-900 dark:text-zinc-100">Logistic Regression</div>
                  <div className="text-2xl font-bold text-green-600 mt-1">
                    {(summaryByModel.logistic.species.reduce((sum, s) => sum + s.auc_mean, 0) / summaryByModel.logistic.species.length * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-zinc-500">Average AUC</div>
                  <div className="text-xs text-zinc-400 mt-1">Fast, interpretable</div>
                </div>
                <div className={`p-3 rounded-lg ${modelType === "mlp" ? "bg-purple-50 dark:bg-purple-900/20 ring-2 ring-purple-500" : "bg-zinc-50 dark:bg-zinc-800"}`}>
                  <div className="font-medium text-zinc-900 dark:text-zinc-100">MLP + MC Dropout</div>
                  <div className="text-2xl font-bold text-purple-600 mt-1">
                    {(summaryByModel.mlp.species.reduce((sum, s) => sum + s.auc_mean, 0) / summaryByModel.mlp.species.length * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-zinc-500">Average AUC</div>
                  <div className="text-xs text-zinc-400 mt-1">Provides uncertainty estimates</div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Explanation Toggle */}
        <button
          onClick={() => setShowExplanation(!showExplanation)}
          className="mb-4 text-sm text-blue-600 hover:text-blue-700 flex items-center gap-1"
        >
          {showExplanation ? "▼" : "▶"} {showExplanation ? "Hide" : "Show"} methodology explanation
        </button>

        {/* Visual Explanation */}
        {showExplanation && (
          <div className="bg-white dark:bg-zinc-900 rounded-xl border border-zinc-200 dark:border-zinc-800 p-6 mb-6">
            <h2 className="font-semibold text-zinc-900 dark:text-zinc-100 mb-4">
              How We Test The Model (Spatial Block Cross-Validation)
            </h2>

            <div className="grid md:grid-cols-3 gap-6">
              {/* Step 1: Thinning */}
              <div className="text-center">
                <div className="w-16 h-16 mx-auto mb-3 bg-blue-100 dark:bg-blue-900/30 rounded-full flex items-center justify-center">
                  <span className="text-2xl">🔍</span>
                </div>
                <h3 className="font-medium text-zinc-900 dark:text-zinc-100 mb-2">
                  1. Thin Clustered Data
                </h3>
                <p className="text-sm text-zinc-600 dark:text-zinc-400">
                  Raw occurrence data is often clustered (many records near roads/cities).
                  We keep only <span className="font-medium text-blue-600">1 point per ~500m</span> to reduce bias.
                </p>
                <div className="mt-2 text-xs text-zinc-500">
                  {currentData.n_occurrences_raw} raw → {currentData.n_occurrences_thinned} thinned
                  <span className="text-red-500 ml-1">
                    ({Math.round((1 - currentData.n_occurrences_thinned / currentData.n_occurrences_raw) * 100)}% removed)
                  </span>
                </div>
              </div>

              {/* Step 2: Spatial Blocks */}
              <div className="text-center">
                <div className="w-16 h-16 mx-auto mb-3 bg-green-100 dark:bg-green-900/30 rounded-full flex items-center justify-center">
                  <span className="text-2xl">🗺️</span>
                </div>
                <h3 className="font-medium text-zinc-900 dark:text-zinc-100 mb-2">
                  2. Divide Into Spatial Blocks
                </h3>
                <p className="text-sm text-zinc-600 dark:text-zinc-400">
                  We split the region into a <span className="font-medium text-green-600">3×3 grid</span>.
                  Nearby points share similar environments, so we must test on <em>separate regions</em>.
                </p>
                <div className="mt-2 flex justify-center gap-1">
                  {[0,1,2,3,4,5,6,7,8].map(i => (
                    <div
                      key={i}
                      className="w-4 h-4 rounded-sm"
                      style={{ backgroundColor: BLOCK_COLORS[i] }}
                    />
                  ))}
                </div>
              </div>

              {/* Step 3: Train/Test Split */}
              <div className="text-center">
                <div className="w-16 h-16 mx-auto mb-3 bg-purple-100 dark:bg-purple-900/30 rounded-full flex items-center justify-center">
                  <span className="text-2xl">🔄</span>
                </div>
                <h3 className="font-medium text-zinc-900 dark:text-zinc-100 mb-2">
                  3. Leave-One-Block-Out
                </h3>
                <p className="text-sm text-zinc-600 dark:text-zinc-400">
                  For each fold: <span className="font-medium text-yellow-600">train on 8 blocks</span>,
                  <span className="font-medium text-red-600 ml-1">test on 1 block</span>.
                  This prevents "cheating" by memorizing nearby locations.
                </p>
                <div className="mt-2 text-xs text-zinc-500">
                  {currentData.n_valid_folds} folds tested
                </div>
              </div>
            </div>

            {/* Why This Matters */}
            <div className="mt-6 p-4 bg-amber-50 dark:bg-amber-900/20 rounded-lg">
              <p className="text-sm text-amber-800 dark:text-amber-200">
                <strong>Why this matters:</strong> Without spatial separation, nearby training and test points
                share almost identical environments. The model appears to perform well (~87% AUC) but is just
                memorizing locations. With proper spatial CV, we see realistic performance (~64% AUC).
              </p>
            </div>
          </div>
        )}

        {/* Species Selector */}
        <div className="bg-white dark:bg-zinc-900 rounded-xl border border-zinc-200 dark:border-zinc-800 p-4 mb-6">
          <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-3">
            Select Species
          </label>
          <div className="flex flex-wrap gap-2">
            {Object.entries(speciesData).map(([slug, data]) => (
              <button
                key={slug}
                onClick={() => setSelectedSpecies(slug)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                  selectedSpecies === slug
                    ? "bg-green-600 text-white shadow-md"
                    : "bg-zinc-100 dark:bg-zinc-800 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-200"
                }`}
              >
                <span className="italic">{data.species}</span>
                <span className="ml-2 text-xs opacity-75">({data.n_occurrences_thinned} pts)</span>
              </button>
            ))}
          </div>
        </div>

        {/* Main Content: Map + Metrics */}
        <div className="grid lg:grid-cols-3 gap-6 mb-6">

          {/* Map */}
          <div className="lg:col-span-2 bg-white dark:bg-zinc-900 rounded-xl border border-zinc-200 dark:border-zinc-800 overflow-hidden">
            {/* Map Controls */}
            <div className="p-3 border-b border-zinc-200 dark:border-zinc-700 flex items-center justify-between">
              <div className="flex gap-2">
                <button
                  onClick={() => setMapView("blocks")}
                  className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                    mapView === "blocks"
                      ? "bg-blue-600 text-white"
                      : "bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400"
                  }`}
                >
                  Show Blocks
                </button>
                <button
                  onClick={() => setMapView("predictions")}
                  className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                    mapView === "predictions"
                      ? "bg-blue-600 text-white"
                      : "bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400"
                  }`}
                >
                  Show Predictions
                </button>
              </div>

              {/* Fold selector */}
              <div className="flex items-center gap-2">
                <span className="text-sm text-zinc-500">Test block:</span>
                <div className="flex gap-1">
                  {currentData.trials.map((trial, idx) => (
                    <button
                      key={idx}
                      onClick={() => setSelectedFoldIdx(idx)}
                      className={`w-7 h-7 rounded text-xs font-bold transition-all ${
                        selectedFoldIdx === idx
                          ? "ring-2 ring-offset-2 ring-blue-500"
                          : ""
                      }`}
                      style={{
                        backgroundColor: BLOCK_COLORS[trial.test_block],
                        color: "white"
                      }}
                    >
                      {trial.test_block}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Map Legend */}
            <div className="p-3 border-b border-zinc-200 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-800/50">
              {mapView === "blocks" ? (
                <div className="flex flex-wrap items-center gap-4 text-sm">
                  <div className="flex items-center gap-2">
                    <div
                      className="w-5 h-5 rounded border-2 border-dashed border-zinc-400"
                      style={{ backgroundColor: BLOCK_COLORS[testBlockId] + "40" }}
                    />
                    <span className="text-zinc-600 dark:text-zinc-400">
                      <strong>Test block {testBlockId}</strong> ({currentTrial.n_test_positive} species points)
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-5 h-5 rounded bg-yellow-500" />
                    <span className="text-zinc-600 dark:text-zinc-400">
                      Training blocks ({currentTrial.n_train_positive} species points)
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-5 h-5 rounded bg-purple-500" />
                    <span className="text-zinc-600 dark:text-zinc-400">
                      Random background ({currentTrial.n_train_negative} points, 5:1 ratio)
                    </span>
                  </div>
                </div>
              ) : (
                <div className="flex flex-wrap items-center gap-4 text-sm">
                  <span className="text-zinc-500">Predictions on test block {testBlockId}:</span>
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-4 rounded-full bg-green-500" />
                    <span className="text-green-700 dark:text-green-400">True Positive ({tp})</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-4 rounded-full bg-orange-500" />
                    <span className="text-orange-700 dark:text-orange-400">False Negative ({fn})</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-4 rounded-full bg-red-500" />
                    <span className="text-red-700 dark:text-red-400">False Positive ({fp})</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-4 rounded-full bg-zinc-400" />
                    <span className="text-zinc-600 dark:text-zinc-400">True Negative ({tn})</span>
                  </div>
                </div>
              )}
            </div>

            {/* Map */}
            <div className="h-[500px]">
              {mounted && (
                <MapContainer
                  center={[52.205, 0.125]}
                  zoom={11}
                  style={{ height: "100%", width: "100%" }}
                >
                  <TileLayer
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                  />

                  {mapView === "blocks" ? (
                    <>
                      {/* Draw block grid */}
                      {blockBounds.map((block) => {
                        const isTestBlock = block.id === testBlockId;
                        return (
                          <Rectangle
                            key={block.id}
                            bounds={block.bounds}
                            pathOptions={{
                              color: BLOCK_COLORS[block.id],
                              fillColor: BLOCK_COLORS[block.id],
                              fillOpacity: isTestBlock ? 0.3 : 0.1,
                              weight: isTestBlock ? 3 : 1,
                              dashArray: isTestBlock ? "10, 5" : undefined,
                            }}
                          >
                            <Popup>
                              <div className="text-sm">
                                <div className="font-bold">Block {block.id}</div>
                                <div>{block.count} occurrence points</div>
                                <div className="text-xs text-zinc-500 mt-1">
                                  {isTestBlock ? "🧪 TEST BLOCK" : "📚 Training block"}
                                </div>
                              </div>
                            </Popup>
                          </Rectangle>
                        );
                      })}

                      {/* Training negatives (purple) */}
                      {currentTrial.train_negative.map((pt, idx) => (
                        <CircleMarker
                          key={`neg-${idx}`}
                          center={[pt.lat, pt.lon]}
                          radius={4}
                          pathOptions={{
                            color: "#7c3aed",
                            fillColor: "#a78bfa",
                            fillOpacity: 0.7,
                            weight: 1,
                          }}
                        />
                      ))}

                      {/* Training positives (yellow) */}
                      {currentTrial.train_positive.map((pt, idx) => (
                        <CircleMarker
                          key={`pos-${idx}`}
                          center={[pt.lat, pt.lon]}
                          radius={6}
                          pathOptions={{
                            color: "#ca8a04",
                            fillColor: "#facc15",
                            fillOpacity: 0.9,
                            weight: 2,
                          }}
                        >
                          <Popup>
                            <div className="text-sm">
                              <div className="font-medium text-yellow-700">Training: Species Present</div>
                              <div className="text-xs text-zinc-500">Model learns from this point</div>
                            </div>
                          </Popup>
                        </CircleMarker>
                      ))}

                      {/* Test points (in test block, shown differently) */}
                      {currentTrial.test_positive.map((pt, idx) => (
                        <CircleMarker
                          key={`test-pos-${idx}`}
                          center={[pt.lat, pt.lon]}
                          radius={6}
                          pathOptions={{
                            color: "#dc2626",
                            fillColor: "#fca5a5",
                            fillOpacity: 0.9,
                            weight: 2,
                          }}
                        >
                          <Popup>
                            <div className="text-sm">
                              <div className="font-medium text-red-700">Test: Species Present</div>
                              <div className="text-xs text-zinc-500">Model is evaluated on this point</div>
                              <div className="text-xs mt-1">Score: {(pt.score ?? 0).toFixed(3)}</div>
                            </div>
                          </Popup>
                        </CircleMarker>
                      ))}
                    </>
                  ) : (
                    <>
                      {/* Test block outline */}
                      <Rectangle
                        bounds={blockBounds.find(b => b.id === testBlockId)?.bounds || [[0,0],[0,0]]}
                        pathOptions={{
                          color: "#3b82f6",
                          fillColor: "transparent",
                          fillOpacity: 0,
                          weight: 3,
                          dashArray: "10, 5",
                        }}
                      />

                      {/* True Negatives */}
                      {currentTrial.test_negative
                        .filter(p => (p.score ?? 0) < threshold)
                        .map((pt, idx) => (
                          <CircleMarker
                            key={`tn-${idx}`}
                            center={[pt.lat, pt.lon]}
                            radius={5}
                            pathOptions={{
                              color: "#6b7280",
                              fillColor: "#9ca3af",
                              fillOpacity: 0.6,
                              weight: 1,
                            }}
                          >
                            <Popup>
                              <div className="text-sm">
                                <div className="font-medium text-zinc-600">✓ True Negative</div>
                                <div className="text-xs">Score: {(pt.score ?? 0).toFixed(3)} (below {threshold.toFixed(2)})</div>
                                <div className="text-xs text-zinc-500">Correctly predicted: no species here</div>
                              </div>
                            </Popup>
                          </CircleMarker>
                        ))}

                      {/* False Positives */}
                      {currentTrial.test_negative
                        .filter(p => (p.score ?? 0) >= threshold)
                        .map((pt, idx) => (
                          <CircleMarker
                            key={`fp-${idx}`}
                            center={[pt.lat, pt.lon]}
                            radius={7}
                            pathOptions={{
                              color: "#dc2626",
                              fillColor: "#f87171",
                              fillOpacity: 0.8,
                              weight: 2,
                            }}
                          >
                            <Popup>
                              <div className="text-sm">
                                <div className="font-medium text-red-600">✗ False Positive</div>
                                <div className="text-xs">Score: {(pt.score ?? 0).toFixed(3)} (above {threshold.toFixed(2)})</div>
                                <div className="text-xs text-zinc-500">Wrong! Predicted species, but none here</div>
                              </div>
                            </Popup>
                          </CircleMarker>
                        ))}

                      {/* False Negatives */}
                      {currentTrial.test_positive
                        .filter(p => (p.score ?? 0) < threshold)
                        .map((pt, idx) => (
                          <CircleMarker
                            key={`fn-${idx}`}
                            center={[pt.lat, pt.lon]}
                            radius={7}
                            pathOptions={{
                              color: "#ea580c",
                              fillColor: "#fb923c",
                              fillOpacity: 0.8,
                              weight: 2,
                            }}
                          >
                            <Popup>
                              <div className="text-sm">
                                <div className="font-medium text-orange-600">✗ False Negative</div>
                                <div className="text-xs">Score: {(pt.score ?? 0).toFixed(3)} (below {threshold.toFixed(2)})</div>
                                <div className="text-xs text-zinc-500">Missed! Species is here but we predicted none</div>
                              </div>
                            </Popup>
                          </CircleMarker>
                        ))}

                      {/* True Positives */}
                      {currentTrial.test_positive
                        .filter(p => (p.score ?? 0) >= threshold)
                        .map((pt, idx) => (
                          <CircleMarker
                            key={`tp-${idx}`}
                            center={[pt.lat, pt.lon]}
                            radius={7}
                            pathOptions={{
                              color: "#16a34a",
                              fillColor: "#4ade80",
                              fillOpacity: 0.9,
                              weight: 2,
                            }}
                          >
                            <Popup>
                              <div className="text-sm">
                                <div className="font-medium text-green-600">✓ True Positive</div>
                                <div className="text-xs">Score: {(pt.score ?? 0).toFixed(3)} (above {threshold.toFixed(2)})</div>
                                <div className="text-xs text-zinc-500">Correct! Predicted species and it's here</div>
                              </div>
                            </Popup>
                          </CircleMarker>
                        ))}
                    </>
                  )}
                </MapContainer>
              )}
            </div>
          </div>

          {/* Metrics Panel */}
          <div className="space-y-4">

            {/* Current Fold Info */}
            <div className="bg-white dark:bg-zinc-900 rounded-xl border border-zinc-200 dark:border-zinc-800 p-4">
              <h3 className="font-medium text-zinc-900 dark:text-zinc-100 mb-3">
                Current Fold: Block {testBlockId}
              </h3>

              <div className="space-y-3">
                <div className="flex justify-between text-sm">
                  <span className="text-zinc-500">Training points:</span>
                  <span className="font-medium">
                    {currentTrial.n_train_positive} species + {currentTrial.n_train_negative} background
                  </span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-zinc-500">Test points:</span>
                  <span className="font-medium">
                    {currentTrial.n_test_positive} species + {currentTrial.n_test_negative} background
                  </span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-zinc-500">Threshold:</span>
                  <span className="font-medium">{threshold.toFixed(2)}</span>
                </div>
              </div>
            </div>

            {/* This Fold's Results */}
            <div className="bg-white dark:bg-zinc-900 rounded-xl border border-zinc-200 dark:border-zinc-800 p-4">
              <h3 className="font-medium text-zinc-900 dark:text-zinc-100 mb-3">
                Fold {testBlockId} Results
              </h3>

              <div className="text-center mb-4">
                <div className={`text-4xl font-bold ${
                  currentTrial.auc >= 0.7 ? "text-green-600" :
                  currentTrial.auc >= 0.55 ? "text-yellow-600" : "text-red-500"
                }`}>
                  {(currentTrial.auc * 100).toFixed(0)}%
                </div>
                <div className="text-sm text-zinc-500">AUC for this fold</div>
                <div className={`text-xs mt-1 ${currentTrial.auc_vs_random > 0 ? "text-green-600" : "text-red-500"}`}>
                  {currentTrial.auc_vs_random > 0 ? "+" : ""}{(currentTrial.auc_vs_random * 100).toFixed(0)}% vs random
                </div>
              </div>

              {/* Mini confusion matrix */}
              <div className="grid grid-cols-2 gap-1 text-center text-sm">
                <div className="bg-green-100 dark:bg-green-900/30 rounded p-2">
                  <div className="text-lg font-bold text-green-700">{tp}</div>
                  <div className="text-xs text-green-600">True Pos</div>
                </div>
                <div className="bg-red-100 dark:bg-red-900/30 rounded p-2">
                  <div className="text-lg font-bold text-red-700">{fn}</div>
                  <div className="text-xs text-red-600">False Neg</div>
                </div>
                <div className="bg-red-100 dark:bg-red-900/30 rounded p-2">
                  <div className="text-lg font-bold text-red-700">{fp}</div>
                  <div className="text-xs text-red-600">False Pos</div>
                </div>
                <div className="bg-green-100 dark:bg-green-900/30 rounded p-2">
                  <div className="text-lg font-bold text-green-700">{tn}</div>
                  <div className="text-xs text-green-600">True Neg</div>
                </div>
              </div>
            </div>

            {/* Overall Results */}
            <div className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl border border-blue-200 dark:border-blue-800 p-4">
              <h3 className="font-medium text-blue-900 dark:text-blue-100 mb-3">
                Average Across All {currentData.n_valid_folds} Folds
              </h3>

              <div className="text-center mb-3">
                <div className="text-3xl font-bold text-blue-700 dark:text-blue-300">
                  {(currentData.auc_mean * 100).toFixed(1)}%
                </div>
                <div className="text-xs text-blue-600 dark:text-blue-400">
                  ± {(currentData.auc_std * 100).toFixed(1)}% AUC
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2 text-sm">
                <div className="text-center p-2 bg-white/50 dark:bg-zinc-800/50 rounded">
                  <div className="font-bold text-zinc-900 dark:text-zinc-100">
                    {(currentData.precision_mean * 100).toFixed(0)}%
                  </div>
                  <div className="text-xs text-zinc-500">Precision</div>
                </div>
                <div className="text-center p-2 bg-white/50 dark:bg-zinc-800/50 rounded">
                  <div className="font-bold text-zinc-900 dark:text-zinc-100">
                    {(currentData.recall_mean * 100).toFixed(0)}%
                  </div>
                  <div className="text-xs text-zinc-500">Recall</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* All Species Comparison */}
        {summary && (
          <div className="bg-white dark:bg-zinc-900 rounded-xl border border-zinc-200 dark:border-zinc-800 p-4">
            <h3 className="font-medium text-zinc-900 dark:text-zinc-100 mb-4">
              All Species Comparison
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-zinc-500 border-b border-zinc-200 dark:border-zinc-700">
                    <th className="pb-2 pr-4">Species</th>
                    <th className="pb-2 px-2 text-right">Raw</th>
                    <th className="pb-2 px-2 text-right">Thinned</th>
                    <th className="pb-2 px-2 text-right">AUC</th>
                    <th className="pb-2 px-2 text-right">vs Random</th>
                    <th className="pb-2 pl-4">Performance</th>
                  </tr>
                </thead>
                <tbody>
                  {summary.species.map((sp) => {
                    const slug = sp.species.toLowerCase().replace(" ", "_");
                    const isSelected = slug === selectedSpecies;
                    const performance = sp.auc_mean >= 0.7 ? "good" : sp.auc_mean >= 0.55 ? "moderate" : "poor";

                    return (
                      <tr
                        key={sp.species}
                        onClick={() => setSelectedSpecies(slug)}
                        className={`border-b border-zinc-100 dark:border-zinc-800 cursor-pointer hover:bg-zinc-50 dark:hover:bg-zinc-800 ${
                          isSelected ? "bg-green-50 dark:bg-green-900/20" : ""
                        }`}
                      >
                        <td className="py-2 pr-4 font-medium italic">{sp.species}</td>
                        <td className="py-2 px-2 text-right text-zinc-500">{sp.n_occurrences_raw}</td>
                        <td className="py-2 px-2 text-right">{sp.n_occurrences_thinned}</td>
                        <td className="py-2 px-2 text-right font-bold">
                          {(sp.auc_mean * 100).toFixed(1)}%
                        </td>
                        <td className={`py-2 px-2 text-right ${sp.auc_vs_random > 0 ? "text-green-600" : "text-red-500"}`}>
                          {sp.auc_vs_random > 0 ? "+" : ""}{(sp.auc_vs_random * 100).toFixed(1)}%
                        </td>
                        <td className="py-2 pl-4">
                          <div className="w-24 h-2 bg-zinc-200 dark:bg-zinc-700 rounded-full overflow-hidden">
                            <div
                              className={`h-full rounded-full ${
                                performance === "good" ? "bg-green-500" :
                                performance === "moderate" ? "bg-yellow-500" : "bg-red-500"
                              }`}
                              style={{ width: `${Math.min(100, sp.auc_mean * 100)}%` }}
                            />
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

      </main>
    </div>
  );
}

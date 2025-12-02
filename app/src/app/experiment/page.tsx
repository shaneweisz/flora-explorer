"use client";

import React, { useState, useEffect } from "react";
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

interface Point {
  lon: number;
  lat: number;
  sim_score?: number;
  clf_score?: number;
}

interface Experiment {
  n: number;
  n_test_positive: number;
  n_test_negative: number;
  similarity: { auc: number; mean_positive: number; mean_negative: number };
  classifier: { auc: number; mean_positive: number; mean_negative: number };
  train: { lon: number; lat: number }[];
  test_positive: Point[];
  test_negative: Point[];
}

interface SpeciesData {
  species: string;
  species_key: number;
  region: string;
  n_occurrences: number;
  experiments: Experiment[];
}

const SPECIES_FILES = ["quercus_robur", "fraxinus_excelsior"];

export default function ExperimentPage() {
  const [speciesData, setSpeciesData] = useState<Record<string, SpeciesData>>({});
  const [selectedSpecies, setSelectedSpecies] = useState<string>("quercus_robur");
  const [selectedN, setSelectedN] = useState<number>(10);
  const [method, setMethod] = useState<"similarity" | "classifier">("classifier");
  const [threshold, setThreshold] = useState<number>(0.5);
  const [mounted, setMounted] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setMounted(true);
    // Load experiment data
    Promise.all(
      SPECIES_FILES.map(async (slug) => {
        try {
          const res = await fetch(`/experiments/${slug}.json`);
          if (res.ok) {
            const data = await res.json();
            return [slug, data] as [string, SpeciesData];
          }
        } catch (e) {
          console.error(`Failed to load ${slug}:`, e);
        }
        return null;
      })
    ).then((results) => {
      const data: Record<string, SpeciesData> = {};
      results.forEach((r) => {
        if (r) data[r[0]] = r[1];
      });
      setSpeciesData(data);
      setLoading(false);
    });
  }, []);

  const currentData = speciesData[selectedSpecies];
  const currentExp = currentData?.experiments.find((e) => e.n === selectedN);
  const availableN = currentData?.experiments.map((e) => e.n) || [];

  // Ensure selectedN is valid for current species
  useEffect(() => {
    if (availableN.length > 0 && !availableN.includes(selectedN)) {
      setSelectedN(availableN[0]);
    }
  }, [availableN, selectedN]);

  if (loading) {
    return (
      <div className="min-h-screen bg-zinc-50 dark:bg-zinc-950 flex items-center justify-center">
        <div className="text-zinc-500">Loading experiment data...</div>
      </div>
    );
  }

  if (!currentData || !currentExp) {
    return (
      <div className="min-h-screen bg-zinc-50 dark:bg-zinc-950 flex items-center justify-center">
        <div className="text-zinc-500">No experiment data found</div>
      </div>
    );
  }

  const metrics = method === "similarity" ? currentExp.similarity : currentExp.classifier;

  // Compute confusion matrix based on threshold
  const getScore = (pt: Point) => method === "similarity" ? pt.sim_score : pt.clf_score;

  const truePositives = currentExp.test_positive.filter(pt => (getScore(pt) ?? 0) >= threshold);
  const falseNegatives = currentExp.test_positive.filter(pt => (getScore(pt) ?? 0) < threshold);
  const trueNegatives = currentExp.test_negative.filter(pt => (getScore(pt) ?? 0) < threshold);
  const falsePositives = currentExp.test_negative.filter(pt => (getScore(pt) ?? 0) >= threshold);

  const tp = truePositives.length;
  const fn = falseNegatives.length;
  const tn = trueNegatives.length;
  const fp = falsePositives.length;

  const accuracy = (tp + tn) / (tp + tn + fp + fn);
  const precision = tp / (tp + fp) || 0;
  const recall = tp / (tp + fn) || 0;

  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-zinc-950 p-4 md:p-8">
      <main className="max-w-6xl mx-auto">
        <div className="mb-6">
          <h1 className="text-2xl font-bold text-zinc-900 dark:text-zinc-100 mb-2">
            Similarity vs Classifier Experiment
          </h1>
          <p className="text-zinc-600 dark:text-zinc-400">
            Comparing methods for predicting species locations from habitat embeddings
          </p>
        </div>

        {/* Controls */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          {/* Species selector */}
          <div className="bg-white dark:bg-zinc-900 rounded-xl p-4 border border-zinc-200 dark:border-zinc-800">
            <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-2">
              Species
            </label>
            <select
              value={selectedSpecies}
              onChange={(e) => setSelectedSpecies(e.target.value)}
              className="w-full px-3 py-2 rounded-lg border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100"
            >
              {Object.entries(speciesData).map(([slug, data]) => (
                <option key={slug} value={slug}>
                  {data.species} ({data.n_occurrences} occurrences)
                </option>
              ))}
            </select>
          </div>

          {/* N selector */}
          <div className="bg-white dark:bg-zinc-900 rounded-xl p-4 border border-zinc-200 dark:border-zinc-800">
            <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-2">
              Training samples (n)
            </label>
            <div className="flex flex-wrap gap-2">
              {availableN.map((n) => (
                <button
                  key={n}
                  onClick={() => setSelectedN(n)}
                  className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
                    selectedN === n
                      ? "bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900"
                      : "bg-zinc-100 text-zinc-700 hover:bg-zinc-200 dark:bg-zinc-800 dark:text-zinc-300"
                  }`}
                >
                  {n}
                </button>
              ))}
            </div>
          </div>

          {/* Method selector */}
          <div className="bg-white dark:bg-zinc-900 rounded-xl p-4 border border-zinc-200 dark:border-zinc-800">
            <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-2">
              Method
            </label>
            <div className="flex gap-2">
              <button
                onClick={() => setMethod("similarity")}
                className={`flex-1 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                  method === "similarity"
                    ? "bg-blue-600 text-white"
                    : "bg-zinc-100 text-zinc-700 hover:bg-zinc-200 dark:bg-zinc-800 dark:text-zinc-300"
                }`}
              >
                Similarity
              </button>
              <button
                onClick={() => setMethod("classifier")}
                className={`flex-1 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                  method === "classifier"
                    ? "bg-green-600 text-white"
                    : "bg-zinc-100 text-zinc-700 hover:bg-zinc-200 dark:bg-zinc-800 dark:text-zinc-300"
                }`}
              >
                Classifier
              </button>
            </div>
          </div>
        </div>

        {/* Metrics */}
        <div className="grid grid-cols-3 gap-4 mb-6">
          <div className="bg-white dark:bg-zinc-900 rounded-xl p-4 border border-zinc-200 dark:border-zinc-800 text-center">
            <div className={`text-3xl font-bold ${metrics.auc >= 0.7 ? "text-green-600" : metrics.auc >= 0.5 ? "text-yellow-600" : "text-red-600"}`}>
              {(metrics.auc * 100).toFixed(1)}%
            </div>
            <div className="text-sm text-zinc-500">AUC</div>
          </div>
          <div className="bg-white dark:bg-zinc-900 rounded-xl p-4 border border-zinc-200 dark:border-zinc-800 text-center">
            <div className="text-3xl font-bold text-blue-600">
              {metrics.mean_positive.toFixed(3)}
            </div>
            <div className="text-sm text-zinc-500">Mean Positive Score</div>
          </div>
          <div className="bg-white dark:bg-zinc-900 rounded-xl p-4 border border-zinc-200 dark:border-zinc-800 text-center">
            <div className="text-3xl font-bold text-red-600">
              {metrics.mean_negative.toFixed(3)}
            </div>
            <div className="text-sm text-zinc-500">Mean Negative Score</div>
          </div>
        </div>

        {/* Confusion Matrix */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          {/* Threshold control */}
          <div className="bg-white dark:bg-zinc-900 rounded-xl p-4 border border-zinc-200 dark:border-zinc-800">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-medium text-zinc-700 dark:text-zinc-300">
                Decision Threshold
              </h3>
              <span className="text-lg font-bold text-zinc-900 dark:text-zinc-100">
                {threshold.toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              className="w-full accent-green-600"
            />
            <div className="flex justify-between text-xs text-zinc-500 mt-1">
              <span>0</span>
              <span>0.5</span>
              <span>1</span>
            </div>
            {/* Derived metrics */}
            <div className="grid grid-cols-3 gap-2 mt-4 pt-4 border-t border-zinc-200 dark:border-zinc-700">
              <div className="text-center">
                <div className="text-lg font-bold text-zinc-900 dark:text-zinc-100">
                  {(accuracy * 100).toFixed(1)}%
                </div>
                <div className="text-xs text-zinc-500">Accuracy</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-bold text-zinc-900 dark:text-zinc-100">
                  {(precision * 100).toFixed(1)}%
                </div>
                <div className="text-xs text-zinc-500">Precision</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-bold text-zinc-900 dark:text-zinc-100">
                  {(recall * 100).toFixed(1)}%
                </div>
                <div className="text-xs text-zinc-500">Recall</div>
              </div>
            </div>
          </div>

          {/* Confusion matrix */}
          <div className="bg-white dark:bg-zinc-900 rounded-xl p-4 border border-zinc-200 dark:border-zinc-800">
            <h3 className="text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-3">
              Confusion Matrix
            </h3>
            <div className="grid grid-cols-3 gap-1 text-center text-sm">
              {/* Header row */}
              <div></div>
              <div className="text-zinc-500 font-medium py-1">Pred +</div>
              <div className="text-zinc-500 font-medium py-1">Pred −</div>
              {/* True positive row */}
              <div className="text-zinc-500 font-medium py-2">Actual +</div>
              <div className="bg-green-100 dark:bg-green-900/30 rounded p-2">
                <div className="text-xl font-bold text-green-700 dark:text-green-400">{tp}</div>
                <div className="text-xs text-green-600 dark:text-green-500">TP</div>
              </div>
              <div className="bg-red-100 dark:bg-red-900/30 rounded p-2">
                <div className="text-xl font-bold text-red-700 dark:text-red-400">{fn}</div>
                <div className="text-xs text-red-600 dark:text-red-500">FN</div>
              </div>
              {/* True negative row */}
              <div className="text-zinc-500 font-medium py-2">Actual −</div>
              <div className="bg-red-100 dark:bg-red-900/30 rounded p-2">
                <div className="text-xl font-bold text-red-700 dark:text-red-400">{fp}</div>
                <div className="text-xs text-red-600 dark:text-red-500">FP</div>
              </div>
              <div className="bg-green-100 dark:bg-green-900/30 rounded p-2">
                <div className="text-xl font-bold text-green-700 dark:text-green-400">{tn}</div>
                <div className="text-xs text-green-600 dark:text-green-500">TN</div>
              </div>
            </div>
          </div>
        </div>

        {/* AUC comparison table */}
        <div className="bg-white dark:bg-zinc-900 rounded-xl p-4 border border-zinc-200 dark:border-zinc-800 mb-6">
          <h3 className="text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-3">
            AUC by Training Size
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-zinc-200 dark:border-zinc-700">
                  <th className="text-left py-2 px-3 text-zinc-500">n</th>
                  <th className="text-right py-2 px-3 text-blue-600">Similarity</th>
                  <th className="text-right py-2 px-3 text-green-600">Classifier</th>
                  <th className="text-right py-2 px-3 text-zinc-500">Δ</th>
                </tr>
              </thead>
              <tbody>
                {currentData.experiments.map((exp) => {
                  const delta = exp.classifier.auc - exp.similarity.auc;
                  return (
                    <tr
                      key={exp.n}
                      className={`border-b border-zinc-100 dark:border-zinc-800 cursor-pointer hover:bg-zinc-50 dark:hover:bg-zinc-800 ${
                        exp.n === selectedN ? "bg-zinc-100 dark:bg-zinc-800" : ""
                      }`}
                      onClick={() => setSelectedN(exp.n)}
                    >
                      <td className="py-2 px-3 font-medium">{exp.n}</td>
                      <td className="py-2 px-3 text-right">{(exp.similarity.auc * 100).toFixed(1)}%</td>
                      <td className="py-2 px-3 text-right">{(exp.classifier.auc * 100).toFixed(1)}%</td>
                      <td className={`py-2 px-3 text-right ${delta > 0 ? "text-green-600" : "text-red-600"}`}>
                        {delta > 0 ? "+" : ""}{(delta * 100).toFixed(1)}%
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Map */}
        <div className="bg-white dark:bg-zinc-900 rounded-xl border border-zinc-200 dark:border-zinc-800 overflow-hidden">
          <div className="p-4 border-b border-zinc-200 dark:border-zinc-700">
            <div className="flex flex-wrap items-center gap-4">
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded-full bg-yellow-500 border-2 border-yellow-700" />
                <span className="text-sm text-zinc-600 dark:text-zinc-400">Train ({currentExp.train.length})</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded-full bg-green-500 border-2 border-green-700" />
                <span className="text-sm text-zinc-600 dark:text-zinc-400">TP ({tp})</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded-full bg-orange-500 border-2 border-orange-700" />
                <span className="text-sm text-zinc-600 dark:text-zinc-400">FN ({fn})</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded-full bg-red-500 border-2 border-red-700" />
                <span className="text-sm text-zinc-600 dark:text-zinc-400">FP ({fp})</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded-full bg-zinc-400 border-2 border-zinc-600" />
                <span className="text-sm text-zinc-600 dark:text-zinc-400">TN ({tn})</span>
              </div>
            </div>
          </div>
          <div className="h-[500px]">
            {mounted && (
              <MapContainer
                center={[52.205, 0.1235]}
                zoom={11}
                style={{ height: "100%", width: "100%" }}
              >
                <TileLayer
                  attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />
                {/* True Negatives (correctly classified negatives) - grey */}
                {trueNegatives.map((pt, idx) => {
                  const score = getScore(pt);
                  return (
                    <CircleMarker
                      key={`tn-${idx}`}
                      center={[pt.lat, pt.lon]}
                      radius={5}
                      pathOptions={{
                        color: "#52525b",
                        fillColor: "#a1a1aa",
                        fillOpacity: 0.6,
                        weight: 1,
                      }}
                    >
                      <Popup>
                        <div className="text-sm">
                          <div className="font-medium text-zinc-600">True Negative (TN)</div>
                          <div>Score: {score?.toFixed(3) ?? "N/A"}</div>
                          <div className="text-xs text-zinc-500">Correctly rejected</div>
                        </div>
                      </Popup>
                    </CircleMarker>
                  );
                })}
                {/* False Positives (incorrectly predicted as positive) - red */}
                {falsePositives.map((pt, idx) => {
                  const score = getScore(pt);
                  return (
                    <CircleMarker
                      key={`fp-${idx}`}
                      center={[pt.lat, pt.lon]}
                      radius={6}
                      pathOptions={{
                        color: "#b91c1c",
                        fillColor: "#ef4444",
                        fillOpacity: 0.8,
                        weight: 2,
                      }}
                    >
                      <Popup>
                        <div className="text-sm">
                          <div className="font-medium text-red-600">False Positive (FP)</div>
                          <div>Score: {score?.toFixed(3) ?? "N/A"}</div>
                          <div className="text-xs text-zinc-500">Incorrectly predicted</div>
                        </div>
                      </Popup>
                    </CircleMarker>
                  );
                })}
                {/* False Negatives (missed real occurrences) - orange */}
                {falseNegatives.map((pt, idx) => {
                  const score = getScore(pt);
                  return (
                    <CircleMarker
                      key={`fn-${idx}`}
                      center={[pt.lat, pt.lon]}
                      radius={6}
                      pathOptions={{
                        color: "#c2410c",
                        fillColor: "#f97316",
                        fillOpacity: 0.8,
                        weight: 2,
                      }}
                    >
                      <Popup>
                        <div className="text-sm">
                          <div className="font-medium text-orange-600">False Negative (FN)</div>
                          <div>Score: {score?.toFixed(3) ?? "N/A"}</div>
                          <div className="text-xs text-zinc-500">Missed occurrence</div>
                        </div>
                      </Popup>
                    </CircleMarker>
                  );
                })}
                {/* True Positives (correctly identified occurrences) - green */}
                {truePositives.map((pt, idx) => {
                  const score = getScore(pt);
                  return (
                    <CircleMarker
                      key={`tp-${idx}`}
                      center={[pt.lat, pt.lon]}
                      radius={6}
                      pathOptions={{
                        color: "#15803d",
                        fillColor: "#22c55e",
                        fillOpacity: 0.8,
                        weight: 2,
                      }}
                    >
                      <Popup>
                        <div className="text-sm">
                          <div className="font-medium text-green-600">True Positive (TP)</div>
                          <div>Score: {score?.toFixed(3) ?? "N/A"}</div>
                          <div className="text-xs text-zinc-500">Correctly identified</div>
                        </div>
                      </Popup>
                    </CircleMarker>
                  );
                })}
                {/* Training points (on top) */}
                {currentExp.train.map((pt, idx) => (
                  <CircleMarker
                    key={`train-${idx}`}
                    center={[pt.lat, pt.lon]}
                    radius={7}
                    pathOptions={{
                      color: "#a16207",
                      fillColor: "#eab308",
                      fillOpacity: 0.9,
                      weight: 2,
                    }}
                  >
                    <Popup>
                      <div className="text-sm">
                        <div className="font-medium text-yellow-600">Training Point</div>
                      </div>
                    </Popup>
                  </CircleMarker>
                ))}
              </MapContainer>
            )}
          </div>
        </div>

        {/* Back link */}
        <div className="mt-6">
          <a
            href="/"
            className="text-sm text-green-600 hover:text-green-700 hover:underline"
          >
            ← Back to Species Explorer
          </a>
        </div>
      </main>
    </div>
  );
}

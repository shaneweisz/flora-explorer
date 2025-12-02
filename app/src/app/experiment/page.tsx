"use client";

import React, { useState, useEffect } from "react";
import dynamic from "next/dynamic";
import SpeciesInfoBar from "../../components/SpeciesInfoBar";

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
  score?: number;
}

interface Trial {
  seed: number;
  auc: number;
  mean_positive: number;
  mean_negative: number;
  n_test_positive: number;
  n_test_negative: number;
  train_positive: { lon: number; lat: number }[];
  train_negative: { lon: number; lat: number }[];
  test_positive: Point[];
  test_negative: Point[];
}

interface Experiment {
  n_positive: number;
  n_negative: number;
  n_trials: number;
  auc_mean: number;
  auc_std: number;
  trials: Trial[];
}

interface SpeciesData {
  species: string;
  species_key: number;
  region: string;
  n_occurrences: number;
  n_trials: number;
  experiments: Experiment[];
}

const SPECIES_FILES = [
  "quercus_robur",
  "fraxinus_excelsior",
  "alnus_glutinosa",
  "crataegus_monogyna",
  "urtica_dioica",
  "salix_caprea",
  "aesculus_hippocastanum",
];

export default function ExperimentPage() {
  const [speciesData, setSpeciesData] = useState<Record<string, SpeciesData>>({});
  const [selectedSpecies, setSelectedSpecies] = useState<string>("quercus_robur");
  const [selectedNPositive, setSelectedNPositive] = useState<number>(10);
  const [selectedTrialIdx, setSelectedTrialIdx] = useState<number>(0);
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
  const currentExp = currentData?.experiments.find((e) => e.n_positive === selectedNPositive);
  const availableNPositive = currentData?.experiments.map((e) => e.n_positive) || [];
  const currentTrial = currentExp?.trials[selectedTrialIdx];

  // Ensure selectedNPositive is valid for current species
  useEffect(() => {
    if (availableNPositive.length > 0 && !availableNPositive.includes(selectedNPositive)) {
      setSelectedNPositive(availableNPositive[0]);
    }
  }, [availableNPositive, selectedNPositive]);

  // Reset trial index when n changes
  useEffect(() => {
    setSelectedTrialIdx(0);
  }, [selectedNPositive]);

  if (loading) {
    return (
      <div className="min-h-screen bg-zinc-50 dark:bg-zinc-950 flex items-center justify-center">
        <div className="text-zinc-500">Loading experiment data...</div>
      </div>
    );
  }

  if (!currentData || !currentExp || !currentTrial) {
    return (
      <div className="min-h-screen bg-zinc-50 dark:bg-zinc-950 flex items-center justify-center">
        <div className="text-zinc-500">No experiment data found</div>
      </div>
    );
  }

  // Compute confusion matrix based on threshold
  const getScore = (pt: Point) => pt.score ?? 0;

  const truePositives = currentTrial.test_positive.filter(pt => getScore(pt) >= threshold);
  const falseNegatives = currentTrial.test_positive.filter(pt => getScore(pt) < threshold);
  const trueNegatives = currentTrial.test_negative.filter(pt => getScore(pt) < threshold);
  const falsePositives = currentTrial.test_negative.filter(pt => getScore(pt) >= threshold);

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
        {/* Species Info Bar */}
        <div className="mb-6">
          <SpeciesInfoBar
            speciesKey={currentData.species_key}
            speciesName={currentData.species}
            occurrenceCount={currentData.n_occurrences}
            region={currentData.region}
          />
        </div>

        <div className="mb-6">
          <p className="text-zinc-600 dark:text-zinc-400">
            Validating classifier performance on held-out occurrences vs random background
            <span className="ml-1 text-zinc-500">({currentData.n_trials} trials per setting)</span>
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
              Positive training samples
              <span className="font-normal text-zinc-500 ml-1">(+ matching negatives)</span>
            </label>
            <div className="flex flex-wrap gap-2">
              {availableNPositive.map((n) => (
                <button
                  key={n}
                  onClick={() => setSelectedNPositive(n)}
                  className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
                    selectedNPositive === n
                      ? "bg-green-600 text-white"
                      : "bg-zinc-100 text-zinc-700 hover:bg-zinc-200 dark:bg-zinc-800 dark:text-zinc-300"
                  }`}
                >
                  {n}
                </button>
              ))}
            </div>
          </div>

          {/* Trial selector */}
          <div className="bg-white dark:bg-zinc-900 rounded-xl p-4 border border-zinc-200 dark:border-zinc-800">
            <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-2">
              Trial
              <span className="font-normal text-zinc-500 ml-1">(seed: {currentTrial.seed})</span>
            </label>
            <div className="flex flex-wrap gap-2">
              {currentExp.trials.map((trial, idx) => (
                <button
                  key={idx}
                  onClick={() => setSelectedTrialIdx(idx)}
                  className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
                    selectedTrialIdx === idx
                      ? "bg-blue-600 text-white"
                      : "bg-zinc-100 text-zinc-700 hover:bg-zinc-200 dark:bg-zinc-800 dark:text-zinc-300"
                  }`}
                >
                  {idx + 1}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Metrics */}
        <div className="grid grid-cols-4 gap-4 mb-6">
          <div className="bg-white dark:bg-zinc-900 rounded-xl p-4 border border-zinc-200 dark:border-zinc-800 text-center">
            <div className={`text-2xl font-bold ${currentExp.auc_mean >= 0.7 ? "text-green-600" : currentExp.auc_mean >= 0.5 ? "text-yellow-600" : "text-red-600"}`}>
              {(currentExp.auc_mean * 100).toFixed(1)}%
              <span className="text-sm font-normal text-zinc-500 ml-1">
                ± {(currentExp.auc_std * 100).toFixed(1)}
              </span>
            </div>
            <div className="text-sm text-zinc-500">Mean AUC</div>
          </div>
          <div className="bg-white dark:bg-zinc-900 rounded-xl p-4 border border-zinc-200 dark:border-zinc-800 text-center">
            <div className={`text-2xl font-bold ${currentTrial.auc >= 0.7 ? "text-green-600" : currentTrial.auc >= 0.5 ? "text-yellow-600" : "text-red-600"}`}>
              {(currentTrial.auc * 100).toFixed(1)}%
            </div>
            <div className="text-sm text-zinc-500">This Trial AUC</div>
          </div>
          <div className="bg-white dark:bg-zinc-900 rounded-xl p-4 border border-zinc-200 dark:border-zinc-800 text-center">
            <div className="text-2xl font-bold text-green-600">
              {currentTrial.mean_positive.toFixed(3)}
            </div>
            <div className="text-sm text-zinc-500">Mean Positive Score</div>
          </div>
          <div className="bg-white dark:bg-zinc-900 rounded-xl p-4 border border-zinc-200 dark:border-zinc-800 text-center">
            <div className="text-2xl font-bold text-red-600">
              {currentTrial.mean_negative.toFixed(3)}
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

        {/* AUC by training size */}
        <div className="bg-white dark:bg-zinc-900 rounded-xl p-4 border border-zinc-200 dark:border-zinc-800 mb-6">
          <h3 className="text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-3">
            AUC by Training Size
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-zinc-200 dark:border-zinc-700">
                  <th className="text-left py-2 px-3 text-zinc-500">Train +</th>
                  <th className="text-left py-2 px-3 text-zinc-500">Train −</th>
                  <th className="text-right py-2 px-3 text-green-600">Mean AUC</th>
                  <th className="text-right py-2 px-3 text-zinc-500">Std</th>
                  <th className="text-right py-2 px-3 text-zinc-500">Trials</th>
                </tr>
              </thead>
              <tbody>
                {currentData.experiments.map((exp) => (
                  <tr
                    key={exp.n_positive}
                    className={`border-b border-zinc-100 dark:border-zinc-800 cursor-pointer hover:bg-zinc-50 dark:hover:bg-zinc-800 ${
                      exp.n_positive === selectedNPositive ? "bg-zinc-100 dark:bg-zinc-800" : ""
                    }`}
                    onClick={() => setSelectedNPositive(exp.n_positive)}
                  >
                    <td className="py-2 px-3 font-medium">{exp.n_positive}</td>
                    <td className="py-2 px-3 font-medium">{exp.n_negative}</td>
                    <td className={`py-2 px-3 text-right font-medium ${exp.auc_mean >= 0.7 ? "text-green-600" : exp.auc_mean >= 0.5 ? "text-yellow-600" : "text-red-600"}`}>
                      {(exp.auc_mean * 100).toFixed(1)}%
                    </td>
                    <td className="py-2 px-3 text-right text-zinc-500">
                      ±{(exp.auc_std * 100).toFixed(1)}%
                    </td>
                    <td className="py-2 px-3 text-right text-zinc-500">{exp.n_trials}</td>
                  </tr>
                ))}
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
                <span className="text-sm text-zinc-600 dark:text-zinc-400">Train + ({currentTrial.train_positive.length})</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded-full bg-purple-500 border-2 border-purple-700" />
                <span className="text-sm text-zinc-600 dark:text-zinc-400">Train − ({currentTrial.train_negative.length})</span>
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
                          <div>Score: {score.toFixed(3)}</div>
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
                          <div>Score: {score.toFixed(3)}</div>
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
                          <div>Score: {score.toFixed(3)}</div>
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
                          <div>Score: {score.toFixed(3)}</div>
                          <div className="text-xs text-zinc-500">Correctly identified</div>
                        </div>
                      </Popup>
                    </CircleMarker>
                  );
                })}
                {/* Training negative points - purple */}
                {currentTrial.train_negative.map((pt, idx) => (
                  <CircleMarker
                    key={`train-neg-${idx}`}
                    center={[pt.lat, pt.lon]}
                    radius={7}
                    pathOptions={{
                      color: "#7e22ce",
                      fillColor: "#a855f7",
                      fillOpacity: 0.9,
                      weight: 2,
                    }}
                  >
                    <Popup>
                      <div className="text-sm">
                        <div className="font-medium text-purple-600">Training Negative</div>
                        <div className="text-xs text-zinc-500">Background sample</div>
                      </div>
                    </Popup>
                  </CircleMarker>
                ))}
                {/* Training positive points (on top) - yellow */}
                {currentTrial.train_positive.map((pt, idx) => (
                  <CircleMarker
                    key={`train-pos-${idx}`}
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
                        <div className="font-medium text-yellow-600">Training Positive</div>
                        <div className="text-xs text-zinc-500">Known occurrence</div>
                      </div>
                    </Popup>
                  </CircleMarker>
                ))}
              </MapContainer>
            )}
          </div>
        </div>

      </main>
    </div>
  );
}

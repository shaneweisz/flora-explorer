"use client";

import React, { useState, useEffect } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";

interface SampleSizeResult {
  n_train: number;
  n_folds: number;
  auc_mean: number;
  auc_std: number;
  auc_vs_random: number;
}

interface SpeciesResult {
  species: string;
  n_total: number;
  sample_sizes: SampleSizeResult[];
}

interface Summary {
  n_train_values: number[];
  n_subsamples: number;
  negative_ratio: number;
  species: SpeciesResult[];
}

const COLORS = [
  "#22c55e", // green
  "#3b82f6", // blue
  "#f59e0b", // amber
  "#ef4444", // red
  "#8b5cf6", // purple
];

export default function SampleSizePage() {
  const [data, setData] = useState<Summary | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/experiments/sample_size/summary.json")
      .then((res) => res.json())
      .then((data) => {
        setData(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Failed to load sample size data:", err);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-zinc-50 dark:bg-zinc-950 flex items-center justify-center">
        <div className="text-zinc-500">Loading...</div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="min-h-screen bg-zinc-50 dark:bg-zinc-950 flex items-center justify-center">
        <div className="text-zinc-500">No data found</div>
      </div>
    );
  }

  // Transform data for recharts
  const chartData = data.n_train_values.map((n) => {
    const point: Record<string, number | string> = { n_train: n };
    data.species.forEach((sp) => {
      const result = sp.sample_sizes.find((s) => s.n_train === n);
      if (result) {
        const key = sp.species.replace(" ", "_");
        point[key] = result.auc_mean;
        point[`${key}_std`] = result.auc_std;
      }
    });
    return point;
  });

  // Calculate average across species for each n
  const avgData = data.n_train_values.map((n) => {
    const values = data.species
      .map((sp) => sp.sample_sizes.find((s) => s.n_train === n)?.auc_mean)
      .filter((v): v is number => v !== undefined);
    return {
      n_train: n,
      avg_auc: values.length > 0 ? values.reduce((a, b) => a + b, 0) / values.length : null,
    };
  });

  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-zinc-950 p-4 md:p-8">
      <main className="max-w-5xl mx-auto">
        <div className="mb-6">
          <h1 className="text-2xl font-bold text-zinc-900 dark:text-zinc-100 mb-2">
            Sample Size Experiment
          </h1>
          <p className="text-zinc-600 dark:text-zinc-400">
            How does predictive performance vary with the number of training samples?
            <span className="block text-sm text-zinc-500 mt-1">
              Spatial block CV with {data.n_subsamples} random subsamples per fold, {data.negative_ratio}:1 negative ratio
            </span>
          </p>
        </div>

        {/* Key Finding Alert */}
        <div className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-xl p-4 mb-6">
          <h3 className="font-medium text-amber-800 dark:text-amber-200 mb-2">
            Key Finding
          </h3>
          <p className="text-amber-700 dark:text-amber-300 text-sm">
            With fewer than 10 training samples, model performance is barely above random chance (AUC ≈ 0.50).
            Meaningful discrimination (AUC {">"} 0.60) typically requires 20+ samples. This suggests the approach
            may have limited utility for truly data-deficient species with {"<"}10 occurrence records.
          </p>
        </div>

        {/* Learning Curve Chart */}
        <div className="bg-white dark:bg-zinc-900 rounded-xl p-4 border border-zinc-200 dark:border-zinc-800 mb-6">
          <h3 className="text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-4">
            Learning Curve: AUC vs Training Sample Size
          </h3>
          <div className="h-[400px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
                <XAxis
                  dataKey="n_train"
                  label={{ value: "Training Samples (n)", position: "bottom", offset: 0 }}
                  stroke="#6b7280"
                />
                <YAxis
                  domain={[0.4, 0.8]}
                  label={{ value: "AUC", angle: -90, position: "insideLeft" }}
                  stroke="#6b7280"
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#18181b",
                    border: "1px solid #3f3f46",
                    borderRadius: "8px",
                  }}
                  labelStyle={{ color: "#a1a1aa" }}
                />
                <Legend />
                {/* Random baseline */}
                <ReferenceLine
                  y={0.5}
                  stroke="#ef4444"
                  strokeDasharray="5 5"
                  label={{ value: "Random (0.50)", position: "right", fill: "#ef4444", fontSize: 12 }}
                />
                {/* Species lines */}
                {data.species.map((sp, idx) => (
                  <Line
                    key={sp.species}
                    type="monotone"
                    dataKey={sp.species.replace(" ", "_")}
                    name={sp.species}
                    stroke={COLORS[idx % COLORS.length]}
                    strokeWidth={2}
                    dot={{ r: 4 }}
                    connectNulls
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Data Table */}
        <div className="bg-white dark:bg-zinc-900 rounded-xl p-4 border border-zinc-200 dark:border-zinc-800 mb-6">
          <h3 className="text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-3">
            Detailed Results
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-zinc-200 dark:border-zinc-700">
                  <th className="text-left py-2 px-3 text-zinc-500">Species</th>
                  <th className="text-right py-2 px-3 text-zinc-500">Total</th>
                  {data.n_train_values.map((n) => (
                    <th key={n} className="text-right py-2 px-3 text-zinc-500">
                      n={n}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.species.map((sp, idx) => (
                  <tr key={sp.species} className="border-b border-zinc-100 dark:border-zinc-800">
                    <td className="py-2 px-3 font-medium" style={{ color: COLORS[idx % COLORS.length] }}>
                      {sp.species}
                    </td>
                    <td className="py-2 px-3 text-right text-zinc-500">{sp.n_total}</td>
                    {data.n_train_values.map((n) => {
                      const result = sp.sample_sizes.find((s) => s.n_train === n);
                      if (!result) {
                        return (
                          <td key={n} className="py-2 px-3 text-right text-zinc-400">
                            -
                          </td>
                        );
                      }
                      const isGood = result.auc_mean >= 0.6;
                      const isBad = result.auc_mean < 0.52;
                      return (
                        <td
                          key={n}
                          className={`py-2 px-3 text-right font-medium ${
                            isGood
                              ? "text-green-600"
                              : isBad
                              ? "text-red-500"
                              : "text-yellow-600"
                          }`}
                        >
                          {(result.auc_mean * 100).toFixed(1)}%
                          <span className="text-xs text-zinc-400 ml-1">
                            ±{(result.auc_std * 100).toFixed(0)}
                          </span>
                        </td>
                      );
                    })}
                  </tr>
                ))}
                {/* Average row */}
                <tr className="bg-zinc-50 dark:bg-zinc-800/50 font-medium">
                  <td className="py-2 px-3">Average</td>
                  <td className="py-2 px-3 text-right text-zinc-500">-</td>
                  {avgData.map((d) => (
                    <td
                      key={d.n_train}
                      className={`py-2 px-3 text-right ${
                        d.avg_auc && d.avg_auc >= 0.6
                          ? "text-green-600"
                          : d.avg_auc && d.avg_auc < 0.52
                          ? "text-red-500"
                          : "text-yellow-600"
                      }`}
                    >
                      {d.avg_auc ? `${(d.avg_auc * 100).toFixed(1)}%` : "-"}
                    </td>
                  ))}
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Interpretation */}
        <div className="bg-white dark:bg-zinc-900 rounded-xl p-4 border border-zinc-200 dark:border-zinc-800">
          <h3 className="text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-3">
            Interpretation
          </h3>
          <div className="space-y-3 text-sm text-zinc-600 dark:text-zinc-400">
            <div className="flex items-start gap-2">
              <span className="text-red-500 font-bold">n=2-5:</span>
              <span>
                Performance is indistinguishable from random guessing (AUC ≈ 0.48-0.54).
                The model cannot learn meaningful habitat preferences from so few examples.
              </span>
            </div>
            <div className="flex items-start gap-2">
              <span className="text-yellow-600 font-bold">n=10:</span>
              <span>
                Slight improvement visible for some species (AUC ≈ 0.53-0.58), but still
                limited predictive power. High variance between folds.
              </span>
            </div>
            <div className="flex items-start gap-2">
              <span className="text-green-600 font-bold">n=20+:</span>
              <span>
                More consistent improvement (AUC ≈ 0.55-0.65). This appears to be the
                minimum sample size for useful predictions.
              </span>
            </div>
            <div className="flex items-start gap-2">
              <span className="text-zinc-500 font-bold">n=50:</span>
              <span>
                Performance plateaus or even decreases slightly for some species.
                Diminishing returns beyond ~20-50 samples.
              </span>
            </div>
          </div>
        </div>

        {/* Back link */}
        <div className="mt-6">
          <a
            href="/experiment"
            className="text-green-600 hover:text-green-700 hover:underline text-sm"
          >
            ← Back to Spatial Block CV Results
          </a>
        </div>
      </main>
    </div>
  );
}

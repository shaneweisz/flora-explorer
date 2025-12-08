"use client";

import { useState, useEffect } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  LabelList,
} from "recharts";

interface CategoryStats {
  code: string;
  name: string;
  count: number;
  color: string;
}

interface StatsResponse {
  totalAssessed: number;
  byCategory: CategoryStats[];
  sampleSize: number;
  lastUpdated: string;
  cached: boolean;
  error?: string;
}

interface YearRange {
  range: string;
  count: number;
  minYear: number;
}

interface AssessmentsResponse {
  yearsSinceAssessment: YearRange[];
  sampleSize: number;
  lastUpdated: string;
  cached: boolean;
  error?: string;
}

interface Species {
  sis_taxon_id: number;
  scientific_name: string;
  category: string;
  year_published: string;
  url: string;
}

interface SpeciesResponse {
  species: Species[];
  total: number;
  error?: string;
}

// IUCN category colors
const CATEGORY_COLORS: Record<string, string> = {
  EX: "#000000",
  EW: "#542344",
  CR: "#d81e05",
  EN: "#fc7f3f",
  VU: "#f9e814",
  NT: "#cce226",
  LC: "#60c659",
  DD: "#d1d1c6",
};

export default function RedListView() {
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [assessments, setAssessments] = useState<AssessmentsResponse | null>(null);
  const [species, setSpecies] = useState<Species[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");

  // Load stats and assessments
  useEffect(() => {
    async function fetchData() {
      setLoading(true);
      setError(null);

      try {
        const [statsRes, assessmentsRes, speciesRes] = await Promise.all([
          fetch("/api/redlist/stats"),
          fetch("/api/redlist/assessments"),
          fetch("/api/redlist/species"),
        ]);

        const statsData = await statsRes.json();
        const assessmentsData = await assessmentsRes.json();
        const speciesData: SpeciesResponse = await speciesRes.json();

        if (statsData.error) throw new Error(statsData.error);
        if (assessmentsData.error) throw new Error(assessmentsData.error);
        if (speciesData.error) throw new Error(speciesData.error);

        setStats(statsData);
        setAssessments(assessmentsData);
        setSpecies(speciesData.species);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load data");
      } finally {
        setLoading(false);
      }
    }

    fetchData();
  }, []);

  // Filter species based on category and search
  const filteredSpecies = species.filter((s) => {
    const matchesCategory = !selectedCategory || s.category === selectedCategory;
    const matchesSearch = !searchQuery ||
      s.scientific_name.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesCategory && matchesSearch;
  });

  // Handle chart bar click
  const handleBarClick = (data: { code: string }) => {
    if (selectedCategory === data.code) {
      setSelectedCategory(null); // Toggle off
    } else {
      setSelectedCategory(data.code);
    }
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-20">
        <div className="animate-spin h-10 w-10 border-4 border-red-600 border-t-transparent rounded-full" />
        <p className="mt-4 text-zinc-500 dark:text-zinc-400">
          Loading Red List statistics...
        </p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 px-6 py-4 rounded-lg mx-4">
        <p className="font-medium">Failed to load Red List data</p>
        <p className="text-sm mt-1">{error}</p>
        <button
          onClick={() => window.location.reload()}
          className="mt-3 text-sm underline hover:no-underline"
        >
          Try again
        </button>
      </div>
    );
  }

  if (!stats || !assessments) {
    return null;
  }

  // Calculate threatened species count (CR + EN + VU)
  const threatenedCount = stats.byCategory
    .filter((c) => ["CR", "EN", "VU"].includes(c.code))
    .reduce((sum, c) => sum + c.count, 0);

  // Calculate percentages for category chart
  const categoryDataWithPercent = stats.byCategory.map((cat) => ({
    ...cat,
    percent: ((cat.count / stats.sampleSize) * 100).toFixed(1),
    label: `${cat.count} (${((cat.count / stats.sampleSize) * 100).toFixed(0)}%)`,
  }));

  // Calculate stale assessments (>10 years)
  const staleCount = assessments.yearsSinceAssessment
    .filter((y) => y.minYear > 10)
    .reduce((sum, y) => sum + y.count, 0);
  const stalePercent = ((staleCount / assessments.sampleSize) * 100).toFixed(0);

  const currentYear = new Date().getFullYear();

  return (
    <div className="space-y-4">
      {/* Stats Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-20 gap-4">
        {/* Left column - Summary stats */}
        <div className="lg:col-span-3 space-y-3">
          <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl p-4">
            <p className="text-xs text-zinc-500 dark:text-zinc-400 uppercase tracking-wide">
              Total Assessed
            </p>
            <p className="text-2xl font-bold text-zinc-800 dark:text-zinc-100 mt-1">
              {stats.totalAssessed.toLocaleString()}
            </p>
            <p className="text-xs text-zinc-400 dark:text-zinc-500 mt-1">
              plant species globally
            </p>
          </div>

          <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl p-4">
            <p className="text-xs text-zinc-500 dark:text-zinc-400 uppercase tracking-wide">
              Threatened
            </p>
            <p className="text-2xl font-bold text-red-600 dark:text-red-500 mt-1">
              {threatenedCount}
            </p>
            <p className="text-xs text-zinc-400 dark:text-zinc-500 mt-1">
              CR + EN + VU ({((threatenedCount / stats.sampleSize) * 100).toFixed(0)}%)
            </p>
          </div>

          <div className="bg-white dark:bg-zinc-900 border border-amber-200 dark:border-amber-800/50 rounded-xl p-4">
            <p className="text-xs text-zinc-500 dark:text-zinc-400 uppercase tracking-wide">
              Stale Assessments
            </p>
            <p className="text-2xl font-bold text-amber-600 dark:text-amber-500 mt-1">
              {stalePercent}%
            </p>
            <p className="text-xs text-zinc-400 dark:text-zinc-500 mt-1">
              assessed 10+ years ago
            </p>
          </div>
        </div>

        {/* Center column - Category distribution (clickable) */}
        <div className="lg:col-span-10 bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl p-4">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-medium text-zinc-700 dark:text-zinc-300">
              Distribution by Category
            </h3>
            {selectedCategory && (
              <button
                onClick={() => setSelectedCategory(null)}
                className="text-xs text-red-600 hover:text-red-700 dark:text-red-400"
              >
                Clear filter
              </button>
            )}
          </div>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={categoryDataWithPercent}
                layout="vertical"
                margin={{ top: 5, right: 80, left: 100, bottom: 5 }}
              >
                <XAxis type="number" hide />
                <YAxis
                  type="category"
                  dataKey="name"
                  tick={{ fontSize: 11, fill: "#71717a" }}
                  tickLine={false}
                  axisLine={false}
                  width={95}
                />
                <Tooltip
                  formatter={(value: number) => [`${value} species`, "Count"]}
                  contentStyle={{
                    backgroundColor: "#18181b",
                    border: "1px solid #3f3f46",
                    borderRadius: "8px",
                    color: "#fff",
                  }}
                />
                <Bar
                  dataKey="count"
                  radius={[0, 4, 4, 0]}
                  cursor="pointer"
                  onClick={(data) => handleBarClick(data)}
                >
                  {categoryDataWithPercent.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={entry.color}
                      opacity={selectedCategory && selectedCategory !== entry.code ? 0.3 : 1}
                    />
                  ))}
                  <LabelList
                    dataKey="label"
                    position="right"
                    style={{ fontSize: 10, fill: "#71717a" }}
                  />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="text-xs text-zinc-400 text-center mt-2">
            Click a bar to filter the species list below
          </p>
        </div>

        {/* Right column - Years chart */}
        <div className="lg:col-span-7">
          <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl p-4 h-full">
            <h3 className="text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-4">
              Years Since Assessment
            </h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={assessments.yearsSinceAssessment}
                  margin={{ top: 5, right: 10, left: 0, bottom: 25 }}
                >
                  <XAxis
                    dataKey="range"
                    tick={{ fontSize: 9, fill: "#71717a" }}
                    tickLine={false}
                    axisLine={false}
                    angle={-15}
                    textAnchor="end"
                    height={45}
                  />
                  <YAxis
                    tick={{ fontSize: 9, fill: "#71717a" }}
                    tickLine={false}
                    axisLine={false}
                    width={30}
                  />
                  <Tooltip
                    formatter={(value: number) => [value, "Species"]}
                    contentStyle={{
                      backgroundColor: "#18181b",
                      border: "1px solid #3f3f46",
                      borderRadius: "8px",
                      color: "#fff",
                    }}
                  />
                  <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]}>
                    <LabelList
                      dataKey="count"
                      position="top"
                      style={{ fontSize: 9, fill: "#71717a" }}
                    />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>

      {/* Search and Species Table */}
      <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl overflow-hidden">
        {/* Search bar */}
        <div className="p-4 border-b border-zinc-200 dark:border-zinc-800">
          <div className="flex items-center gap-4">
            <div className="relative flex-1 max-w-md">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search species..."
                className="w-full px-4 py-2 pl-10 rounded-lg border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100 placeholder-zinc-400 focus:outline-none focus:ring-2 focus:ring-red-500 text-sm"
              />
              <svg
                className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-400"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            </div>
            {selectedCategory && (
              <span className="px-3 py-1 text-sm rounded-full" style={{ backgroundColor: CATEGORY_COLORS[selectedCategory] + "20", color: CATEGORY_COLORS[selectedCategory] }}>
                {selectedCategory}
              </span>
            )}
            <span className="text-sm text-zinc-500">
              {filteredSpecies.length} species
            </span>
          </div>
        </div>

        {/* Species table */}
        <div className="overflow-x-auto max-h-96">
          <table className="w-full text-sm">
            <thead className="bg-zinc-50 dark:bg-zinc-800 sticky top-0">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-zinc-500 uppercase tracking-wider">
                  Species
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-zinc-500 uppercase tracking-wider">
                  Category
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-zinc-500 uppercase tracking-wider">
                  Year Assessed
                </th>
                <th className="px-4 py-3 text-center text-xs font-medium text-zinc-500 uppercase tracking-wider">
                  Link
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-200 dark:divide-zinc-800">
              {filteredSpecies.map((s) => {
                const yearsSince = currentYear - parseInt(s.year_published);
                return (
                  <tr key={s.sis_taxon_id} className="hover:bg-zinc-50 dark:hover:bg-zinc-800/50">
                    <td className="px-4 py-3">
                      <span className="italic text-zinc-900 dark:text-zinc-100">
                        {s.scientific_name}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <span
                        className="px-2 py-1 text-xs font-medium rounded"
                        style={{
                          backgroundColor: CATEGORY_COLORS[s.category] + "20",
                          color: s.category === "EX" || s.category === "EW" ? "#fff" : CATEGORY_COLORS[s.category],
                          ...(s.category === "EX" || s.category === "EW" ? { backgroundColor: CATEGORY_COLORS[s.category] } : {})
                        }}
                      >
                        {s.category}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-zinc-600 dark:text-zinc-400">
                      {s.year_published}
                      {yearsSince > 10 && (
                        <span className="ml-2 text-xs text-amber-600">({yearsSince}y ago)</span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-center">
                      <a
                        href={s.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-red-600 hover:text-red-700"
                      >
                        <svg className="w-4 h-4 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                        </svg>
                      </a>
                    </td>
                  </tr>
                );
              })}
              {filteredSpecies.length === 0 && (
                <tr>
                  <td colSpan={4} className="px-4 py-8 text-center text-zinc-500">
                    No species found
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Sample note */}
      <p className="text-xs text-zinc-400 dark:text-zinc-500 text-center">
        Showing sample of {stats.sampleSize} species from IUCN Red List
      </p>
    </div>
  );
}

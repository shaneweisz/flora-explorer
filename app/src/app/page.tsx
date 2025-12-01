"use client";

import React, { useState, useEffect, useCallback } from "react";
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
const Rectangle = dynamic(
  () => import("react-leaflet").then((mod) => mod.Rectangle),
  { ssr: false }
);

// ============================================================================
// Types
// ============================================================================

interface SpeciesRecord {
  species_key: number;
  occurrence_count: number;
  canonicalName?: string;
  vernacularName?: string;
}

interface SpeciesDetails {
  key: number;
  scientificName: string;
  canonicalName: string;
  vernacularName?: string;
  kingdom: string;
  family: string;
  genus: string;
  gbifUrl: string;
  imageUrl?: string;
  occurrenceCount?: number;
}

interface OccurrenceFeature {
  type: "Feature";
  properties: {
    gbifID: number;
    species: string;
    eventDate?: string;
  };
  geometry: {
    type: "Point";
    coordinates: [number, number];
  };
}

interface Stats {
  total: number;
  filtered: number;
  totalOccurrences: number;
  median: number;
  distribution: {
    lte1: number;
    lte10: number;
    lte100: number;
    lte1000: number;
    lte10000: number;
  };
}

interface ApiResponse {
  data: SpeciesRecord[];
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
  };
  stats: Stats;
}

type FilterPreset = "all" | "dataDeficient" | "veryRare" | "singletons";
type RegionMode = "global" | "cambridge";

// ============================================================================
// Constants
// ============================================================================

const FILTER_PRESETS: Record<FilterPreset, { minCount: number; maxCount: number; label: string }> = {
  all: { minCount: 0, maxCount: 999999999, label: "All Species" },
  dataDeficient: { minCount: 0, maxCount: 100, label: "≤ 100" },
  veryRare: { minCount: 0, maxCount: 10, label: "≤ 10" },
  singletons: { minCount: 1, maxCount: 1, label: "= 1" },
};

const REGION_CONFIG: Record<RegionMode, {
  label: string;
  description: string;
  apiEndpoint: string;
  occurrencesEndpoint: string;
  center: [number, number];
  zoom: number;
  bounds?: [[number, number], [number, number]];
}> = {
  global: {
    label: "Global",
    description: "Worldwide GBIF occurrence data",
    apiEndpoint: "/api/species",
    occurrencesEndpoint: "/api/occurrences",
    center: [20, 0],
    zoom: 2,
  },
  cambridge: {
    label: "Cambridge",
    description: "Cambridge test region",
    apiEndpoint: "/api/cambridge/species",
    occurrencesEndpoint: "/api/cambridge/occurrences",
    center: [52.205, 0.1235],
    zoom: 11,
    bounds: [[52.092, -0.003], [52.318, 0.250]],
  },
};

// Shared search endpoint for all regions
const SEARCH_ENDPOINT = "/api/search";

// ============================================================================
// Helper Functions
// ============================================================================

const formatNumber = (num: number) => num.toLocaleString();
const getPercentage = (count: number, total: number) => ((count / total) * 100).toFixed(1);

// ============================================================================
// Components
// ============================================================================

function StatsCards({ stats }: { stats: Stats }) {
  return (
    <>
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="bg-white dark:bg-zinc-900 rounded-xl p-4 shadow-sm border border-zinc-200 dark:border-zinc-800">
          <div className="text-2xl font-bold text-zinc-900 dark:text-zinc-100">
            {formatNumber(stats.total)}
          </div>
          <div className="text-sm text-zinc-500">Total Species</div>
        </div>
        <div className="bg-white dark:bg-zinc-900 rounded-xl p-4 shadow-sm border border-zinc-200 dark:border-zinc-800">
          <div className="text-2xl font-bold text-zinc-900 dark:text-zinc-100">
            {formatNumber(stats.totalOccurrences)}
          </div>
          <div className="text-sm text-zinc-500">Total Occurrences</div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <div className="md:col-span-2 bg-white dark:bg-zinc-900 rounded-xl p-6 shadow-sm border border-zinc-200 dark:border-zinc-800">
          <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100 mb-4">
            Distribution Breakdown
          </h2>
          <div className="space-y-3">
            {[
              { label: "≤ 1", count: stats.distribution.lte1 },
              { label: "≤ 10", count: stats.distribution.lte10 },
              { label: "≤ 100", count: stats.distribution.lte100 },
              { label: "≤ 1,000", count: stats.distribution.lte1000 },
              { label: "≤ 10,000", count: stats.distribution.lte10000 },
            ].map(({ label, count }) => (
              <div key={label} className="flex items-center gap-4">
                <div className="w-20 text-sm text-zinc-600 dark:text-zinc-400">{label}</div>
                <div className="flex-1 bg-zinc-100 dark:bg-zinc-800 rounded-full h-4 overflow-hidden">
                  <div
                    className="bg-orange-500 h-full rounded-full transition-all duration-500"
                    style={{ width: `${(count / stats.total) * 100}%` }}
                  />
                </div>
                <div className="w-32 text-sm text-right text-zinc-600 dark:text-zinc-400">
                  {formatNumber(count)} ({getPercentage(count, stats.total)}%)
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="flex flex-col gap-4">
          <div className="bg-white dark:bg-zinc-900 rounded-xl p-4 shadow-sm border border-zinc-200 dark:border-zinc-800 flex-1">
            <div className="text-2xl font-bold text-zinc-900 dark:text-zinc-100">
              {formatNumber(stats.median)}
            </div>
            <div className="text-sm text-zinc-500">Median Occurrences</div>
          </div>
          <div className="bg-white dark:bg-zinc-900 rounded-xl p-4 shadow-sm border border-zinc-200 dark:border-zinc-800 flex-1">
            <div className="text-2xl font-bold text-zinc-900 dark:text-zinc-100">
              {formatNumber(Math.round(stats.totalOccurrences / stats.total))}
            </div>
            <div className="text-sm text-zinc-500">Mean Occurrences</div>
          </div>
        </div>
      </div>
    </>
  );
}

interface ExpandedRowProps {
  speciesKey: number;
  regionMode: RegionMode;
  mounted: boolean;
  colSpan: number;
}

function ExpandedRow({ speciesKey, regionMode, mounted, colSpan }: ExpandedRowProps) {
  const [occurrences, setOccurrences] = useState<OccurrenceFeature[]>([]);
  const [loadingOccurrences, setLoadingOccurrences] = useState(true);

  const config = REGION_CONFIG[regionMode];

  // Fetch occurrences immediately
  useEffect(() => {
    setLoadingOccurrences(true);
    fetch(`${config.occurrencesEndpoint}?speciesKey=${speciesKey}&limit=500`)
      .then((res) => res.json())
      .then((data) => setOccurrences(data.features || []))
      .catch(console.error)
      .finally(() => setLoadingOccurrences(false));
  }, [speciesKey, config.occurrencesEndpoint]);

  return (
    <tr>
      <td colSpan={colSpan} className="p-0">
        <div className="bg-zinc-50 dark:bg-zinc-800/50 border-t border-zinc-200 dark:border-zinc-700">
          <div className="p-2">
            {/* Map Only */}
            <div className="h-[300px] md:h-[400px] rounded-lg overflow-hidden border border-zinc-200 dark:border-zinc-700 relative">
              {loadingOccurrences ? (
                <div className="flex items-center justify-center h-full bg-zinc-100 dark:bg-zinc-800">
                  <div className="text-zinc-400">Loading occurrences...</div>
                </div>
              ) : mounted ? (
                <MapContainer
                  center={config.center}
                  zoom={config.zoom}
                  style={{ height: "100%", width: "100%" }}
                >
                  <TileLayer
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                  />
                  {config.bounds && (
                    <Rectangle
                      bounds={config.bounds}
                      pathOptions={{ color: "#22c55e", weight: 2, fillOpacity: 0.05 }}
                    />
                  )}
                  {occurrences.map((feature, idx) => {
                    const [lon, lat] = feature.geometry.coordinates;
                    return (
                      <CircleMarker
                        key={feature.properties.gbifID || idx}
                        center={[lat, lon]}
                        radius={6}
                        pathOptions={{
                          color: "#3b82f6",
                          fillColor: "#3b82f6",
                          fillOpacity: 0.7,
                          weight: 1,
                        }}
                      >
                        <Popup>
                          <div className="text-sm">
                            <div className="font-medium italic">{feature.properties.species}</div>
                            {feature.properties.eventDate && (
                              <div className="text-xs">{feature.properties.eventDate}</div>
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
                </div>
              )}
            </div>
          </div>
        </div>
      </td>
    </tr>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export default function Home() {
  // Region mode
  const [regionMode, setRegionMode] = useState<RegionMode>("global");

  // Data state
  const [data, setData] = useState<SpeciesRecord[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [pagination, setPagination] = useState({ page: 1, limit: 10, total: 0, totalPages: 0 });
  const [loading, setLoading] = useState(true);
  const [filterPreset, setFilterPreset] = useState<FilterPreset>("all");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc");

  // Species details cache (for images in table)
  const [speciesCache, setSpeciesCache] = useState<Record<number, SpeciesDetails>>({});
  const [loadingSpecies, setLoadingSpecies] = useState<Set<number>>(new Set());

  // Search
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<SpeciesDetails[] | null>(null);
  const [searching, setSearching] = useState(false);

  // Modal
  const [selectedSpeciesKey, setSelectedSpeciesKey] = useState<number | null>(null);
  const [selectedOccurrenceCount, setSelectedOccurrenceCount] = useState<number>(0);
  const [mounted, setMounted] = useState(false);

  const config = REGION_CONFIG[regionMode];

  useEffect(() => {
    setMounted(true);
  }, []);

  // Fetch data based on region mode
  const fetchData = useCallback(async () => {
    setLoading(true);
    const { minCount, maxCount } = FILTER_PRESETS[filterPreset];
    const params = new URLSearchParams({
      page: pagination.page.toString(),
      limit: pagination.limit.toString(),
      minCount: minCount.toString(),
      maxCount: maxCount.toString(),
      sort: sortOrder,
    });

    try {
      const response = await fetch(`${config.apiEndpoint}?${params}`);
      const result: ApiResponse = await response.json();

      setData(result.data);
      setStats(result.stats);
      setPagination(result.pagination);
    } catch (error) {
      console.error("Error fetching data:", error);
    } finally {
      setLoading(false);
    }
  }, [pagination.page, pagination.limit, filterPreset, sortOrder, config.apiEndpoint]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Auto-preload species details for visible rows (for images)
  useEffect(() => {
    const preloadSpeciesDetails = async () => {
      const keysToLoad = data
        .map((record) => record.species_key)
        .filter((key) => !speciesCache[key] && !loadingSpecies.has(key));

      if (keysToLoad.length === 0) return;

      setLoadingSpecies((prev) => new Set([...prev, ...keysToLoad]));

      const batchSize = 10;
      for (let i = 0; i < keysToLoad.length; i += batchSize) {
        const batch = keysToLoad.slice(i, i + batchSize);
        const results = await Promise.all(
          batch.map(async (speciesKey) => {
            try {
              const response = await fetch(`/api/species/${speciesKey}`);
              if (response.ok) {
                return await response.json();
              }
            } catch (error) {
              console.error(`Failed to fetch species ${speciesKey}:`, error);
            }
            return null;
          })
        );

        const newCache: Record<number, SpeciesDetails> = {};
        results.forEach((details) => {
          if (details) {
            newCache[details.key] = details;
          }
        });

        setSpeciesCache((prev) => ({ ...prev, ...newCache }));
      }

      setLoadingSpecies((prev) => {
        const next = new Set(prev);
        keysToLoad.forEach((key) => next.delete(key));
        return next;
      });
    };

    preloadSpeciesDetails();
  }, [data, speciesCache, loadingSpecies]);

  const handleFilterChange = (preset: FilterPreset) => {
    setFilterPreset(preset);
    setPagination((prev) => ({ ...prev, page: 1 }));
  };

  const handleRegionChange = (mode: RegionMode) => {
    setRegionMode(mode);
    setSearchResults(null);
    setSearchQuery("");
    setPagination((prev) => ({ ...prev, page: 1 }));
    setSelectedSpeciesKey(null);
  };

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchQuery.trim()) {
      setSearchResults(null);
      return;
    }

    setSearching(true);
    try {
      const response = await fetch(`${SEARCH_ENDPOINT}?q=${encodeURIComponent(searchQuery)}`);
      const data = await response.json();
      let results = data.results || [];

      // If in a regional mode, fetch region-specific occurrence counts
      if (regionMode !== "global" && results.length > 0) {
        const regionConfig = REGION_CONFIG[regionMode];
        const bounds = regionConfig.bounds;

        if (bounds) {
          const geometry = `POLYGON((${bounds[0][1]} ${bounds[0][0]}, ${bounds[1][1]} ${bounds[0][0]}, ${bounds[1][1]} ${bounds[1][0]}, ${bounds[0][1]} ${bounds[1][0]}, ${bounds[0][1]} ${bounds[0][0]}))`;

          // Fetch regional counts in parallel
          const countsPromises = results.map(async (species: SpeciesDetails) => {
            try {
              const countResponse = await fetch(
                `https://api.gbif.org/v1/occurrence/count?taxonKey=${species.key}&geometry=${encodeURIComponent(geometry)}&hasCoordinate=true&hasGeospatialIssue=false`
              );
              if (countResponse.ok) {
                const count = parseInt(await countResponse.text(), 10) || 0;
                return { ...species, occurrenceCount: count };
              }
            } catch {
              // Keep original count on error
            }
            return species;
          });

          results = await Promise.all(countsPromises);
          // Filter out species with 0 occurrences in the region
          results = results.filter((s: SpeciesDetails) => (s.occurrenceCount ?? 0) > 0);
        }
      }

      setSearchResults(results);
    } catch (error) {
      console.error("Search failed:", error);
      setSearchResults([]);
    }
    setSearching(false);
  };

  const clearSearch = () => {
    setSearchQuery("");
    setSearchResults(null);
  };

  const handleRowClick = (speciesKey: number, occurrenceCount: number) => {
    // Toggle: if clicking the same row, close it; otherwise open the new one
    if (selectedSpeciesKey === speciesKey) {
      setSelectedSpeciesKey(null);
    } else {
      setSelectedSpeciesKey(speciesKey);
      setSelectedOccurrenceCount(occurrenceCount);
    }
  };

  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-zinc-950 p-4 md:p-8">
      <main className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-zinc-900 dark:text-zinc-100 mb-2">
            Plant Species Data Explorer
          </h1>
          <p className="text-zinc-600 dark:text-zinc-400">
            {stats
              ? `Explore ${formatNumber(stats.total)} plant species - ${config.description}`
              : `Loading ${config.description}...`}
          </p>
        </div>

        {/* Region Toggle */}
        <div className="mb-6 flex gap-2">
          {(Object.keys(REGION_CONFIG) as RegionMode[]).map((mode) => (
            <button
              key={mode}
              onClick={() => handleRegionChange(mode)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2 ${
                regionMode === mode
                  ? "bg-green-600 text-white"
                  : "bg-white text-zinc-700 border border-zinc-200 hover:bg-zinc-50 dark:bg-zinc-800 dark:text-zinc-300 dark:border-zinc-700"
              }`}
            >
              {mode === "cambridge" && (
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
              )}
              {REGION_CONFIG[mode].label}
            </button>
          ))}
        </div>

        {/* Stats */}
        {stats && <StatsCards stats={stats} />}

        {/* Search */}
        <div className="mb-6">
          <form onSubmit={handleSearch} className="flex gap-2">
            <div className="relative flex-1 max-w-md">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search for a plant species..."
                className="w-full px-4 py-2 pl-10 rounded-lg border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100 placeholder-zinc-400 focus:outline-none focus:ring-2 focus:ring-green-500"
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
            <button
              type="submit"
              disabled={searching}
              className="px-4 py-2 rounded-lg text-sm font-medium bg-green-600 text-white hover:bg-green-700 disabled:opacity-50"
            >
              {searching ? "..." : "Search"}
            </button>
            {searchResults !== null && (
              <button
                type="button"
                onClick={clearSearch}
                className="px-4 py-2 rounded-lg text-sm font-medium bg-white text-zinc-700 border border-zinc-200 hover:bg-zinc-50 dark:bg-zinc-800 dark:text-zinc-300 dark:border-zinc-700"
              >
                Clear
              </button>
            )}
          </form>
        </div>

        {/* Filters */}
        {searchResults === null && (
          <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
            <div className="flex gap-2">
              {(Object.keys(FILTER_PRESETS) as FilterPreset[]).map((preset) => (
                <button
                  key={preset}
                  onClick={() => handleFilterChange(preset)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    filterPreset === preset
                      ? "bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900"
                      : "bg-white text-zinc-700 border border-zinc-200 hover:bg-zinc-50 dark:bg-zinc-800 dark:text-zinc-300 dark:border-zinc-700"
                  }`}
                >
                  {FILTER_PRESETS[preset].label}
                </button>
              ))}
            </div>
            <select
              value={sortOrder}
              onChange={(e) => setSortOrder(e.target.value as "asc" | "desc")}
              className="px-3 py-2 rounded-lg text-sm bg-white text-zinc-700 border border-zinc-200 dark:bg-zinc-800 dark:text-zinc-300 dark:border-zinc-700"
            >
              <option value="desc">Most occurrences</option>
              <option value="asc">Least occurrences</option>
            </select>
          </div>
        )}

        {/* Results count */}
        <div className="text-sm text-zinc-500 mb-4">
          {searchResults !== null
            ? `Found ${searchResults.length} species`
            : `Showing ${formatNumber(pagination.total)} species`}
        </div>

        {/* Table */}
        <div className="bg-white dark:bg-zinc-900 rounded-xl shadow-sm border border-zinc-200 dark:border-zinc-800 overflow-hidden mb-6">
          <table className="w-full">
            <thead className="bg-zinc-50 dark:bg-zinc-800">
              <tr>
                {searchResults === null && (
                  <th className="px-2 sm:px-4 py-3 text-left text-xs font-medium text-zinc-500 uppercase w-12 sm:w-20">
                    Rank
                  </th>
                )}
                <th className="hidden md:table-cell px-4 py-3 text-left text-xs font-medium text-zinc-500 uppercase w-20">
                  Image
                </th>
                <th className="px-2 sm:px-4 py-3 text-left text-xs font-medium text-zinc-500 uppercase">
                  Species
                </th>
                <th className="px-2 sm:px-4 py-3 text-right text-xs font-medium text-zinc-500 uppercase w-20 sm:w-28">
                  Count
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-200 dark:divide-zinc-800">
              {loading ? (
                <tr>
                  <td colSpan={6} className="px-4 py-8 text-center text-zinc-500">
                    Loading...
                  </td>
                </tr>
              ) : searchResults !== null ? (
                searchResults.map((species) => (
                  <React.Fragment key={species.key}>
                    <tr
                      className={`hover:bg-zinc-50 dark:hover:bg-zinc-800 cursor-pointer ${selectedSpeciesKey === species.key ? "bg-zinc-100 dark:bg-zinc-800" : ""}`}
                      onClick={() => handleRowClick(species.key, species.occurrenceCount || 0)}
                    >
                      <td className="hidden md:table-cell px-4 py-2">
                        {species.imageUrl ? (
                          <img src={species.imageUrl} alt="" className="w-16 h-16 object-cover rounded" />
                        ) : (
                          <div className="w-16 h-16 bg-zinc-100 dark:bg-zinc-800 rounded" />
                        )}
                      </td>
                      <td className="px-2 sm:px-4 py-2">
                        <div className="text-sm text-zinc-900 dark:text-zinc-100">{species.vernacularName || "—"}</div>
                        <div className="text-xs text-zinc-500 italic">
                          {species.canonicalName}
                          <a
                            href={`https://www.gbif.org/species/${species.key}`}
                            target="_blank"
                            rel="noopener noreferrer"
                            onClick={(e) => e.stopPropagation()}
                            className="ml-1 text-green-600 hover:text-green-700"
                          >
                            <svg className="w-3 h-3 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                            </svg>
                          </a>
                        </div>
                      </td>
                      <td className="px-2 sm:px-4 py-2 text-sm text-right font-medium text-zinc-900 dark:text-zinc-100">
                        {species.occurrenceCount ? formatNumber(species.occurrenceCount) : "—"}
                      </td>
                    </tr>
                    {selectedSpeciesKey === species.key && (
                      <ExpandedRow
                        speciesKey={species.key}
                        regionMode={regionMode}
                        mounted={mounted}
                        colSpan={5}
                      />
                    )}
                  </React.Fragment>
                ))
              ) : (
                data.map((record, index) => {
                  const rank =
                    sortOrder === "desc"
                      ? (pagination.page - 1) * pagination.limit + index + 1
                      : pagination.total - ((pagination.page - 1) * pagination.limit + index);
                  const cached = speciesCache[record.species_key];
                  const isLoading = loadingSpecies.has(record.species_key);

                  // Use cached data if available, otherwise use inline data from API
                  const displayName = cached?.canonicalName || record.canonicalName || (isLoading ? "Loading..." : "—");
                  const commonName = cached?.vernacularName || record.vernacularName || (isLoading ? "..." : "—");

                  return (
                    <React.Fragment key={record.species_key}>
                      <tr
                        className={`hover:bg-zinc-50 dark:hover:bg-zinc-800 cursor-pointer ${selectedSpeciesKey === record.species_key ? "bg-zinc-100 dark:bg-zinc-800" : ""}`}
                        onClick={() => handleRowClick(record.species_key, record.occurrence_count)}
                      >
                        <td className="px-2 sm:px-4 py-2 text-sm text-zinc-500">#{formatNumber(rank)}</td>
                        <td className="hidden md:table-cell px-4 py-2">
                          {isLoading ? (
                            <div className="w-16 h-16 bg-zinc-200 dark:bg-zinc-700 rounded animate-pulse" />
                          ) : cached?.imageUrl ? (
                            <img src={cached.imageUrl} alt="" className="w-16 h-16 object-cover rounded" />
                          ) : (
                            <div className="w-16 h-16 bg-zinc-100 dark:bg-zinc-800 rounded" />
                          )}
                        </td>
                        <td className="px-2 sm:px-4 py-2">
                          <div className="text-sm text-zinc-900 dark:text-zinc-100">{commonName}</div>
                          <div className="text-xs text-zinc-500 italic">
                            {displayName}
                            <a
                              href={`https://www.gbif.org/species/${record.species_key}`}
                              target="_blank"
                              rel="noopener noreferrer"
                              onClick={(e) => e.stopPropagation()}
                              className="ml-1 text-green-600 hover:text-green-700"
                            >
                              <svg className="w-3 h-3 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                              </svg>
                            </a>
                          </div>
                        </td>
                        <td className="px-2 sm:px-4 py-2 text-sm text-right font-medium text-zinc-900 dark:text-zinc-100">
                          {formatNumber(record.occurrence_count)}
                        </td>
                      </tr>
                      {selectedSpeciesKey === record.species_key && (
                        <ExpandedRow
                          speciesKey={record.species_key}
                          regionMode={regionMode}
                          mounted={mounted}
                          colSpan={6}
                        />
                      )}
                    </React.Fragment>
                  );
                })
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {searchResults === null && (
          <div className="flex items-center justify-between">
            <div className="text-sm text-zinc-500">
              Page {pagination.page} of {formatNumber(pagination.totalPages)}
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => setPagination((p) => ({ ...p, page: p.page - 1 }))}
                disabled={pagination.page <= 1}
                className="px-4 py-2 rounded-lg text-sm bg-white text-zinc-700 border border-zinc-200 disabled:opacity-50 dark:bg-zinc-800 dark:text-zinc-300 dark:border-zinc-700"
              >
                Previous
              </button>
              <button
                onClick={() => setPagination((p) => ({ ...p, page: p.page + 1 }))}
                disabled={pagination.page >= pagination.totalPages}
                className="px-4 py-2 rounded-lg text-sm bg-white text-zinc-700 border border-zinc-200 disabled:opacity-50 dark:bg-zinc-800 dark:text-zinc-300 dark:border-zinc-700"
              >
                Next
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

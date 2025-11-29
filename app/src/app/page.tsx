"use client";

import { useState, useEffect, useCallback } from "react";
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
type ModalTab = "details" | "map";

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

interface SpeciesModalProps {
  species: SpeciesDetails | null;
  speciesKey: number;
  occurrenceCount: number;
  regionMode: RegionMode;
  onClose: () => void;
  mounted: boolean;
}

function SpeciesModal({ species, speciesKey, occurrenceCount, regionMode, onClose, mounted }: SpeciesModalProps) {
  const [activeTab, setActiveTab] = useState<ModalTab>("details");
  const [occurrences, setOccurrences] = useState<OccurrenceFeature[]>([]);
  const [loadingOccurrences, setLoadingOccurrences] = useState(false);
  const [loadingDetails, setLoadingDetails] = useState(!species);
  const [details, setDetails] = useState<SpeciesDetails | null>(species);

  const config = REGION_CONFIG[regionMode];

  // Fetch species details if not provided
  useEffect(() => {
    if (!species && speciesKey) {
      setLoadingDetails(true);
      fetch(`/api/species/${speciesKey}`)
        .then((res) => res.json())
        .then((data) => setDetails(data))
        .catch(console.error)
        .finally(() => setLoadingDetails(false));
    }
  }, [species, speciesKey]);

  // Fetch occurrences when map tab is selected
  useEffect(() => {
    if (activeTab === "map" && occurrences.length === 0 && !loadingOccurrences) {
      setLoadingOccurrences(true);
      fetch(`${config.occurrencesEndpoint}?speciesKey=${speciesKey}&limit=500`)
        .then((res) => res.json())
        .then((data) => setOccurrences(data.features || []))
        .catch(console.error)
        .finally(() => setLoadingOccurrences(false));
    }
  }, [activeTab, speciesKey, config.occurrencesEndpoint, occurrences.length, loadingOccurrences]);

  const displaySpecies = details || species;
  const displayName = displaySpecies?.canonicalName || `Species ${speciesKey}`;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50" onClick={onClose}>
      <div
        className="bg-white dark:bg-zinc-900 rounded-2xl max-w-4xl w-full shadow-xl max-h-[90vh] overflow-hidden flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="p-6 border-b border-zinc-200 dark:border-zinc-800">
          <div className="flex justify-between items-start">
            <div>
              <h2 className="text-xl font-bold text-zinc-900 dark:text-zinc-100 italic">
                {loadingDetails ? "Loading..." : displayName}
              </h2>
              {displaySpecies?.vernacularName && (
                <p className="text-zinc-500">{displaySpecies.vernacularName}</p>
              )}
              <p className="text-sm text-zinc-400 mt-1">
                {formatNumber(occurrenceCount)} occurrences {regionMode === "cambridge" ? "in Cambridge" : "worldwide"}
              </p>
            </div>
            <button onClick={onClose} className="text-zinc-400 hover:text-zinc-600">
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Tabs */}
          <div className="flex gap-4 mt-4">
            <button
              onClick={() => setActiveTab("details")}
              className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                activeTab === "details"
                  ? "bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900"
                  : "text-zinc-600 hover:bg-zinc-100 dark:text-zinc-400 dark:hover:bg-zinc-800"
              }`}
            >
              Details
            </button>
            <button
              onClick={() => setActiveTab("map")}
              className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors flex items-center gap-2 ${
                activeTab === "map"
                  ? "bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900"
                  : "text-zinc-600 hover:bg-zinc-100 dark:text-zinc-400 dark:hover:bg-zinc-800"
              }`}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
              </svg>
              Occurrences Map
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto p-6">
          {activeTab === "details" ? (
            loadingDetails ? (
              <div className="flex items-center justify-center h-48">
                <div className="text-zinc-400">Loading species details...</div>
              </div>
            ) : displaySpecies ? (
              <div>
                {displaySpecies.imageUrl && (
                  <img
                    src={displaySpecies.imageUrl}
                    alt=""
                    className="w-full h-48 object-cover rounded-lg mb-6"
                  />
                )}
                <dl className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <dt className="text-zinc-500">Kingdom</dt>
                    <dd className="font-medium text-zinc-900 dark:text-zinc-100">
                      {displaySpecies.kingdom || "—"}
                    </dd>
                  </div>
                  <div>
                    <dt className="text-zinc-500">Family</dt>
                    <dd className="font-medium text-zinc-900 dark:text-zinc-100">
                      {displaySpecies.family || "—"}
                    </dd>
                  </div>
                  <div>
                    <dt className="text-zinc-500">Genus</dt>
                    <dd className="font-medium text-zinc-900 dark:text-zinc-100">
                      {displaySpecies.genus || "—"}
                    </dd>
                  </div>
                  <div>
                    <dt className="text-zinc-500">GBIF Key</dt>
                    <dd className="font-mono text-zinc-900 dark:text-zinc-100">
                      {displaySpecies.key}
                    </dd>
                  </div>
                </dl>
                <div className="mt-6">
                  <a
                    href={displaySpecies.gbifUrl || `https://www.gbif.org/species/${speciesKey}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
                  >
                    View on GBIF
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                    </svg>
                  </a>
                </div>
              </div>
            ) : (
              <div className="text-zinc-400">Unable to load species details</div>
            )
          ) : (
            /* Map Tab */
            <div className="h-[500px] rounded-lg overflow-hidden border border-zinc-200 dark:border-zinc-700">
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
              {!loadingOccurrences && occurrences.length > 0 && (
                <div className="absolute bottom-4 left-4 bg-white dark:bg-zinc-800 px-3 py-1 rounded-full text-sm text-zinc-600 dark:text-zinc-300 shadow">
                  {occurrences.length} occurrences shown
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
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
  const [refreshing, setRefreshing] = useState(false);
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

  const handleRefreshFromGBIF = async () => {
    if (refreshing || regionMode !== "global") return;

    const confirmed = window.confirm(
      "This will fetch fresh data from GBIF. This may take several minutes. Continue?"
    );
    if (!confirmed) return;

    setRefreshing(true);
    try {
      const response = await fetch("/api/refresh", { method: "POST" });
      const result = await response.json();

      if (result.success) {
        alert("Data refresh completed! Reloading...");
        fetchData();
      } else {
        alert(`Refresh failed: ${result.error}`);
      }
    } catch (error) {
      alert(`Refresh failed: ${error}`);
    }
    setRefreshing(false);
  };

  const handleRowClick = (speciesKey: number, occurrenceCount: number) => {
    setSelectedSpeciesKey(speciesKey);
    setSelectedOccurrenceCount(occurrenceCount);
  };

  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-zinc-950 p-4 md:p-8">
      <main className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-6 flex justify-between items-start">
          <div>
            <h1 className="text-3xl font-bold text-zinc-900 dark:text-zinc-100 mb-2">
              Plant Species Data Explorer
            </h1>
            <p className="text-zinc-600 dark:text-zinc-400">
              {stats
                ? `Explore ${formatNumber(stats.total)} plant species - ${config.description}`
                : `Loading ${config.description}...`}
            </p>
          </div>
          {regionMode === "global" && (
            <button
              onClick={handleRefreshFromGBIF}
              disabled={refreshing}
              className="px-4 py-2 rounded-lg text-sm font-medium bg-white text-zinc-700 border border-zinc-200 hover:bg-zinc-50 disabled:opacity-50 disabled:cursor-not-allowed dark:bg-zinc-800 dark:text-zinc-300 dark:border-zinc-700 dark:hover:bg-zinc-700 flex items-center gap-2"
            >
              <svg
                className={`w-4 h-4 ${refreshing ? "animate-spin" : ""}`}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                />
              </svg>
              {refreshing ? "Fetching..." : "Refresh"}
            </button>
          )}
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
                  <th className="px-4 py-3 text-left text-xs font-medium text-zinc-500 uppercase w-20">
                    Rank
                  </th>
                )}
                <th className="px-4 py-3 text-left text-xs font-medium text-zinc-500 uppercase w-20">
                  Image
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-zinc-500 uppercase">
                  Common Name
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-zinc-500 uppercase">
                  Species
                </th>
                <th className="px-4 py-3 text-right text-xs font-medium text-zinc-500 uppercase w-32">
                  Occurrences
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-200 dark:divide-zinc-800">
              {loading ? (
                <tr>
                  <td colSpan={5} className="px-4 py-8 text-center text-zinc-500">
                    Loading...
                  </td>
                </tr>
              ) : searchResults !== null ? (
                searchResults.map((species) => (
                  <tr
                    key={species.key}
                    className="hover:bg-zinc-50 dark:hover:bg-zinc-800 cursor-pointer"
                    onClick={() => handleRowClick(species.key, species.occurrenceCount || 0)}
                  >
                    <td className="px-4 py-2">
                      {species.imageUrl ? (
                        <img src={species.imageUrl} alt="" className="w-16 h-16 object-cover rounded" />
                      ) : (
                        <div className="w-16 h-16 bg-zinc-100 dark:bg-zinc-800 rounded" />
                      )}
                    </td>
                    <td className="px-4 py-2 text-sm text-zinc-600 dark:text-zinc-400">
                      {species.vernacularName || "—"}
                    </td>
                    <td className="px-4 py-2 text-sm italic text-zinc-900 dark:text-zinc-100">
                      {species.canonicalName}
                    </td>
                    <td className="px-4 py-2 text-sm text-right font-medium text-zinc-900 dark:text-zinc-100">
                      {species.occurrenceCount ? formatNumber(species.occurrenceCount) : "—"}
                    </td>
                  </tr>
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
                    <tr
                      key={record.species_key}
                      className="hover:bg-zinc-50 dark:hover:bg-zinc-800 cursor-pointer"
                      onClick={() => handleRowClick(record.species_key, record.occurrence_count)}
                    >
                      <td className="px-4 py-2 text-sm text-zinc-500">#{formatNumber(rank)}</td>
                      <td className="px-4 py-2">
                        {isLoading ? (
                          <div className="w-16 h-16 bg-zinc-200 dark:bg-zinc-700 rounded animate-pulse" />
                        ) : cached?.imageUrl ? (
                          <img src={cached.imageUrl} alt="" className="w-16 h-16 object-cover rounded" />
                        ) : (
                          <div className="w-16 h-16 bg-zinc-100 dark:bg-zinc-800 rounded" />
                        )}
                      </td>
                      <td className="px-4 py-2 text-sm text-zinc-600 dark:text-zinc-400">{commonName}</td>
                      <td className="px-4 py-2 text-sm italic text-zinc-900 dark:text-zinc-100">{displayName}</td>
                      <td className="px-4 py-2 text-sm text-right font-medium text-zinc-900 dark:text-zinc-100">
                        {formatNumber(record.occurrence_count)}
                      </td>
                    </tr>
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

        {/* Unified Species Modal */}
        {selectedSpeciesKey && (
          <SpeciesModal
            species={speciesCache[selectedSpeciesKey] || null}
            speciesKey={selectedSpeciesKey}
            occurrenceCount={selectedOccurrenceCount}
            regionMode={regionMode}
            onClose={() => setSelectedSpeciesKey(null)}
            mounted={mounted}
          />
        )}
      </main>
    </div>
  );
}

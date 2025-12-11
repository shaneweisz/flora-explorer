"use client";

import React, { useState, useEffect, useCallback } from "react";
import dynamic from "next/dynamic";
import Link from "next/link";
import { ThemeToggle } from "../../components/ThemeToggle";
import { getTaxonConfig, CATEGORY_COLORS } from "@/config/taxa";

// Dynamically import GBIFTaxaSummary component
const GBIFTaxaSummary = dynamic(
  () => import("../../components/gbif/GBIFTaxaSummary"),
  { ssr: false }
);

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

// Locate button component that uses useMap hook
const LocateControl = dynamic(
  () => import("../../components/LocateControl"),
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
  scientific_name?: string;
  redlist_category?: string | null;
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

interface Stats {
  total: number;
  filtered: number;
  totalOccurrences: number;
  median: number;
  distribution: {
    eq1: number;
    gt1_lte10: number;
    gt10_lte100: number;
    gt100_lte1000: number;
    gt1000_lte10000: number;
    gt10000: number;
  };
  redlist?: {
    assessed: number;
    notAssessed: number;
    assessedOccurrences: number;
    notAssessedOccurrences: number;
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

type FilterPreset = "all" | "eq1" | "gt1_lte10" | "gt10_lte100" | "gt100_lte1000" | "gt1000_lte10000" | "gt10000";
type RedlistFilter = "all" | "NE" | "DD" | "LC" | "NT" | "VU" | "EN" | "CR" | "EW" | "EX";
type RegionMode = "global" | "country";

// Dynamically import WorldMap component
const WorldMap = dynamic(
  () => import("../../components/WorldMap"),
  { ssr: false }
);


// ============================================================================
// Constants
// ============================================================================

const FILTER_PRESETS: Record<FilterPreset, { minCount: number; maxCount: number; label: string }> = {
  all: { minCount: 0, maxCount: 999999999, label: "All Species" },
  eq1: { minCount: 1, maxCount: 1, label: "= 1" },
  gt1_lte10: { minCount: 2, maxCount: 10, label: "2–10" },
  gt10_lte100: { minCount: 11, maxCount: 100, label: "11–100" },
  gt100_lte1000: { minCount: 101, maxCount: 1000, label: "101–1K" },
  gt1000_lte10000: { minCount: 1001, maxCount: 10000, label: "1K–10K" },
  gt10000: { minCount: 10001, maxCount: 999999999, label: "> 10K" },
};

// Shared search endpoint for all regions
const SEARCH_ENDPOINT = "/api/search";

// ============================================================================
// Helper Functions
// ============================================================================

const formatNumber = (num: number) => num.toLocaleString();
const getPercentage = (count: number, total: number) => ((count / total) * 100).toFixed(1);

// Color scale from grey (low prob) to red (high prob)
function getProbabilityColor(probability: number): string {
  // Grey: rgb(180, 180, 180) -> Red: rgb(220, 38, 38)
  const r = Math.round(180 + (220 - 180) * probability);
  const g = Math.round(180 - (180 - 38) * probability);
  const b = Math.round(180 - (180 - 38) * probability);
  return `rgb(${r}, ${g}, ${b})`;
}

// Get candidate key from canonical name (just lowercase it to match API format)
function getCandidateKey(canonicalName: string | undefined): string | undefined {
  if (!canonicalName) return undefined;
  return canonicalName.toLowerCase();
}

// ============================================================================
// Components
// ============================================================================

// Record type breakdown interface
interface RecordTypeBreakdown {
  humanObservation: number;
  preservedSpecimen: number;
  machineObservation: number;
  other: number;
  iNaturalist: number;
}

// Breakdown tooltip component (shown on hover over count)
function BreakdownTooltip({
  speciesKey,
  countryCode,
  totalCount,
  onDataLoaded,
}: {
  speciesKey: number;
  countryCode?: string | null;
  totalCount: number;
  onDataLoaded?: (data: RecordTypeBreakdown) => void;
}) {
  const [breakdown, setBreakdown] = useState<RecordTypeBreakdown | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchBreakdown = async () => {
      try {
        const params = new URLSearchParams();
        if (countryCode) params.set("country", countryCode);
        const res = await fetch(`/api/species/${speciesKey}/breakdown?${params}`);
        if (res.ok) {
          const data = await res.json();
          setBreakdown(data);
          onDataLoaded?.(data);
        }
      } catch (err) {
        console.error("Failed to fetch breakdown:", err);
      } finally {
        setLoading(false);
      }
    };
    fetchBreakdown();
  }, [speciesKey, countryCode, onDataLoaded]);

  return (
    <div className="bg-zinc-800 dark:bg-zinc-900 border border-zinc-700 rounded-lg shadow-lg p-2 text-xs text-left min-w-[200px]">
      <div className="text-zinc-300 font-medium mb-1">Breakdown by type:</div>
      {loading ? (
        <div className="text-zinc-400 animate-pulse">Loading...</div>
      ) : breakdown ? (
        <div className="space-y-0.5 text-zinc-400">
          <div className="flex justify-between">
            <span>Human observations</span>
            <a
              href={`https://www.gbif.org/occurrence/search?basis_of_record=HUMAN_OBSERVATION&taxon_key=${speciesKey}${countryCode ? `&country=${countryCode}` : ""}`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-zinc-200 hover:text-blue-400 hover:underline tabular-nums"
              onClick={(e) => e.stopPropagation()}
            >
              {breakdown.humanObservation.toLocaleString()}
            </a>
          </div>
          {breakdown.iNaturalist > 0 && (
            <div className="flex justify-between pl-3 text-[11px]">
              <span className="text-amber-400">iNaturalist</span>
              <a
                href={`https://www.gbif.org/occurrence/search?dataset_key=50c9509d-22c7-4a22-a47d-8c48425ef4a7&taxon_key=${speciesKey}${countryCode ? `&country=${countryCode}` : ""}`}
                target="_blank"
                rel="noopener noreferrer"
                className="text-zinc-300 hover:text-amber-400 hover:underline tabular-nums"
                onClick={(e) => e.stopPropagation()}
              >
                {breakdown.iNaturalist.toLocaleString()}
              </a>
            </div>
          )}
          <div className="flex justify-between">
            <span>Preserved specimens</span>
            <a
              href={`https://www.gbif.org/occurrence/search?basis_of_record=PRESERVED_SPECIMEN&taxon_key=${speciesKey}${countryCode ? `&country=${countryCode}` : ""}`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-zinc-200 hover:text-blue-400 hover:underline tabular-nums"
              onClick={(e) => e.stopPropagation()}
            >
              {breakdown.preservedSpecimen.toLocaleString()}
            </a>
          </div>
          <div className="flex justify-between">
            <span>Machine observations</span>
            <a
              href={`https://www.gbif.org/occurrence/search?basis_of_record=MACHINE_OBSERVATION&taxon_key=${speciesKey}${countryCode ? `&country=${countryCode}` : ""}`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-zinc-200 hover:text-blue-400 hover:underline tabular-nums"
              onClick={(e) => e.stopPropagation()}
            >
              {breakdown.machineObservation.toLocaleString()}
            </a>
          </div>
          {breakdown.other > 0 && (
            <div className="flex justify-between">
              <span>Other</span>
              <span className="text-zinc-200 tabular-nums">{breakdown.other.toLocaleString()}</span>
            </div>
          )}
        </div>
      ) : (
        <div className="text-zinc-500">No data available</div>
      )}
    </div>
  );
}

interface ExpandedRowProps {
  speciesKey: number;
  speciesName?: string;
  regionMode: RegionMode;
  countryCode?: string | null;
  mounted: boolean;
  colSpan: number;
  hasCandidates?: boolean;
}

function ExpandedRow({ speciesKey, speciesName, regionMode, countryCode, mounted, colSpan, hasCandidates }: ExpandedRowProps) {
  const [occurrences, setOccurrences] = useState<OccurrenceFeature[]>([]);
  const [candidates, setCandidates] = useState<CandidateFeature[]>([]);
  const [loadingOccurrences, setLoadingOccurrences] = useState(true);
  const [loadingCandidates, setLoadingCandidates] = useState(false);
  const [showCandidates, setShowCandidates] = useState(true);
  const [heatmapOpacity, setHeatmapOpacity] = useState(0.7);

  // Build occurrences endpoint - add country filter if in country mode
  const occurrencesEndpoint = "/api/occurrences";

  // Fetch occurrences immediately
  useEffect(() => {
    setLoadingOccurrences(true);
    const params = new URLSearchParams({
      speciesKey: speciesKey.toString(),
      limit: "500",
    });
    if (regionMode === "country" && countryCode) {
      params.set("country", countryCode);
    }
    fetch(`${occurrencesEndpoint}?${params}`)
      .then((res) => res.json())
      .then((data) => setOccurrences(data.features || []))
      .catch(console.error)
      .finally(() => setLoadingOccurrences(false));
  }, [speciesKey, occurrencesEndpoint, regionMode, countryCode]);

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
  const sortedCandidates = [...candidates].sort((a, b) => a.properties.probability - b.properties.probability);

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
                    Show heatmap ({loadingCandidates ? "..." : candidates.length} points)
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
                    <div className="w-12 h-3 rounded" style={{background: "linear-gradient(to right, rgb(180,180,180), rgb(220,38,38))"}} />
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
                  {showCandidates && sortedCandidates.map((feature, idx) => {
                    const [lon, lat] = feature.geometry.coordinates;
                    const prob = feature.properties.probability;
                    const color = getProbabilityColor(prob);
                    // Opacity scales with probability: low prob = more transparent
                    const opacity = 0.2 + (prob * 0.8 * heatmapOpacity);
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
                            <div className="font-medium text-orange-600">Predicted Location</div>
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
                            <div className="font-medium italic">{feature.properties.species}</div>
                            {feature.properties.basisOfRecord && (
                              <div className="text-xs text-gray-600">
                                {formatBasisOfRecord(feature.properties.basisOfRecord)}
                              </div>
                            )}
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
                  {hasCandidates && showCandidates && ` • ${candidates.length} predictions`}
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
  // Taxon selection (null = show summary view)
  const [selectedTaxon, setSelectedTaxon] = useState<string | null>(null);

  // Region mode
  const [regionMode, setRegionMode] = useState<RegionMode>("global");
  const [selectedCountry, setSelectedCountry] = useState<string | null>(null);
  const [selectedCountryName, setSelectedCountryName] = useState<string | null>(null);

  // Data state
  const [data, setData] = useState<SpeciesRecord[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [pagination, setPagination] = useState({ page: 1, limit: 10, total: 0, totalPages: 0 });
  const [loading, setLoading] = useState(true);
  const [filterPreset, setFilterPreset] = useState<FilterPreset>("all");
  const [redlistFilter, setRedlistFilter] = useState<RedlistFilter>("all");
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
  const [mounted, setMounted] = useState(false);

  // Candidates
  const [availableCandidates, setAvailableCandidates] = useState<string[]>([]);
  const [showOnlyWithCandidates, setShowOnlyWithCandidates] = useState(false);

  // Get taxon config
  const taxonConfig = selectedTaxon ? getTaxonConfig(selectedTaxon) : null;

  // Compute API endpoint based on mode and taxon
  const apiEndpoint = regionMode === "country" && selectedCountry
    ? `/api/country/${selectedCountry}/species`
    : "/api/species";

  useEffect(() => {
    setMounted(true);
  }, []);

  // Fetch available candidates list
  useEffect(() => {
    fetch("/api/candidates")
      .then((res) => res.json())
      .then((data) => {
        if (data.available) {
          setAvailableCandidates(data.available.map((s: string) => s.toLowerCase()));
        }
      })
      .catch(console.error);
  }, []);

  // Reset state when taxon changes
  useEffect(() => {
    setData([]);
    setStats(null);
    setPagination({ page: 1, limit: 10, total: 0, totalPages: 0 });
    setFilterPreset("all");
    setRedlistFilter("all");
    setSearchQuery("");
    setSearchResults(null);
    setSelectedSpeciesKey(null);
    setSpeciesCache({});
  }, [selectedTaxon]);

  // Fetch data based on region mode and taxon
  const fetchData = useCallback(async () => {
    // Don't fetch if no taxon selected (showing summary view)
    if (!selectedTaxon) {
      setLoading(false);
      setData([]);
      setStats(null);
      return;
    }

    // Don't fetch if in country mode but no country selected
    if (regionMode === "country" && !selectedCountry) {
      setLoading(false);
      setData([]);
      setStats(null);
      return;
    }

    setLoading(true);
    const { minCount, maxCount } = FILTER_PRESETS[filterPreset];
    const params = new URLSearchParams({
      taxon: selectedTaxon,
      page: pagination.page.toString(),
      limit: pagination.limit.toString(),
      minCount: minCount.toString(),
      maxCount: maxCount.toString(),
      sort: sortOrder,
    });

    // Add redlist filter if not "all"
    if (redlistFilter !== "all") {
      params.set("redlist", redlistFilter);
    }

    try {
      const response = await fetch(`${apiEndpoint}?${params}`);
      const result: ApiResponse = await response.json();

      setData(result.data);
      setStats(result.stats);
      setPagination(result.pagination);
    } catch (error) {
      console.error("Error fetching data:", error);
    } finally {
      setLoading(false);
    }
  }, [pagination.page, pagination.limit, filterPreset, redlistFilter, sortOrder, apiEndpoint, regionMode, selectedCountry, selectedTaxon]);

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

  const handleRedlistFilterChange = (filter: RedlistFilter) => {
    setRedlistFilter(filter);
    setPagination((prev) => ({ ...prev, page: 1 }));
  };

  const handleCountrySelect = (countryCode: string, countryName: string) => {
    // If clicking the same country, revert to global
    if (selectedCountry === countryCode) {
      setSelectedCountry(null);
      setSelectedCountryName(null);
      setRegionMode("global");
    } else {
      setSelectedCountry(countryCode);
      setSelectedCountryName(countryName);
      setRegionMode("country");
    }
    setSearchResults(null);
    setSearchQuery("");
    setPagination((prev) => ({ ...prev, page: 1 }));
    setSelectedSpeciesKey(null);
  };

  const handleClearCountry = () => {
    setSelectedCountry(null);
    setSelectedCountryName(null);
    setRegionMode("global");
    setPagination((prev) => ({ ...prev, page: 1 }));
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

      // If in country mode with a country selected, fetch country-specific occurrence counts
      if (regionMode === "country" && selectedCountry && results.length > 0) {
        // Fetch country-specific counts in parallel
        const countsPromises = results.map(async (species: SpeciesDetails) => {
          try {
            const countResponse = await fetch(
              `https://api.gbif.org/v1/occurrence/count?taxonKey=${species.key}&country=${selectedCountry}&hasCoordinate=true&hasGeospatialIssue=false`
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
        // Filter out species with 0 occurrences in the country
        results = results.filter((s: SpeciesDetails) => (s.occurrenceCount ?? 0) > 0);
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

  const handleRowClick = (speciesKey: number) => {
    // Toggle: if clicking the same row, close it; otherwise open the new one
    if (selectedSpeciesKey === speciesKey) {
      setSelectedSpeciesKey(null);
    } else {
      setSelectedSpeciesKey(speciesKey);
    }
  };

  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-zinc-950 p-4 md:p-8">
      <main className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-6 flex items-start justify-between">
          <div>
            <h1 className="text-3xl font-bold text-zinc-900 dark:text-zinc-100 mb-2">
              GBIF Dashboard
            </h1>
            <p className="text-zinc-600 dark:text-zinc-400">
              Click a taxon row for details, click again to return
            </p>
          </div>
          <div className="flex items-center gap-2">
            {/* Tab Navigation */}
            <div className="flex gap-1 bg-zinc-100 dark:bg-zinc-800 rounded-lg p-1">
              <Link
                href="/"
                className="px-4 py-1.5 rounded-md text-sm font-medium transition-colors text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-100"
              >
                Red List Dashboard
              </Link>
              <div className="px-4 py-1.5 rounded-md text-sm font-medium bg-white dark:bg-zinc-900 text-zinc-900 dark:text-zinc-100 shadow-sm">
                GBIF Dashboard
              </div>
            </div>
            <ThemeToggle />
            <a
              href="/experiment"
              className="p-2 text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-300 transition-colors"
              title="Classification Experiment"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
              </svg>
            </a>
          </div>
        </div>

        {/* Always show taxa summary */}
        <GBIFTaxaSummary
          onSelectTaxon={setSelectedTaxon}
          selectedTaxon={selectedTaxon}
        />

        {/* Show species details below when a taxon is selected */}
        {selectedTaxon && (
          <div className="mt-4 space-y-4">
            {/* Map and Distribution - 60% | 40% layout */}
            <div className="grid grid-cols-1 lg:grid-cols-5 gap-4 mb-4">
          {/* World Map - takes 3 columns (60%) */}
          {mounted && (
            <div className="lg:col-span-3">
              <WorldMap
                selectedCountry={selectedCountry}
                onCountrySelect={handleCountrySelect}
                onClearSelection={handleClearCountry}
                selectedTaxon={selectedTaxon}
              />
            </div>
          )}

          {/* Distribution breakdown - takes 2 columns (40%) */}
          {stats && (
            <div className="lg:col-span-2 bg-white dark:bg-zinc-900 rounded-xl p-4 shadow-sm border border-zinc-200 dark:border-zinc-800">
              <div className="flex items-center justify-between mb-3">
                <h2 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
                  Filter by Count
                </h2>
                {filterPreset !== "all" && (
                  <button
                    onClick={() => handleFilterChange("all")}
                    className="text-xs text-orange-600 hover:text-orange-700 dark:text-orange-400"
                  >
                    Clear filter
                  </button>
                )}
              </div>
              <div className="space-y-1.5">
                {[
                  { label: "= 1", count: stats.distribution.eq1, preset: "eq1" as FilterPreset },
                  { label: "2–10", count: stats.distribution.gt1_lte10, preset: "gt1_lte10" as FilterPreset },
                  { label: "11–100", count: stats.distribution.gt10_lte100, preset: "gt10_lte100" as FilterPreset },
                  { label: "101–1K", count: stats.distribution.gt100_lte1000, preset: "gt100_lte1000" as FilterPreset },
                  { label: "1K–10K", count: stats.distribution.gt1000_lte10000, preset: "gt1000_lte10000" as FilterPreset },
                  { label: "> 10K", count: stats.distribution.gt10000, preset: "gt10000" as FilterPreset },
                ].map(({ label, count, preset }) => {
                  const isActive = filterPreset === preset;
                  return (
                    <button
                      key={label}
                      onClick={() => handleFilterChange(isActive ? "all" : preset)}
                      className={`w-full flex items-center gap-2 p-1 rounded-lg transition-colors ${
                        isActive
                          ? "bg-orange-100 dark:bg-orange-900/30"
                          : "hover:bg-zinc-100 dark:hover:bg-zinc-800"
                      }`}
                    >
                      <div className="w-14 text-xs text-zinc-500 shrink-0 text-left">{label}</div>
                      <div className="flex-1 bg-zinc-100 dark:bg-zinc-800 rounded-full h-2 overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all duration-500 ${
                            isActive ? "bg-orange-600" : "bg-orange-500"
                          }`}
                          style={{ width: `${(count / stats.total) * 100}%` }}
                        />
                      </div>
                      <div className="w-20 text-[11px] text-right text-zinc-500 shrink-0">
                        {formatNumber(count)} ({getPercentage(count, stats.total)}%)
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* Search and Sort row */}
        <div className="flex flex-wrap items-center gap-2 mb-4">
          <form onSubmit={handleSearch} className="flex gap-2 flex-1">
            <div className="relative flex-1 max-w-md">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder={`Search for ${taxonConfig?.name.toLowerCase() || 'species'}...`}
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
          {/* Red List category filter */}
          <select
            value={redlistFilter}
            onChange={(e) => handleRedlistFilterChange(e.target.value as RedlistFilter)}
            className="px-3 py-2 rounded-lg text-sm bg-white text-zinc-700 border border-zinc-200 dark:bg-zinc-800 dark:text-zinc-300 dark:border-zinc-700"
          >
            <option value="all">All Categories</option>
            <option value="NE">NE - Not Evaluated</option>
            <option value="DD">DD - Data Deficient</option>
            <option value="LC">LC - Least Concern</option>
            <option value="NT">NT - Near Threatened</option>
            <option value="VU">VU - Vulnerable</option>
            <option value="EN">EN - Endangered</option>
            <option value="CR">CR - Critically Endangered</option>
            <option value="EW">EW - Extinct in Wild</option>
            <option value="EX">EX - Extinct</option>
          </select>
          {availableCandidates.length > 0 && (
            <button
              onClick={() => setShowOnlyWithCandidates(!showOnlyWithCandidates)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2 ${
                showOnlyWithCandidates
                  ? "bg-orange-500 text-white"
                  : "bg-white text-zinc-700 border border-zinc-200 hover:bg-zinc-50 dark:bg-zinc-800 dark:text-zinc-300 dark:border-zinc-700"
              }`}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
              </svg>
              Has Predictions
            </button>
          )}
          <select
            value={sortOrder}
            onChange={(e) => setSortOrder(e.target.value as "asc" | "desc")}
            className="px-3 py-2 rounded-lg text-sm bg-white text-zinc-700 border border-zinc-200 dark:bg-zinc-800 dark:text-zinc-300 dark:border-zinc-700"
          >
            <option value="desc">Most occurrences</option>
            <option value="asc">Least occurrences</option>
          </select>
        </div>

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
                <th className="hidden md:table-cell px-4 py-3 text-left text-xs font-medium text-zinc-500 uppercase w-14">
                  Image
                </th>
                <th className="px-2 sm:px-4 py-3 text-left text-xs font-medium text-zinc-500 uppercase">
                  Species
                </th>
                <th className="px-2 sm:px-4 py-3 text-right text-xs font-medium text-zinc-500 uppercase w-20 sm:w-28">
                  Count
                </th>
                <th className="hidden sm:table-cell px-2 sm:px-4 py-3 text-center text-xs font-medium text-zinc-500 uppercase w-16">
                  Red List
                </th>
                <th className="hidden sm:table-cell px-2 sm:px-4 py-3 text-center text-xs font-medium text-zinc-500 uppercase w-14">
                  GBIF
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-200 dark:divide-zinc-800">
              {loading ? (
                <tr>
                  <td colSpan={7} className="px-4 py-8 text-center text-zinc-500">
                    Loading...
                  </td>
                </tr>
              ) : searchResults !== null ? (
                searchResults.map((species) => (
                  <React.Fragment key={species.key}>
                    <tr
                      className={`hover:bg-zinc-50 dark:hover:bg-zinc-800 cursor-pointer ${selectedSpeciesKey === species.key ? "bg-zinc-100 dark:bg-zinc-800" : ""}`}
                      onClick={() => handleRowClick(species.key)}
                    >
                      <td className="hidden md:table-cell px-4 py-2">
                        {species.imageUrl ? (
                          <img
                            src={species.imageUrl}
                            alt=""
                            className="w-10 h-10 object-cover rounded cursor-pointer hover:ring-2 hover:ring-green-400"
                            onMouseEnter={(e) => {
                              const img = e.currentTarget;
                              const rect = img.getBoundingClientRect();
                              const preview = document.getElementById('gbif-image-preview');
                              if (preview) {
                                (preview as HTMLImageElement).src = species.imageUrl || '';
                                preview.style.display = 'block';
                                preview.style.top = `${rect.top - 192 - 8}px`;
                                preview.style.left = `${rect.left}px`;
                              }
                            }}
                            onMouseLeave={() => {
                              const preview = document.getElementById('gbif-image-preview');
                              if (preview) {
                                preview.style.display = 'none';
                              }
                            }}
                          />
                        ) : (
                          <div className="w-10 h-10 bg-zinc-100 dark:bg-zinc-800 rounded" />
                        )}
                      </td>
                      <td className="px-2 sm:px-4 py-2">
                        <div className="text-sm text-zinc-900 dark:text-zinc-100">
                          <span className="italic">{species.canonicalName}</span>
                          {species.vernacularName && <span className="text-zinc-500 ml-1">({species.vernacularName})</span>}
                        </div>
                      </td>
                      <td className="px-2 sm:px-4 py-2 text-sm text-right font-medium text-zinc-900 dark:text-zinc-100">
                        {species.occurrenceCount ? (
                          <span className="relative group/count">
                            <a
                              href={`https://www.gbif.org/occurrence/search?taxon_key=${species.key}${regionMode === "country" && selectedCountry ? `&country=${selectedCountry}` : ""}`}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 underline decoration-dotted hover:decoration-solid"
                              onClick={(e) => e.stopPropagation()}
                            >
                              {formatNumber(species.occurrenceCount)}
                            </a>
                            <div className="absolute right-0 bottom-full z-10 hidden group-hover/count:block pb-2 pr-2 -mr-2">
                              <BreakdownTooltip
                                speciesKey={species.key}
                                countryCode={regionMode === "country" ? selectedCountry : null}
                                totalCount={species.occurrenceCount}
                              />
                            </div>
                          </span>
                        ) : "—"}
                      </td>
                      <td className="hidden sm:table-cell px-2 sm:px-4 py-2 text-center">
                        <a
                          href={`https://www.gbif.org/species/${species.key}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          onClick={(e) => e.stopPropagation()}
                          className="text-green-600 hover:text-green-700"
                        >
                          <svg className="w-4 h-4 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                          </svg>
                        </a>
                      </td>
                    </tr>
                    {selectedSpeciesKey === species.key && (
                      <ExpandedRow
                        speciesKey={species.key}
                        speciesName={getCandidateKey(species.canonicalName)}
                        regionMode={regionMode}
                        countryCode={selectedCountry}
                        mounted={mounted}
                        colSpan={5}
                        hasCandidates={!!getCandidateKey(species.canonicalName) && availableCandidates.includes(getCandidateKey(species.canonicalName) ?? "")}
                      />
                    )}
                  </React.Fragment>
                ))
              ) : (
                data
                  .filter((record) => {
                    if (!showOnlyWithCandidates) return true;
                    const name = speciesCache[record.species_key]?.canonicalName || record.canonicalName;
                    const candidateKey = getCandidateKey(name);
                    return candidateKey && availableCandidates.includes(candidateKey);
                  })
                  .map((record, index) => {
                  const rank =
                    sortOrder === "desc"
                      ? (pagination.page - 1) * pagination.limit + index + 1
                      : pagination.total - ((pagination.page - 1) * pagination.limit + index);
                  const cached = speciesCache[record.species_key];
                  const isLoading = loadingSpecies.has(record.species_key);

                  // Use cached data if available, otherwise use inline data from API
                  const displayName = cached?.canonicalName || record.canonicalName || (isLoading ? "Loading..." : "—");
                  const commonName = cached?.vernacularName || record.vernacularName || (isLoading ? "..." : "—");
                  const candidateKey = getCandidateKey(displayName);
                  const hasCandidates = !!(candidateKey && availableCandidates.includes(candidateKey));

                  return (
                    <React.Fragment key={record.species_key}>
                      <tr
                        className={`hover:bg-zinc-50 dark:hover:bg-zinc-800 cursor-pointer ${selectedSpeciesKey === record.species_key ? "bg-zinc-100 dark:bg-zinc-800" : ""}`}
                        onClick={() => handleRowClick(record.species_key)}
                      >
                        <td className="px-2 sm:px-4 py-2 text-sm text-zinc-500">#{formatNumber(rank)}</td>
                        <td className="hidden md:table-cell px-4 py-2">
                          {isLoading ? (
                            <div className="w-10 h-10 bg-zinc-200 dark:bg-zinc-700 rounded animate-pulse" />
                          ) : cached?.imageUrl ? (
                            <img
                              src={cached.imageUrl}
                              alt=""
                              className="w-10 h-10 object-cover rounded cursor-pointer hover:ring-2 hover:ring-green-400"
                              onMouseEnter={(e) => {
                                const img = e.currentTarget;
                                const rect = img.getBoundingClientRect();
                                const preview = document.getElementById('gbif-image-preview');
                                if (preview) {
                                  (preview as HTMLImageElement).src = cached.imageUrl || '';
                                  preview.style.display = 'block';
                                  preview.style.top = `${rect.top - 192 - 8}px`;
                                  preview.style.left = `${rect.left}px`;
                                }
                              }}
                              onMouseLeave={() => {
                                const preview = document.getElementById('gbif-image-preview');
                                if (preview) {
                                  preview.style.display = 'none';
                                }
                              }}
                            />
                          ) : (
                            <div className="w-10 h-10 bg-zinc-100 dark:bg-zinc-800 rounded" />
                          )}
                        </td>
                        <td className="px-2 sm:px-4 py-2">
                          <div className="flex items-center gap-2">
                            <span className="text-sm text-zinc-900 dark:text-zinc-100">
                              <span className="italic">{displayName}</span>
                              {commonName && commonName !== "—" && <span className="text-zinc-500 ml-1">({commonName})</span>}
                            </span>
                            {hasCandidates && (
                              <span className="px-1.5 py-0.5 text-[10px] font-medium bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400 rounded">
                                AI
                              </span>
                            )}
                          </div>
                        </td>
                        <td className="px-2 sm:px-4 py-2 text-sm text-right font-medium text-zinc-900 dark:text-zinc-100">
                          <span className="relative group/count">
                            <a
                              href={`https://www.gbif.org/occurrence/search?taxon_key=${record.species_key}${regionMode === "country" && selectedCountry ? `&country=${selectedCountry}` : ""}`}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 underline decoration-dotted hover:decoration-solid"
                              onClick={(e) => e.stopPropagation()}
                            >
                              {formatNumber(record.occurrence_count)}
                            </a>
                            <div className="absolute right-0 bottom-full z-10 hidden group-hover/count:block pb-2 pr-2 -mr-2">
                              <BreakdownTooltip
                                speciesKey={record.species_key}
                                countryCode={regionMode === "country" ? selectedCountry : null}
                                totalCount={record.occurrence_count}
                              />
                            </div>
                          </span>
                        </td>
                        <td className="hidden sm:table-cell px-2 sm:px-4 py-2 text-center">
                          {record.redlist_category ? (
                            <a
                              href={`https://www.iucnredlist.org/search?query=${encodeURIComponent(displayName)}`}
                              target="_blank"
                              rel="noopener noreferrer"
                              onClick={(e) => e.stopPropagation()}
                              className="hover:opacity-80 transition-opacity"
                            >
                              <span
                                className="px-1.5 py-0.5 text-xs font-medium rounded"
                                style={{
                                  backgroundColor: CATEGORY_COLORS[record.redlist_category] || "#6b7280",
                                  color: ["LC", "NT", "VU"].includes(record.redlist_category) ? "#000" : "#fff",
                                }}
                              >
                                {record.redlist_category}
                              </span>
                            </a>
                          ) : (
                            <span className="px-1.5 py-0.5 text-xs font-medium rounded bg-zinc-200 dark:bg-zinc-700 text-zinc-600 dark:text-zinc-400">NE</span>
                          )}
                        </td>
                        <td className="hidden sm:table-cell px-2 sm:px-4 py-2 text-center">
                          <a
                            href={`https://www.gbif.org/species/${record.species_key}`}
                            target="_blank"
                            rel="noopener noreferrer"
                            onClick={(e) => e.stopPropagation()}
                            className="text-green-600 hover:text-green-700"
                          >
                            <svg className="w-4 h-4 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                            </svg>
                          </a>
                        </td>
                      </tr>
                      {selectedSpeciesKey === record.species_key && (
                        <ExpandedRow
                          speciesKey={record.species_key}
                          speciesName={candidateKey}
                          regionMode={regionMode}
                          countryCode={selectedCountry}
                          mounted={mounted}
                          colSpan={7}
                          hasCandidates={hasCandidates}
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
          </div>
        )}
      </main>

      {/* Fixed image preview portal */}
      <img
        id="gbif-image-preview"
        alt=""
        className="fixed z-[9999] w-48 h-48 object-cover rounded shadow-xl border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-900 pointer-events-none"
        style={{ display: 'none' }}
      />
    </div>
  );
}

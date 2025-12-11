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
import TaxaSummary from "./TaxaSummary";
import { CATEGORY_COLORS } from "@/config/taxa";

interface CategoryStats {
  code: string;
  name: string;
  count: number;
  color: string;
}

interface TaxonInfo {
  id: string;
  name: string;
  estimatedDescribed: number;
  estimatedSource: string;
  color: string;
}

interface StatsResponse {
  totalAssessed: number;
  byCategory: CategoryStats[];
  sampleSize: number;
  lastUpdated: string;
  cached: boolean;
  error?: string;
  taxon?: TaxonInfo;
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

interface PreviousAssessment {
  year: string;
  assessment_id: number;
  category: string;
}

interface Species {
  sis_taxon_id: number;
  assessment_id: number;
  scientific_name: string;
  family: string | null;
  category: string;
  assessment_date: string | null;
  year_published: string;
  url: string;
  population_trend: string | null;
  countries: string[];
  assessment_count: number;
  previous_assessments: PreviousAssessment[];
}

interface GbifByRecordType {
  humanObservation: number;
  preservedSpecimen: number;
  machineObservation: number;
  other: number;
  iNaturalist?: number;
}

interface InatObservation {
  url: string;
  date: string | null;
  imageUrl: string | null;
  location: string | null;
  observer: string | null;
}

interface SpeciesDetails {
  criteria: string | null;
  commonName: string | null;
  gbifUrl: string | null;
  gbifOccurrences: number | null;
  gbifOccurrencesSinceAssessment: number | null;
  gbifByRecordType: GbifByRecordType | null;
  gbifNewByRecordType: GbifByRecordType | null;
  recentInatObservations: InatObservation[];
  inatTotalCount: number;
}

interface SpeciesResponse {
  species: Species[];
  total: number;
  error?: string;
  taxon?: TaxonInfo;
}

interface RedListViewProps {
  onTaxonChange?: (taxonName: string | null) => void;
}

// Component for iNaturalist observation preview with navigation
function InatObservationPreview({
  observations,
  currentIndex,
  onNavigate,
  totalCount,
}: {
  observations: InatObservation[];
  currentIndex: number;
  onNavigate: (delta: number) => void;
  totalCount: number;
}) {
  if (observations.length === 0) return null;
  const obs = observations[currentIndex] || observations[0];
  if (!obs?.imageUrl) return null;

  return (
    <div className="bg-zinc-900 border border-zinc-700 rounded-lg shadow-xl p-2 w-56">
      {/* Navigation header */}
      {observations.length > 1 && (
        <div className="flex items-center justify-between mb-2 text-[10px] text-zinc-400">
          <button
            type="button"
            onMouseDown={(e) => { e.preventDefault(); e.stopPropagation(); onNavigate(-1); }}
            disabled={currentIndex === 0}
            className="p-1 hover:bg-zinc-800 rounded disabled:opacity-30 disabled:cursor-not-allowed"
          >
            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
          </button>
          <span>{currentIndex + 1} / {observations.length}{totalCount > observations.length ? ` of ${totalCount.toLocaleString()}` : ''}</span>
          <button
            type="button"
            onMouseDown={(e) => { e.preventDefault(); e.stopPropagation(); onNavigate(1); }}
            disabled={currentIndex >= observations.length - 1}
            className="p-1 hover:bg-zinc-800 rounded disabled:opacity-30 disabled:cursor-not-allowed"
          >
            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </button>
        </div>
      )}
      <a href={obs.url} target="_blank" rel="noopener noreferrer" onClick={(e) => e.stopPropagation()}>
        <img
          src={obs.imageUrl.replace('/original.', '/medium.')}
          alt="iNaturalist observation"
          className="w-full h-40 object-cover rounded mb-2 hover:opacity-90"
        />
      </a>
      <div className="text-[10px] text-zinc-300 space-y-0.5">
        {obs.date && <div>{obs.date}</div>}
        {obs.observer && <div className="truncate">{obs.observer}</div>}
        {obs.location && <div className="truncate text-zinc-400">{obs.location}</div>}
      </div>
    </div>
  );
}

export default function RedListView({ onTaxonChange }: RedListViewProps) {
  // Selected taxon (null = show summary table), synced with URL hash
  // Initialize to null to avoid hydration mismatch, then read hash in useEffect
  const [selectedTaxon, setSelectedTaxon] = useState<string | null>(null);

  // Read initial hash and sync URL hash with selected taxon
  useEffect(() => {
    // Read initial hash on mount
    const hash = window.location.hash.slice(1);
    if (hash) {
      setSelectedTaxon(hash);
    }

    const handleHashChange = () => {
      const newHash = window.location.hash.slice(1);
      setSelectedTaxon(newHash || null);
    };

    window.addEventListener("hashchange", handleHashChange);
    return () => window.removeEventListener("hashchange", handleHashChange);
  }, []);

  // Update URL hash when taxon changes (and push to history for back button)
  const handleTaxonSelect = (taxonId: string | null) => {
    if (taxonId) {
      window.history.pushState(null, "", `#${taxonId}`);
    } else {
      window.history.pushState(null, "", window.location.pathname);
    }
    setSelectedTaxon(taxonId);
  };
  const [taxonInfo, setTaxonInfo] = useState<TaxonInfo | null>(null);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [assessments, setAssessments] = useState<AssessmentsResponse | null>(null);
  const [species, setSpecies] = useState<Species[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filters (multi-select using Sets)
  const [selectedCategories, setSelectedCategories] = useState<Set<string>>(new Set());
  const [selectedYearRanges, setSelectedYearRanges] = useState<Set<string>>(new Set());
  const [selectedAssessmentCounts, setSelectedAssessmentCounts] = useState<Set<string>>(new Set());
  const [searchQuery, setSearchQuery] = useState("");

  // Sorting
  type SortField = "year" | "category" | null;
  type SortDirection = "asc" | "desc";
  const [sortField, setSortField] = useState<SortField>("year");
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc");

  // Pagination
  const [currentPage, setCurrentPage] = useState(1);
  const PAGE_SIZE = 10;

  // Species details cache (images, criteria, common names)
  const [speciesDetails, setSpeciesDetails] = useState<Record<number, SpeciesDetails>>({});

  // Track current iNat observation index per species (for navigation)
  const [inatIndex, setInatIndex] = useState<Record<number, number>>({});

  // Load stats and assessments when taxon changes
  useEffect(() => {
    // If no taxon selected, don't fetch detailed data
    if (!selectedTaxon) {
      setLoading(false);
      setStats(null);
      setAssessments(null);
      setSpecies([]);
      setTaxonInfo(null);
      return;
    }

    async function fetchData() {
      setLoading(true);
      setError(null);

      try {
        const taxonParam = `?taxon=${selectedTaxon}`;
        const [statsRes, assessmentsRes, speciesRes] = await Promise.all([
          fetch(`/api/redlist/stats${taxonParam}`),
          fetch(`/api/redlist/assessments${taxonParam}`),
          fetch(`/api/redlist/species${taxonParam}`),
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
        setTaxonInfo(statsData.taxon || null);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load data");
      } finally {
        setLoading(false);
      }
    }

    fetchData();
  }, [selectedTaxon]);

  // Notify parent when taxon changes
  useEffect(() => {
    onTaxonChange?.(taxonInfo?.name || null);
  }, [taxonInfo, onTaxonChange]);

  // Reset filters when taxon changes
  useEffect(() => {
    setSelectedCategories(new Set());
    setSelectedYearRanges(new Set());
    setSelectedAssessmentCounts(new Set());
    setSearchQuery("");
    setCurrentPage(1);
    setSpeciesDetails({});
  }, [selectedTaxon]);

  // Helper to check if species matches year range filter (based on assessment date)
  const matchesYearRangeFilter = (assessmentDate: string | null): boolean => {
    if (selectedYearRanges.size === 0) return true;
    if (!assessmentDate) return false;
    const currentYear = new Date().getFullYear();
    const assessmentYear = new Date(assessmentDate).getFullYear();
    const yearsSince = currentYear - assessmentYear;

    // Check if matches ANY of the selected ranges
    for (const range of selectedYearRanges) {
      switch (range) {
        case "0-1 years": if (yearsSince <= 1) return true; break;
        case "2-5 years": if (yearsSince >= 2 && yearsSince <= 5) return true; break;
        case "6-10 years": if (yearsSince >= 6 && yearsSince <= 10) return true; break;
        case "11-20 years": if (yearsSince >= 11 && yearsSince <= 20) return true; break;
        case "20+ years": if (yearsSince > 20) return true; break;
      }
    }
    return false;
  };

  // Helper to check if species matches assessment count filter
  const matchesAssessmentCountFilter = (count: number): boolean => {
    if (selectedAssessmentCounts.size === 0) return true;
    // Check if matches ANY of the selected counts
    for (const selected of selectedAssessmentCounts) {
      switch (selected) {
        case "1": if (count === 1) return true; break;
        case "2": if (count === 2) return true; break;
        case "3": if (count === 3) return true; break;
        case "4+": if (count >= 4) return true; break;
      }
    }
    return false;
  };

  // Filter species based on category, year range, assessment count, and search
  const filteredSpecies = species.filter((s) => {
    const matchesCategory = selectedCategories.size === 0 || selectedCategories.has(s.category);
    const matchesYear = matchesYearRangeFilter(s.assessment_date);
    const matchesAssessment = matchesAssessmentCountFilter(s.assessment_count);
    const matchesSearch = !searchQuery ||
      s.scientific_name.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesCategory && matchesYear && matchesAssessment && matchesSearch;
  });

  // Calculate reassessment distribution from species data
  const totalSpecies = species.length;
  const reassessmentDistribution = [
    { range: "1", count: species.filter(s => s.assessment_count === 1).length },
    { range: "2", count: species.filter(s => s.assessment_count === 2).length },
    { range: "3", count: species.filter(s => s.assessment_count === 3).length },
    { range: "4+", count: species.filter(s => s.assessment_count >= 4).length },
  ].map(item => ({
    ...item,
    label: `${item.count.toLocaleString()} (${totalSpecies > 0 ? ((item.count / totalSpecies) * 100).toFixed(1) : 0}%)`
  }));

  // Category order for sorting (most threatened first)
  const CATEGORY_ORDER: Record<string, number> = {
    EX: 0, EW: 1, CR: 2, EN: 3, VU: 4, NT: 5, LC: 6, DD: 7,
  };

  // Sort filtered species
  const sortedSpecies = [...filteredSpecies].sort((a, b) => {
    if (!sortField) return 0;

    let comparison = 0;
    if (sortField === "year") {
      // Sort by assessment date
      const dateA = a.assessment_date ? new Date(a.assessment_date).getTime() : 0;
      const dateB = b.assessment_date ? new Date(b.assessment_date).getTime() : 0;
      comparison = dateA - dateB;
    } else if (sortField === "category") {
      comparison = (CATEGORY_ORDER[a.category] ?? 99) - (CATEGORY_ORDER[b.category] ?? 99);
    }

    return sortDirection === "asc" ? comparison : -comparison;
  });

  // Pagination calculations
  const totalPages = Math.ceil(sortedSpecies.length / PAGE_SIZE);
  const paginatedSpecies = sortedSpecies.slice(
    (currentPage - 1) * PAGE_SIZE,
    currentPage * PAGE_SIZE
  );

  // Handle sort toggle
  const handleSort = (field: "year" | "category") => {
    if (sortField === field) {
      // Toggle direction or clear sort
      if (sortDirection === "desc") {
        setSortDirection("asc");
      } else {
        setSortField(null);
      }
    } else {
      setSortField(field);
      setSortDirection("desc");
    }
    setCurrentPage(1);
  };

  // Reset to page 1 when filters change
  useEffect(() => {
    setCurrentPage(1);
  }, [selectedCategories, selectedYearRanges, selectedAssessmentCounts, searchQuery]);

  // Fetch details for visible species
  useEffect(() => {
    async function fetchDetails() {
      const speciesToFetch = paginatedSpecies.filter(
        (s) => !speciesDetails[s.sis_taxon_id]
      );

      if (speciesToFetch.length === 0) return;

      const detailPromises = speciesToFetch.map(async (s) => {
        try {
          // Extract assessment year and month for GBIF filtering
          const assessmentDate = s.assessment_date ? new Date(s.assessment_date) : null;
          const assessmentYear = assessmentDate ? assessmentDate.getFullYear().toString() : "";
          const assessmentMonth = assessmentDate ? (assessmentDate.getMonth() + 1).toString() : ""; // 1-12
          const res = await fetch(
            `/api/redlist/species/${s.sis_taxon_id}?assessmentId=${s.assessment_id}&name=${encodeURIComponent(s.scientific_name)}&assessmentYear=${assessmentYear}&assessmentMonth=${assessmentMonth}`
          );
          if (res.ok) {
            const data = await res.json();
            return { id: s.sis_taxon_id, data };
          }
        } catch {
          // Ignore errors for individual species
        }
        return null;
      });

      const results = await Promise.all(detailPromises);
      const newDetails: Record<number, SpeciesDetails> = {};

      results.forEach((result) => {
        if (result) {
          newDetails[result.id] = {
            criteria: result.data.criteria,
            commonName: result.data.commonName,
            gbifUrl: result.data.gbifUrl,
            gbifOccurrences: result.data.gbifOccurrences,
            gbifOccurrencesSinceAssessment: result.data.gbifOccurrencesSinceAssessment,
            gbifByRecordType: result.data.gbifByRecordType,
            gbifNewByRecordType: result.data.gbifNewByRecordType,
            recentInatObservations: result.data.recentInatObservations || [],
            inatTotalCount: result.data.inatTotalCount || 0,
          };
        }
      });

      if (Object.keys(newDetails).length > 0) {
        setSpeciesDetails((prev) => ({ ...prev, ...newDetails }));
      }
    }

    if (paginatedSpecies.length > 0) {
      fetchDetails();
    }
  }, [paginatedSpecies, speciesDetails]);

  // Handle category bar click (Cmd/Ctrl+click for multi-select, regular click replaces)
  const handleCategoryClick = (data: { payload?: { code?: string } }, event: React.MouseEvent) => {
    const code = data.payload?.code;
    if (!code) return;
    const isMultiSelect = event.metaKey || event.ctrlKey;
    setSelectedCategories(prev => {
      if (isMultiSelect) {
        // Toggle in/out of set
        const next = new Set(prev);
        if (next.has(code)) {
          next.delete(code);
        } else {
          next.add(code);
        }
        return next;
      } else {
        // Single select: toggle off if already selected, otherwise replace
        if (prev.size === 1 && prev.has(code)) {
          return new Set();
        }
        return new Set([code]);
      }
    });
  };

  // Handle year range bar click (Cmd/Ctrl+click for multi-select, regular click replaces)
  const handleYearClick = (data: { payload?: { range?: string } }, event: React.MouseEvent) => {
    const range = data.payload?.range;
    if (!range) return;
    const isMultiSelect = event.metaKey || event.ctrlKey;
    setSelectedYearRanges(prev => {
      if (isMultiSelect) {
        const next = new Set(prev);
        if (next.has(range)) {
          next.delete(range);
        } else {
          next.add(range);
        }
        return next;
      } else {
        if (prev.size === 1 && prev.has(range)) {
          return new Set();
        }
        return new Set([range]);
      }
    });
  };

  // Handle assessment count bar click (Cmd/Ctrl+click for multi-select, regular click replaces)
  const handleAssessmentCountClick = (data: { payload?: { range?: string } }, event: React.MouseEvent) => {
    const range = data.payload?.range;
    if (!range) return;
    const isMultiSelect = event.metaKey || event.ctrlKey;
    setSelectedAssessmentCounts(prev => {
      if (isMultiSelect) {
        const next = new Set(prev);
        if (next.has(range)) {
          next.delete(range);
        } else {
          next.add(range);
        }
        return next;
      } else {
        if (prev.size === 1 && prev.has(range)) {
          return new Set();
        }
        return new Set([range]);
      }
    });
  };

  // Render loading state for details section
  const renderDetailsLoading = () => (
    <div className="flex flex-col items-center justify-center py-12">
      <div className="animate-spin h-8 w-8 border-4 border-red-600 border-t-transparent rounded-full" />
      <p className="mt-4 text-zinc-500 dark:text-zinc-400">
        Loading Red List statistics...
      </p>
    </div>
  );

  // Render error state for details section
  const renderDetailsError = () => (
    <div className="bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 px-6 py-4 rounded-lg">
      <p className="font-medium">Failed to load Red List data</p>
      <p className="text-sm mt-1">{error}</p>
    </div>
  );

  // Calculate values only when data is available
  const threatenedCount = stats?.byCategory
    .filter((c) => ["CR", "EN", "VU"].includes(c.code))
    .reduce((sum, c) => sum + c.count, 0) ?? 0;

  const categoryDataWithPercent = stats?.byCategory.map((cat) => ({
    ...cat,
    percent: ((cat.count / stats.sampleSize) * 100).toFixed(1),
    label: `${cat.count} (${((cat.count / stats.sampleSize) * 100).toFixed(1)}%)`,
  })) ?? [];

  const outdatedCount = assessments?.yearsSinceAssessment
    .filter((y) => y.minYear > 10)
    .reduce((sum, y) => sum + y.count, 0) ?? 0;
  const outdatedPercent = assessments && stats ? ((outdatedCount / assessments.sampleSize) * 100).toFixed(0) : "0";

  const assessedPercent = stats && taxonInfo ? ((stats.sampleSize / taxonInfo.estimatedDescribed) * 100).toFixed(1) : "0";

  const currentYear = new Date().getFullYear();

  return (
    <div className="space-y-4">
      {/* Always show Taxa Summary table */}
      <TaxaSummary
        onSelectTaxon={handleTaxonSelect}
        selectedTaxon={selectedTaxon}
      />

      {/* Show details below when a taxon is selected */}
      {selectedTaxon && (
        <div className="space-y-3">
          {/* Loading state */}
          {loading && renderDetailsLoading()}

          {/* Error state */}
          {error && renderDetailsError()}

          {/* Details content */}
          {!loading && !error && stats && assessments && taxonInfo && (
            <>
              {/* Three charts side by side */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Number of Assessments chart - horizontal bars */}
        <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl p-3 flex flex-col">
          <div className="flex items-center justify-between mb-1">
            <h3 className="text-sm font-medium text-zinc-700 dark:text-zinc-300">
              # of Assessments <span className="font-normal text-zinc-400">(click to filter)</span>
            </h3>
            {selectedAssessmentCounts.size > 0 && (
              <button
                onClick={() => setSelectedAssessmentCounts(new Set())}
                className="text-xs text-purple-600 hover:text-purple-700 dark:text-purple-400"
              >
                Clear
              </button>
            )}
          </div>
          <div className="flex-1 min-h-[130px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={reassessmentDistribution}
                layout="vertical"
                margin={{ top: 5, right: 85, left: 5, bottom: 5 }}
                barCategoryGap={4}
              >
                <XAxis type="number" hide />
                <YAxis
                  type="category"
                  dataKey="range"
                  tick={{ fontSize: 11, fill: "#a1a1aa" }}
                  tickLine={false}
                  axisLine={false}
                  width={26}
                />
                <Tooltip
                  formatter={(value: number) => [value, "Species"]}
                  contentStyle={{
                    backgroundColor: "#18181b",
                    border: "1px solid #3f3f46",
                    borderRadius: "8px",
                  }}
                  itemStyle={{ color: "#fff" }}
                  labelStyle={{ color: "#a1a1aa" }}
                />
                <Bar
                  dataKey="count"
                  radius={[0, 4, 4, 0]}
                  cursor="pointer"
                  onClick={(data, _index, event) => handleAssessmentCountClick(data, event)}
                >
                  {reassessmentDistribution.map((entry, index) => (
                    <Cell
                      key={`assessment-cell-${index}`}
                      fill="#8b5cf6"
                      opacity={selectedAssessmentCounts.size > 0 && !selectedAssessmentCounts.has(entry.range) ? 0.3 : 1}
                    />
                  ))}
                  <LabelList
                    dataKey="label"
                    position="right"
                    style={{ fontSize: 11, fill: "#a1a1aa" }}
                  />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Years Since Assessment chart - horizontal bars */}
        <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl p-3 flex flex-col">
          <div className="flex items-center justify-between mb-1">
            <h3 className="text-sm font-medium text-zinc-700 dark:text-zinc-300">
              Years Since Last Assessed <span className="font-normal text-zinc-400">(click to filter)</span>
            </h3>
            {selectedYearRanges.size > 0 && (
              <button
                onClick={() => setSelectedYearRanges(new Set())}
                className="text-xs text-blue-600 hover:text-blue-700 dark:text-blue-400"
              >
                Clear
              </button>
            )}
          </div>
          <div className="flex-1 min-h-[150px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={assessments.yearsSinceAssessment.map(y => {
                  const totalYears = assessments.yearsSinceAssessment.reduce((sum, item) => sum + item.count, 0);
                  return {
                    ...y,
                    shortRange: y.range.replace(' years', 'y').replace('20+y', '>20y'),
                    label: `${y.count.toLocaleString()} (${totalYears > 0 ? ((y.count / totalYears) * 100).toFixed(1) : 0}%)`
                  };
                })}
                layout="vertical"
                margin={{ top: 5, right: 85, left: 5, bottom: 5 }}
                barCategoryGap={4}
              >
                <XAxis type="number" hide />
                <YAxis
                  type="category"
                  dataKey="shortRange"
                  tick={{ fontSize: 11, fill: "#a1a1aa" }}
                  tickLine={false}
                  axisLine={false}
                  width={36}
                />
                <Tooltip
                  formatter={(value: number) => [value, "Species"]}
                  contentStyle={{
                    backgroundColor: "#18181b",
                    border: "1px solid #3f3f46",
                    borderRadius: "8px",
                  }}
                  itemStyle={{ color: "#fff" }}
                  labelStyle={{ color: "#a1a1aa" }}
                />
                <Bar
                  dataKey="count"
                  radius={[0, 4, 4, 0]}
                  cursor="pointer"
                  onClick={(data, _index, event) => handleYearClick(data, event)}
                >
                  {assessments.yearsSinceAssessment.map((entry, index) => (
                    <Cell
                      key={`year-cell-${index}`}
                      fill="#3b82f6"
                      opacity={selectedYearRanges.size > 0 && !selectedYearRanges.has(entry.range) ? 0.3 : 1}
                    />
                  ))}
                  <LabelList
                    dataKey="label"
                    position="right"
                    style={{ fontSize: 11, fill: "#a1a1aa" }}
                  />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Category distribution - horizontal bars */}
        <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl p-3 flex flex-col">
          <div className="flex items-center justify-between mb-1">
            <h3 className="text-sm font-medium text-zinc-700 dark:text-zinc-300">
              Distribution by Category <span className="font-normal text-zinc-400">(click to filter)</span>
            </h3>
            {selectedCategories.size > 0 && (
              <button
                onClick={() => setSelectedCategories(new Set())}
                className="text-xs text-red-600 hover:text-red-700 dark:text-red-400"
              >
                Clear
              </button>
            )}
          </div>
          <div className="flex-1 min-h-[200px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={categoryDataWithPercent}
                layout="vertical"
                margin={{ top: 5, right: 75, left: 5, bottom: 5 }}
                barCategoryGap={4}
              >
                <XAxis type="number" hide />
                <YAxis
                  type="category"
                  dataKey="code"
                  tick={{ fontSize: 11, fill: "#a1a1aa" }}
                  tickLine={false}
                  axisLine={false}
                  width={26}
                  interval={0}
                />
                <Tooltip
                  formatter={(value: number) => [`${value} species`, "Count"]}
                  contentStyle={{
                    backgroundColor: "#18181b",
                    border: "1px solid #3f3f46",
                    borderRadius: "8px",
                  }}
                  itemStyle={{ color: "#fff" }}
                  labelStyle={{ color: "#a1a1aa" }}
                />
                <Bar
                  dataKey="count"
                  radius={[0, 4, 4, 0]}
                  cursor="pointer"
                  onClick={(data, _index, event) => handleCategoryClick(data, event)}
                >
                  {categoryDataWithPercent.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={entry.color}
                      opacity={selectedCategories.size > 0 && !selectedCategories.has(entry.code) ? 0.3 : 1}
                    />
                  ))}
                  <LabelList
                    dataKey="label"
                    position="right"
                    style={{ fontSize: 11, fill: "#a1a1aa" }}
                  />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
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
                placeholder="Search by scientific name..."
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
            {Array.from(selectedCategories).map(cat => (
              <button
                key={cat}
                onClick={() => setSelectedCategories(prev => { const next = new Set(prev); next.delete(cat); return next; })}
                className="px-3 py-1 text-sm rounded-full flex items-center gap-1 hover:opacity-80"
                style={{ backgroundColor: CATEGORY_COLORS[cat] + "20", color: CATEGORY_COLORS[cat] }}
              >
                {cat}
                <span className="text-xs">×</span>
              </button>
            ))}
            {Array.from(selectedYearRanges).map(range => (
              <button
                key={range}
                onClick={() => setSelectedYearRanges(prev => { const next = new Set(prev); next.delete(range); return next; })}
                className="px-3 py-1 text-sm rounded-full bg-blue-100 text-blue-600 dark:bg-blue-900/30 dark:text-blue-400 flex items-center gap-1 hover:opacity-80"
              >
                {range}
                <span className="text-xs">×</span>
              </button>
            ))}
            {Array.from(selectedAssessmentCounts).map(count => (
              <button
                key={count}
                onClick={() => setSelectedAssessmentCounts(prev => { const next = new Set(prev); next.delete(count); return next; })}
                className="px-3 py-1 text-sm rounded-full bg-purple-100 text-purple-600 dark:bg-purple-900/30 dark:text-purple-400 flex items-center gap-1 hover:opacity-80"
              >
                {count} assessment{count !== "1" ? "s" : ""}
                <span className="text-xs">×</span>
              </button>
            ))}
            {(selectedCategories.size > 0 || selectedYearRanges.size > 0 || selectedAssessmentCounts.size > 0) && (
              <button
                onClick={() => { setSelectedCategories(new Set()); setSelectedYearRanges(new Set()); setSelectedAssessmentCounts(new Set()); }}
                className="text-xs text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300 underline"
              >
                Clear all
              </button>
            )}
            <span className="text-sm text-zinc-500">
              {filteredSpecies.length} species
            </span>
          </div>
        </div>

        {/* Species table */}
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-zinc-50 dark:bg-zinc-800">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-zinc-500 uppercase tracking-wider">
                  Species
                </th>
                <th
                  className="px-4 py-3 text-left text-xs font-medium text-zinc-500 uppercase tracking-wider cursor-pointer hover:text-zinc-700 dark:hover:text-zinc-300 select-none"
                  onClick={() => handleSort("category")}
                >
                  <span className="flex items-center gap-1">
                    Category
                    {sortField === "category" && (
                      <span className="text-red-500">{sortDirection === "desc" ? "↓" : "↑"}</span>
                    )}
                  </span>
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-zinc-500 uppercase tracking-wider">
                  Criteria
                </th>
                <th
                  className="px-4 py-3 text-left text-xs font-medium text-zinc-500 uppercase tracking-wider cursor-pointer hover:text-zinc-700 dark:hover:text-zinc-300 select-none"
                  onClick={() => handleSort("year")}
                >
                  <span className="flex items-center gap-1">
                    Assessment Date
                    {sortField === "year" && (
                      <span className="text-red-500">{sortDirection === "desc" ? "↓" : "↑"}</span>
                    )}
                  </span>
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-zinc-500 uppercase tracking-wider">
                  Published Year
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-zinc-500 uppercase tracking-wider">
                  Previous Assessments
                </th>
                <th className="px-4 py-3 text-right text-xs font-medium text-zinc-500 uppercase tracking-wider">
                  GBIF Occurrences
                </th>
                <th className="px-4 py-3 text-right text-xs font-medium text-zinc-500 uppercase tracking-wider">
                  GBIF Occurrences Since Assessed
                </th>
                <th className="px-4 py-3 text-center text-xs font-medium text-zinc-500 uppercase tracking-wider">
                  Red List
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-200 dark:divide-zinc-800">
              {paginatedSpecies.map((s) => {
                const assessmentDateObj = s.assessment_date ? new Date(s.assessment_date) : null;
                const assessmentYear = assessmentDateObj ? assessmentDateObj.getFullYear() : null;
                const assessmentMonth = assessmentDateObj ? assessmentDateObj.getMonth() + 1 : null; // 1-12
                const yearsSinceAssessment = assessmentYear ? currentYear - assessmentYear : null;
                const details = speciesDetails[s.sis_taxon_id];
                return (
                  <tr key={s.sis_taxon_id} className="hover:bg-zinc-50 dark:hover:bg-zinc-800/50">
                    <td className="px-4 py-3">
                      <a
                        href={s.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="italic font-medium text-zinc-900 dark:text-zinc-100 hover:text-red-600 dark:hover:text-red-400 hover:underline"
                      >
                        {s.scientific_name}
                      </a>
                      {details?.commonName && (
                        <span className="text-zinc-500 dark:text-zinc-400 text-sm ml-2">
                          ({details.commonName})
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-3">
                      <span
                        className="px-2 py-0.5 text-xs font-medium rounded"
                        style={{
                          backgroundColor: CATEGORY_COLORS[s.category] + "20",
                          color: s.category === "EX" || s.category === "EW" ? "#fff" : CATEGORY_COLORS[s.category],
                          ...(s.category === "EX" || s.category === "EW" ? { backgroundColor: CATEGORY_COLORS[s.category] } : {})
                        }}
                      >
                        {s.category}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-zinc-500 dark:text-zinc-400 text-xs">
                      {details?.criteria || "—"}
                    </td>
                    <td className="px-4 py-3 text-zinc-600 dark:text-zinc-400">
                      <a
                        href={s.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="hover:text-red-500 hover:underline"
                      >
                        {s.assessment_date
                          ? new Date(s.assessment_date).toLocaleDateString("en-GB", {
                              day: "numeric",
                              month: "short",
                              year: "numeric",
                            })
                          : "—"}
                      </a>
                      {yearsSinceAssessment !== null && yearsSinceAssessment > 10 && (
                        <span className="ml-1 text-xs text-amber-600">({yearsSinceAssessment}y ago)</span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-zinc-600 dark:text-zinc-400">
                      {s.year_published}
                    </td>
                    <td className="px-4 py-3 text-zinc-500 dark:text-zinc-400 text-sm">
                      {s.previous_assessments.length > 0
                        ? s.previous_assessments.map((pa, idx) => (
                            <span key={pa.assessment_id}>
                              <a
                                href={`https://www.iucnredlist.org/species/${s.sis_taxon_id}/${pa.assessment_id}`}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="hover:underline"
                              >
                                {pa.year}
                                <span className="ml-0.5 text-[10px]">
                                  ({pa.category})
                                </span>
                              </a>
                              {idx < s.previous_assessments.length - 1 && ", "}
                            </span>
                          ))
                        : "—"}
                    </td>
                    <td className="px-4 py-3 text-right text-zinc-600 dark:text-zinc-400 text-sm tabular-nums">
                      {details === undefined ? (
                        <span className="text-zinc-400 animate-pulse">...</span>
                      ) : details?.gbifOccurrences != null && details?.gbifUrl ? (
                        <span className="relative group/gbif">
                          <a
                            href={`https://www.gbif.org/occurrence/search?taxon_key=${details.gbifUrl.split('/').pop()}`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 underline decoration-dotted hover:decoration-solid"
                          >
                            {details.gbifOccurrences.toLocaleString()}
                          </a>
                          {details?.gbifByRecordType && (
                            <div className="absolute right-0 top-full z-10 hidden group-hover/gbif:block pt-2 pr-2 -mr-2">
                              <div className="bg-zinc-800 dark:bg-zinc-900 border border-zinc-700 rounded-lg shadow-lg p-2 text-xs text-left min-w-[200px]">
                                <div className="text-zinc-300 font-medium mb-1">Breakdown by type:</div>
                                <div className="space-y-0.5 text-zinc-400">
                                  <div className="flex justify-between">
                                    <span>Human observations</span>
                                    <a
                                      href={`https://www.gbif.org/occurrence/search?basis_of_record=HUMAN_OBSERVATION&taxon_key=${details.gbifUrl?.split('/').pop()}`}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      className="text-zinc-200 hover:text-blue-400 hover:underline tabular-nums"
                                      onClick={(e) => e.stopPropagation()}
                                    >
                                      {details.gbifByRecordType.humanObservation.toLocaleString()}
                                    </a>
                                  </div>
                                  {details.recentInatObservations.length > 0 && details.inatTotalCount > 0 && (
                                    <div className="flex justify-between pl-3 text-[11px]">
                                      <span className="relative group/inat">
                                        <span className="text-amber-400 hover:text-amber-300 cursor-pointer">iNaturalist</span>
                                        <div className="absolute right-0 bottom-full z-20 hidden group-hover/inat:block pb-2">
                                          <InatObservationPreview
                                            observations={details.recentInatObservations}
                                            currentIndex={inatIndex[s.sis_taxon_id] || 0}
                                            onNavigate={(delta) => setInatIndex(prev => ({
                                              ...prev,
                                              [s.sis_taxon_id]: Math.max(0, Math.min(details.recentInatObservations.length - 1, (prev[s.sis_taxon_id] || 0) + delta))
                                            }))}
                                            totalCount={details.inatTotalCount}
                                          />
                                        </div>
                                      </span>
                                      <a
                                        href={`https://www.gbif.org/occurrence/search?dataset_key=50c9509d-22c7-4a22-a47d-8c48425ef4a7&taxon_key=${details.gbifUrl?.split('/').pop()}`}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="text-zinc-300 hover:text-amber-400 hover:underline tabular-nums"
                                        onClick={(e) => e.stopPropagation()}
                                      >
                                        {details.inatTotalCount.toLocaleString()}
                                      </a>
                                    </div>
                                  )}
                                  <div className="flex justify-between">
                                    <span>Preserved specimens</span>
                                    <a
                                      href={`https://www.gbif.org/occurrence/search?basis_of_record=PRESERVED_SPECIMEN&taxon_key=${details.gbifUrl?.split('/').pop()}`}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      className="text-zinc-200 hover:text-blue-400 hover:underline tabular-nums"
                                      onClick={(e) => e.stopPropagation()}
                                    >
                                      {details.gbifByRecordType.preservedSpecimen.toLocaleString()}
                                    </a>
                                  </div>
                                  <div className="flex justify-between">
                                    <span>Machine observations</span>
                                    <a
                                      href={`https://www.gbif.org/occurrence/search?basis_of_record=MACHINE_OBSERVATION&taxon_key=${details.gbifUrl?.split('/').pop()}`}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      className="text-zinc-200 hover:text-blue-400 hover:underline tabular-nums"
                                      onClick={(e) => e.stopPropagation()}
                                    >
                                      {details.gbifByRecordType.machineObservation.toLocaleString()}
                                    </a>
                                  </div>
                                  {details.gbifByRecordType.other > 0 && (
                                    <div className="flex justify-between">
                                      <span>Other</span>
                                      <span className="text-zinc-200 tabular-nums">{details.gbifByRecordType.other.toLocaleString()}</span>
                                    </div>
                                  )}
                                </div>
                              </div>
                            </div>
                          )}
                        </span>
                      ) : details?.gbifOccurrences != null ? (
                        details.gbifOccurrences.toLocaleString()
                      ) : "—"}
                    </td>
                    <td className="px-4 py-3 text-right text-zinc-600 dark:text-zinc-400 text-sm tabular-nums">
                      {details === undefined ? (
                        <span className="text-zinc-400 animate-pulse">...</span>
                      ) : details?.gbifOccurrencesSinceAssessment != null && details?.gbifUrl && assessmentYear ? (
                        <span className="relative group/newgbif">
                          <a
                            href={`https://www.gbif.org/occurrence/search?taxon_key=${details.gbifUrl.split('/').pop()}&year=${assessmentYear + 1},${new Date().getFullYear()}`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 underline decoration-dotted hover:decoration-solid"
                            title={assessmentMonth ? `Data count includes ${assessmentYear} from month ${assessmentMonth + 1} onwards` : undefined}
                          >
                            {details.gbifOccurrencesSinceAssessment.toLocaleString()}
                          </a>
                          {details?.gbifNewByRecordType && (
                            <div className="absolute right-0 top-full z-10 hidden group-hover/newgbif:block pt-2 pr-2 -mr-2">
                              <div className="bg-zinc-800 dark:bg-zinc-900 border border-zinc-700 rounded-lg shadow-lg p-2 text-xs text-left min-w-[200px]">
                                <div className="text-zinc-300 font-medium mb-1">After {assessmentYear}:</div>
                                <div className="space-y-0.5 text-zinc-400">
                                  <div className="flex justify-between">
                                    <span>Human observations</span>
                                    <a
                                      href={`https://www.gbif.org/occurrence/search?basis_of_record=HUMAN_OBSERVATION&taxon_key=${details.gbifUrl?.split('/').pop()}&year=${assessmentYear + 1},${new Date().getFullYear()}`}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      className="text-zinc-200 hover:text-blue-400 hover:underline tabular-nums"
                                      onClick={(e) => e.stopPropagation()}
                                    >
                                      {details.gbifNewByRecordType.humanObservation.toLocaleString()}
                                    </a>
                                  </div>
                                  {details.gbifNewByRecordType.iNaturalist != null && details.gbifNewByRecordType.iNaturalist > 0 && details.recentInatObservations.length > 0 && (
                                    <div className="flex justify-between pl-3 text-[11px]">
                                      <span className="relative group/inat2">
                                        <span className="text-amber-400 hover:text-amber-300 cursor-pointer">iNaturalist</span>
                                        <div className="absolute right-0 bottom-full z-20 hidden group-hover/inat2:block pb-2">
                                          <InatObservationPreview
                                            observations={details.recentInatObservations}
                                            currentIndex={inatIndex[s.sis_taxon_id] || 0}
                                            onNavigate={(delta) => setInatIndex(prev => ({
                                              ...prev,
                                              [s.sis_taxon_id]: Math.max(0, Math.min(details.recentInatObservations.length - 1, (prev[s.sis_taxon_id] || 0) + delta))
                                            }))}
                                            totalCount={details.inatTotalCount}
                                          />
                                        </div>
                                      </span>
                                      <a
                                        href={`https://www.gbif.org/occurrence/search?dataset_key=50c9509d-22c7-4a22-a47d-8c48425ef4a7&taxon_key=${details.gbifUrl?.split('/').pop()}&year=${assessmentYear + 1},${new Date().getFullYear()}`}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="text-zinc-300 hover:text-amber-400 hover:underline tabular-nums"
                                        onClick={(e) => e.stopPropagation()}
                                      >
                                        {details.gbifNewByRecordType.iNaturalist.toLocaleString()}
                                      </a>
                                    </div>
                                  )}
                                  <div className="flex justify-between">
                                    <span>Preserved specimens</span>
                                    <a
                                      href={`https://www.gbif.org/occurrence/search?basis_of_record=PRESERVED_SPECIMEN&taxon_key=${details.gbifUrl?.split('/').pop()}&year=${assessmentYear + 1},${new Date().getFullYear()}`}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      className="text-zinc-200 hover:text-blue-400 hover:underline tabular-nums"
                                      onClick={(e) => e.stopPropagation()}
                                    >
                                      {details.gbifNewByRecordType.preservedSpecimen.toLocaleString()}
                                    </a>
                                  </div>
                                  <div className="flex justify-between">
                                    <span>Machine observations</span>
                                    <a
                                      href={`https://www.gbif.org/occurrence/search?basis_of_record=MACHINE_OBSERVATION&taxon_key=${details.gbifUrl?.split('/').pop()}&year=${assessmentYear + 1},${new Date().getFullYear()}`}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      className="text-zinc-200 hover:text-blue-400 hover:underline tabular-nums"
                                      onClick={(e) => e.stopPropagation()}
                                    >
                                      {details.gbifNewByRecordType.machineObservation.toLocaleString()}
                                    </a>
                                  </div>
                                  {details.gbifNewByRecordType.other > 0 && (
                                    <div className="flex justify-between">
                                      <span>Other</span>
                                      <span className="text-zinc-200 tabular-nums">{details.gbifNewByRecordType.other.toLocaleString()}</span>
                                    </div>
                                  )}
                                </div>
                              </div>
                            </div>
                          )}
                        </span>
                      ) : details?.gbifOccurrencesSinceAssessment != null ? (
                        details.gbifOccurrencesSinceAssessment.toLocaleString()
                      ) : "—"}
                    </td>
                    <td className="px-4 py-3 text-center">
                      <a
                        href={s.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center justify-center text-red-600 dark:text-red-400 hover:text-red-700 dark:hover:text-red-300"
                        title="View on IUCN Red List"
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                        </svg>
                      </a>
                    </td>
                  </tr>
                );
              })}
              {filteredSpecies.length === 0 && (
                <tr>
                  <td colSpan={9} className="px-4 py-8 text-center text-zinc-500">
                    No species found
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between px-4 py-3 border-t border-zinc-200 dark:border-zinc-800">
            <div className="text-sm text-zinc-500">
              Showing {(currentPage - 1) * PAGE_SIZE + 1}-{Math.min(currentPage * PAGE_SIZE, filteredSpecies.length)} of {filteredSpecies.length}
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                disabled={currentPage === 1}
                className="px-3 py-1 text-sm rounded-lg border border-zinc-200 dark:border-zinc-700 disabled:opacity-50 disabled:cursor-not-allowed hover:bg-zinc-50 dark:hover:bg-zinc-800"
              >
                Previous
              </button>
              <span className="text-sm text-zinc-600 dark:text-zinc-400">
                Page {currentPage} of {totalPages}
              </span>
              <button
                onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                disabled={currentPage === totalPages}
                className="px-3 py-1 text-sm rounded-lg border border-zinc-200 dark:border-zinc-700 disabled:opacity-50 disabled:cursor-not-allowed hover:bg-zinc-50 dark:hover:bg-zinc-800"
              >
                Next
              </button>
            </div>
          </div>
        )}
      </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

"use client";

import { useState, useEffect } from "react";
import { FaInfoCircle } from "react-icons/fa";
import TaxaIcon from "@/components/TaxaIcon";

const IUCN_SOURCE_URL = "https://nc.iucnredlist.org/redlist/content/attachment_files/2025-2_RL_Table1a.pdf";

interface TaxonSummary {
  id: string;
  name: string;
  color: string;
  estimatedDescribed: number;
  estimatedSource: string;
  estimatedSourceUrl?: string;
  available: boolean;
  totalAssessed: number;
  percentAssessed: number;
  byCategory: {
    code: string;
    count: number;
    color: string;
  }[];
  outdated: number;
  percentOutdated: number;
  lastUpdated: string | null;
}

interface Props {
  onSelectTaxon: (taxonId: string | null) => void;
  selectedTaxon: string | null;
}

// Helper to get color styling for percentages
const getAssessedStyle = (percent: number) => ({
  backgroundColor: percent >= 50 ? "rgba(34, 197, 94, 0.15)" : percent >= 20 ? "rgba(234, 179, 8, 0.15)" : "rgba(239, 68, 68, 0.15)",
  color: percent >= 50 ? "#16a34a" : percent >= 20 ? "#ca8a04" : "#dc2626",
});

const getOutdatedStyle = (percent: number) => ({
  backgroundColor: percent < 20 ? "rgba(34, 197, 94, 0.15)" : percent < 40 ? "rgba(234, 179, 8, 0.15)" : "rgba(239, 68, 68, 0.15)",
  color: percent < 20 ? "#16a34a" : percent < 40 ? "#ca8a04" : "#dc2626",
});

export default function TaxaSummary({ onSelectTaxon, selectedTaxon }: Props) {
  const [taxa, setTaxa] = useState<TaxonSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchTaxa() {
      try {
        const res = await fetch("/api/redlist/taxa");
        if (!res.ok) throw new Error("Failed to load taxa");
        const data = await res.json();
        setTaxa(data.taxa);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load taxa");
      } finally {
        setLoading(false);
      }
    }
    fetchTaxa();
  }, []);

  if (loading) {
    return (
      <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl p-6">
        <div className="animate-pulse space-y-3">
          <div className="h-4 bg-zinc-200 dark:bg-zinc-700 rounded w-1/4"></div>
          <div className="h-32 bg-zinc-200 dark:bg-zinc-700 rounded"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 px-4 py-3 rounded-lg">
        {error}
      </div>
    );
  }

  // Calculate totals
  const totalAssessed = taxa.reduce((sum, t) => sum + t.totalAssessed, 0);
  const totalOutdated = taxa.reduce((sum, t) => sum + t.outdated, 0);
  const totalDescribed = taxa.reduce((sum, t) => sum + t.estimatedDescribed, 0);
  const totalPercentAssessed = (totalAssessed / totalDescribed) * 100;
  const totalPercentOutdated = (totalOutdated / totalAssessed) * 100;

  // Check if "all" is selected or a specific taxon
  const isAllSelected = selectedTaxon === "all";
  const hasSpecificTaxon = selectedTaxon && selectedTaxon !== "all";

  // Render a single row (used for both "All Species" and individual taxa)
  const renderRow = (
    id: string,
    name: string,
    color: string,
    estimatedDescribed: number,
    assessed: number,
    percentAssessed: number,
    outdated: number,
    percentOutdated: number,
    isSelected?: boolean
  ) => (
    <div
      key={id}
      onClick={() => onSelectTaxon(isSelected ? null : id)}
      className={`px-4 py-3 cursor-pointer transition-colors ${
        isSelected
          ? "bg-zinc-100 dark:bg-zinc-800"
          : "hover:bg-zinc-50 dark:hover:bg-zinc-800/50"
      }`}
    >
      <div className="grid grid-cols-6 gap-4 items-center">
        {/* Name with icon */}
        <div className="flex items-center gap-2">
          <TaxaIcon taxonId={id} size={22} className="flex-shrink-0" style={{ color }} />
          <span className="font-medium text-zinc-900 dark:text-zinc-100">{name}</span>
        </div>

        {/* Est. Described */}
        <div className="text-right">
          <span className="text-base text-zinc-700 dark:text-zinc-300 tabular-nums">
            {estimatedDescribed.toLocaleString()}
          </span>
        </div>

        {/* Assessed */}
        <div className="text-right">
          <span className="text-base text-zinc-700 dark:text-zinc-300 tabular-nums">
            {assessed.toLocaleString()}
          </span>
        </div>

        {/* % Assessed */}
        <div className="text-right">
          <span
            className="text-base font-medium px-2 py-0.5 rounded tabular-nums"
            style={getAssessedStyle(percentAssessed)}
          >
            {percentAssessed.toFixed(1)}%
          </span>
        </div>

        {/* Outdated */}
        <div className="text-right">
          <span className="text-base text-zinc-700 dark:text-zinc-300 tabular-nums">
            {outdated.toLocaleString()}
          </span>
        </div>

        {/* % Outdated */}
        <div className="text-right">
          <span
            className="text-base font-medium px-2 py-0.5 rounded tabular-nums"
            style={getOutdatedStyle(percentOutdated)}
          >
            {percentOutdated.toFixed(1)}%
          </span>
        </div>
      </div>
    </div>
  );

  // If a specific taxon is selected (not "all"), show just that taxon
  if (hasSpecificTaxon) {
    const taxon = taxa.find(t => t.id === selectedTaxon);
    if (!taxon) return null;

    return (
      <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl">
        {/* Column headers */}
        <div className="px-4 py-2 bg-zinc-50 dark:bg-zinc-800 border-b border-zinc-200 dark:border-zinc-700">
          <div className="grid grid-cols-6 gap-4 text-xs font-medium text-zinc-500 uppercase tracking-wider">
            <div>Taxon</div>
            <div className="text-right flex items-center justify-end gap-1 overflow-visible">
              Est. Described
              <span className="relative group">
                <a
                  href={IUCN_SOURCE_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  onClick={(e) => e.stopPropagation()}
                  className="text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-300"
                >
                  <FaInfoCircle size={12} />
                </a>
                <span className="absolute left-full top-1/2 -translate-y-1/2 ml-2 px-2 py-1 text-xs text-white bg-zinc-800 dark:bg-zinc-700 rounded whitespace-nowrap opacity-0 invisible group-hover:opacity-100 group-hover:visible z-50 shadow-lg normal-case">
                  Source: IUCN Red List Table 1a (2025-2)
                </span>
              </span>
            </div>
            <div className="text-right">Assessed</div>
            <div className="text-right">% Assessed</div>
            <div className="text-right">Outdated (10+y)</div>
            <div className="text-right">% Outdated</div>
          </div>
        </div>

        {renderRow(
          taxon.id,
          taxon.name,
          taxon.color,
          taxon.estimatedDescribed,
          taxon.totalAssessed,
          taxon.percentAssessed,
          taxon.outdated,
          taxon.percentOutdated,
          true
        )}
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl">
      {/* Column headers */}
      <div className="px-4 py-2 bg-zinc-50 dark:bg-zinc-800 border-b border-zinc-200 dark:border-zinc-700">
        <div className="grid grid-cols-6 gap-4 text-xs font-medium text-zinc-500 uppercase tracking-wider">
          <div>Taxon</div>
          <div className="text-right flex items-center justify-end gap-1 overflow-visible">
              Est. Described
              <span className="relative group">
                <a
                  href={IUCN_SOURCE_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  onClick={(e) => e.stopPropagation()}
                  className="text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-300"
                >
                  <FaInfoCircle size={12} />
                </a>
                <span className="absolute left-full top-1/2 -translate-y-1/2 ml-2 px-2 py-1 text-xs text-white bg-zinc-800 dark:bg-zinc-700 rounded whitespace-nowrap opacity-0 invisible group-hover:opacity-100 group-hover:visible z-50 shadow-lg normal-case">
                  Source: IUCN Red List Table 1a (2025-2)
                </span>
              </span>
            </div>
          <div className="text-right">Assessed</div>
          <div className="text-right">% Assessed</div>
          <div className="text-right">Outdated (10+y)</div>
          <div className="text-right">% Outdated</div>
        </div>
      </div>

      {/* All Species row */}
      <div className={!isAllSelected ? "border-b-2 border-zinc-200 dark:border-zinc-700" : ""}>
        {renderRow(
          "all",
          "All Species",
          "#22c55e",
          totalDescribed,
          totalAssessed,
          totalPercentAssessed,
          totalOutdated,
          totalPercentOutdated,
          isAllSelected
        )}
      </div>

      {/* Taxa breakdown - shown when not exploring all */}
      {!isAllSelected && (
        <>
          {/* Individual taxa rows */}
          <div className="divide-y divide-zinc-100 dark:divide-zinc-800">
            {taxa.map((taxon) => (
              <div
                key={taxon.id}
                onClick={() => {
                  if (!taxon.available) return;
                  onSelectTaxon(taxon.id);
                }}
                className={`px-4 py-3 transition-colors ${
                  taxon.available
                    ? "cursor-pointer hover:bg-zinc-50 dark:hover:bg-zinc-800/50"
                    : "opacity-50 cursor-not-allowed"
                }`}
              >
                <div className="grid grid-cols-6 gap-4 items-center">
                  {/* Taxon name with icon */}
                  <div className="flex items-center gap-2">
                    <TaxaIcon
                      taxonId={taxon.id}
                      size={22}
                      className="flex-shrink-0"
                      style={{ color: taxon.color }}
                    />
                    <span className="font-medium text-zinc-900 dark:text-zinc-100">
                      {taxon.name}
                    </span>
                  </div>

                  {/* Est. Described */}
                  <div className="text-right">
                    <span className="text-base text-zinc-700 dark:text-zinc-300 tabular-nums">
                      {taxon.estimatedDescribed.toLocaleString()}
                    </span>
                  </div>

                  {/* Assessed */}
                  <div className="text-right">
                    <span className="text-base text-zinc-700 dark:text-zinc-300 tabular-nums">
                      {taxon.available ? taxon.totalAssessed.toLocaleString() : "—"}
                    </span>
                  </div>

                  {/* % Assessed */}
                  <div className="text-right">
                    {taxon.available ? (
                      <span
                        className="text-base font-medium px-2 py-0.5 rounded tabular-nums"
                        style={getAssessedStyle(taxon.percentAssessed)}
                      >
                        {taxon.percentAssessed.toFixed(1)}%
                      </span>
                    ) : (
                      <span className="text-base text-zinc-400">—</span>
                    )}
                  </div>

                  {/* Outdated */}
                  <div className="text-right">
                    <span className="text-base text-zinc-700 dark:text-zinc-300 tabular-nums">
                      {taxon.available ? taxon.outdated.toLocaleString() : "—"}
                    </span>
                  </div>

                  {/* % Outdated */}
                  <div className="text-right">
                    {taxon.available ? (
                      <span
                        className="text-base font-medium px-2 py-0.5 rounded tabular-nums"
                        style={getOutdatedStyle(taxon.percentOutdated)}
                      >
                        {taxon.percentOutdated.toFixed(1)}%
                      </span>
                    ) : (
                      <span className="text-base text-zinc-400">—</span>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

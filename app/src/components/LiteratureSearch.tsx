"use client";

import { useState, useEffect } from "react";

interface LiteratureResult {
  title: string;
  url: string;
  doi: string | null;
  year: number | null;
  date: string | null;
  citations: number | null;
  source: string;
  sourceType: "academic" | "grey";
  abstract: string | null;
  authors: string | null;
}

// Paper card with collapsible abstract
function PaperCard({ paper }: { paper: LiteratureResult }) {
  const [showAbstract, setShowAbstract] = useState(false);

  return (
    <div className="bg-white dark:bg-zinc-800/50 rounded-lg p-3 border border-zinc-200 dark:border-zinc-700">
      <a
        href={paper.url}
        target="_blank"
        rel="noopener noreferrer"
        className="text-sm text-zinc-900 dark:text-zinc-100 hover:text-blue-600 dark:hover:text-blue-400 line-clamp-2 leading-snug"
      >
        {paper.title}
      </a>
      <div className="flex items-center gap-2 mt-1.5 text-xs text-zinc-500 flex-wrap">
        {paper.year && (
          <span className="font-medium text-zinc-600 dark:text-zinc-400">{paper.year}</span>
        )}
        {paper.source && (
          <span className="text-zinc-400 dark:text-zinc-500">{paper.source}</span>
        )}
        {paper.citations !== null && paper.citations > 0 && (
          <span className="text-amber-600 dark:text-amber-500">
            {paper.citations.toLocaleString()} citations
          </span>
        )}
        {paper.abstract && (
          <button
            onClick={() => setShowAbstract(!showAbstract)}
            className="text-blue-500 hover:text-blue-600 dark:hover:text-blue-400"
          >
            {showAbstract ? "Hide abstract" : "Abstract"}
          </button>
        )}
      </div>
      {paper.authors && (
        <div className="text-xs text-zinc-400 mt-1 truncate">
          {paper.authors}
        </div>
      )}
      {showAbstract && paper.abstract && (
        <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-2 leading-relaxed">
          {paper.abstract}
        </p>
      )}
    </div>
  );
}

interface LiteratureResponse {
  scientificName: string;
  assessmentYear: number;
  totalPapersSinceAssessment: number;
  topPapers: LiteratureResult[];
  openAlexSearchUrl: string;
}

interface NewLiteratureSinceAssessmentProps {
  scientificName: string;
  assessmentYear: number;
  className?: string;
}

// Build the OpenAlex search URL for a species name after a given year
function buildOpenAlexUrl(scientificName: string, sinceYear: number): string {
  // OpenAlex uses this URL format for their web UI
  // Excludes datasets (GBIF occurrence downloads), sorted by most recent
  return `https://openalex.org/works?page=1&filter=default.search%3A%22${encodeURIComponent(scientificName)}%22,publication_year%3A%3E${sinceYear},type%3A%21dataset&sort=publication_date%3Adesc`;
}

export default function NewLiteratureSinceAssessment({
  scientificName,
  assessmentYear,
  className = "",
}: NewLiteratureSinceAssessmentProps) {
  const [data, setData] = useState<LiteratureResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(true); // Auto-expand to show papers

  const openAlexUrl = buildOpenAlexUrl(scientificName, assessmentYear);

  // Human-readable query description
  const queryDescription = `search="${scientificName}" AND year>${assessmentYear} AND type!=dataset`;

  useEffect(() => {
    async function fetchLiterature() {
      setLoading(true);
      setError(null);

      try {
        const params = new URLSearchParams({
          scientificName,
          assessmentYear: assessmentYear.toString(),
          limit: "5",
        });

        const response = await fetch(`/api/literature?${params}`);
        if (!response.ok) {
          throw new Error("Failed to fetch literature");
        }
        const result: LiteratureResponse = await response.json();
        setData(result);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load");
      } finally {
        setLoading(false);
      }
    }

    if (scientificName && assessmentYear) {
      fetchLiterature();
    }
  }, [scientificName, assessmentYear]);

  if (loading) {
    return (
      <div className={`flex items-center gap-2 text-sm text-zinc-400 ${className}`}>
        <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
        </svg>
        Checking OpenAlex...
      </div>
    );
  }

  if (error || !data) {
    return null; // Silently fail - don't clutter UI
  }

  const { totalPapersSinceAssessment, topPapers } = data;

  return (
    <div className={`${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-medium text-zinc-700 dark:text-zinc-300">
            Literature
          </h3>
          <span className="text-sm text-zinc-500">
            {totalPapersSinceAssessment.toLocaleString()} paper{totalPapersSinceAssessment !== 1 ? "s" : ""} since {assessmentYear}
          </span>
          <a
            href={openAlexUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-blue-500 hover:underline"
          >
            View on OpenAlex →
          </a>
        </div>
        <button
          onClick={() => setExpanded(!expanded)}
          className="text-xs text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-300"
        >
          {expanded ? "Collapse" : "Expand"}
        </button>
      </div>

      {/* Query info - subtle */}
      <div className="text-[10px] text-zinc-400 font-mono mb-2">
        OpenAlex: {queryDescription}
      </div>

      {/* Papers list */}
      {expanded && topPapers.length > 0 && (
        <div className="space-y-2">
          {topPapers.map((paper, index) => (
            <PaperCard key={index} paper={paper} />
          ))}

          {totalPapersSinceAssessment > topPapers.length && (
            <div className="text-center pt-1">
              <a
                href={openAlexUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs text-blue-500 hover:underline"
              >
                + {(totalPapersSinceAssessment - topPapers.length).toLocaleString()} more on OpenAlex
              </a>
            </div>
          )}
        </div>
      )}

      {expanded && topPapers.length === 0 && (
        <div className="text-sm text-zinc-500 py-2">
          No papers found.{" "}
          <a href={openAlexUrl} target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:underline">
            Verify on OpenAlex
          </a>
        </div>
      )}

      {/* Subtle note at bottom */}
      {expanded && (
        <p className="text-[10px] text-zinc-400 mt-3">
          Simple text search — may miss synonyms or indirect references
        </p>
      )}
    </div>
  );
}

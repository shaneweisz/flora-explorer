"use client";

import { useState, useEffect } from "react";

interface SpeciesDetails {
  canonicalName?: string;
  vernacularName?: string;
  imageUrl?: string;
  gbifUrl?: string;
}

interface SpeciesInfoBarProps {
  speciesKey: number;
  speciesName: string;
  occurrenceCount?: number;
  region?: string;
}

export default function SpeciesInfoBar({
  speciesKey,
  speciesName,
  occurrenceCount,
  region,
}: SpeciesInfoBarProps) {
  const [details, setDetails] = useState<SpeciesDetails | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    fetch(`/api/species/${speciesKey}`)
      .then((res) => res.json())
      .then((data) => {
        if (!data.error) {
          setDetails(data);
        }
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [speciesKey]);

  const displayName = details?.canonicalName || speciesName;
  const commonName = details?.vernacularName;
  const imageUrl = details?.imageUrl;
  const gbifUrl = details?.gbifUrl || `https://www.gbif.org/species/${speciesKey}`;

  return (
    <div className="bg-white dark:bg-zinc-900 rounded-xl p-4 border border-zinc-200 dark:border-zinc-800">
      <div className="flex items-center gap-4">
        {/* Species image */}
        <div className="w-16 h-16 rounded-lg bg-zinc-100 dark:bg-zinc-800 flex items-center justify-center overflow-hidden flex-shrink-0">
          {imageUrl ? (
            <img
              src={imageUrl}
              alt={displayName}
              className="w-16 h-16 object-cover rounded-lg"
            />
          ) : (
            <svg
              className="w-8 h-8 text-zinc-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
              />
            </svg>
          )}
        </div>

        {/* Species info */}
        <div className="flex-1 min-w-0">
          <div className="flex items-baseline gap-2 flex-wrap">
            <h1 className="text-xl font-bold text-zinc-900 dark:text-zinc-100 italic">
              {displayName}
            </h1>
            {commonName && (
              <span className="text-lg text-zinc-500">({commonName})</span>
            )}
            {loading && (
              <span className="text-sm text-zinc-400">Loading...</span>
            )}
          </div>
          <div className="flex items-center gap-4 mt-1 text-sm text-zinc-500 flex-wrap">
            {occurrenceCount !== undefined && (
              <span>
                {occurrenceCount.toLocaleString()} occurrences
                {region && ` in ${region}`}
              </span>
            )}
            <a
              href={gbifUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="text-green-600 hover:text-green-700 hover:underline"
            >
              View on GBIF →
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}

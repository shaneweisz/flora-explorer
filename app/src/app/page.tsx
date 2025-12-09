"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { ThemeToggle } from "../components/ThemeToggle";

// Dynamically import RedListView component
const RedListView = dynamic(
  () => import("../components/redlist/RedListView"),
  {
    ssr: false,
    loading: () => (
      <div className="flex flex-col items-center justify-center py-20">
        <div className="animate-spin h-10 w-10 border-4 border-red-600 border-t-transparent rounded-full" />
        <p className="mt-4 text-zinc-500 dark:text-zinc-400">
          Loading Red List view...
        </p>
      </div>
    ),
  }
);

export default function RedListPage() {
  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-zinc-950 p-4 md:p-8">
      <main className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-6 flex items-start justify-between">
          <div>
            <h1 className="text-3xl font-bold text-zinc-900 dark:text-zinc-100 mb-2">
              IUCN Red List Dashboard
            </h1>
            <p className="text-zinc-600 dark:text-zinc-400">
              Assessment statistics by taxon
            </p>
          </div>
          <div className="flex items-center gap-2">
            {/* Tab Navigation */}
            <div className="flex gap-1 bg-zinc-100 dark:bg-zinc-800 rounded-lg p-1">
              <div className="px-4 py-1.5 rounded-md text-sm font-medium bg-white dark:bg-zinc-900 text-zinc-900 dark:text-zinc-100 shadow-sm">
                Red List
              </div>
              <Link
                href="/gbif"
                className="px-4 py-1.5 rounded-md text-sm font-medium transition-colors text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200"
              >
                GBIF
              </Link>
            </div>
            <ThemeToggle />
            <a
              href="/experiment"
              className="p-2 text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-300 transition-colors"
              title="Classification Experiment"
            >
              <svg
                className="w-5 h-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z"
                />
              </svg>
            </a>
          </div>
        </div>

        {/* Red List Content */}
        <RedListView />
      </main>
    </div>
  );
}

/**
 * Taxa configuration for IUCN Red List and GBIF data
 *
 * Each taxon has:
 * - id: Unique identifier used in API routes and file names
 * - name: Display name
 * - apiEndpoint: IUCN API endpoint path (kingdom or class)
 * - estimatedDescribed: Estimated number of described species (hardcoded for now)
 * - estimatedSource: Source/citation for the estimate
 * - dataFile: Path to the pre-computed JSON file (Red List)
 * - gbifDataFile: Path to the pre-computed CSV file (GBIF species counts)
 * - gbifKingdomKey: GBIF backbone taxonomy kingdom key
 * - gbifClassKey: GBIF backbone taxonomy class key (for filtering within kingdom)
 * - color: Theme color for UI elements
 */

export interface TaxonConfig {
  id: string;
  name: string;
  apiEndpoint: string; // Primary API endpoint
  apiEndpoints?: string[]; // Multiple API endpoints for combined taxa (e.g., Fishes = Actinopterygii + Chondrichthyes)
  estimatedDescribed: number;
  estimatedSource: string;
  estimatedSourceUrl?: string;
  dataFile: string; // Primary data file
  dataFiles?: string[]; // Multiple data files for combined taxa
  gbifDataFile: string;
  gbifKingdomKey?: number;
  gbifClassKey?: number;
  gbifClassKeys?: number[]; // Multiple class keys for taxa like Reptilia
  gbifOrderKeys?: number[]; // Multiple order keys for taxa like Fish (no class in GBIF)
  color: string;
  icon?: string;
}

// Estimated described species from IUCN Red List Table 1a (version 2025-2)
// Source: https://nc.iucnredlist.org/redlist/content/attachment_files/2025-2_RL_Table1a.pdf
const IUCN_SOURCE = "IUCN 2025-2";
const IUCN_SOURCE_URL = "https://nc.iucnredlist.org/redlist/content/attachment_files/2025-2_RL_Table1a.pdf";

export const TAXA: TaxonConfig[] = [
  {
    id: "aves",
    name: "Birds",
    apiEndpoint: "class/Aves",
    estimatedDescribed: 11185,
    estimatedSource: IUCN_SOURCE,
    estimatedSourceUrl: IUCN_SOURCE_URL,
    dataFile: "redlist-aves.json",
    gbifDataFile: "gbif-aves.csv",
    gbifKingdomKey: 1,
    gbifClassKey: 212,
    color: "#3b82f6", // blue-500
  },
  {
    id: "mammalia",
    name: "Mammals",
    apiEndpoint: "class/Mammalia",
    estimatedDescribed: 6819,
    estimatedSource: IUCN_SOURCE,
    estimatedSourceUrl: IUCN_SOURCE_URL,
    dataFile: "redlist-mammalia.json",
    gbifDataFile: "gbif-mammalia.csv",
    gbifKingdomKey: 1,
    gbifClassKey: 359,
    color: "#f97316", // orange-500
  },
  {
    id: "reptilia",
    name: "Reptiles",
    apiEndpoint: "class/Reptilia",
    estimatedDescribed: 12502,
    estimatedSource: IUCN_SOURCE,
    estimatedSourceUrl: IUCN_SOURCE_URL,
    dataFile: "redlist-reptilia.json",
    gbifDataFile: "gbif-reptilia.csv",
    gbifKingdomKey: 1,
    // GBIF splits Reptilia into: Squamata, Crocodylia, Testudines
    gbifClassKeys: [11592253, 11493978, 11418114],
    color: "#84cc16", // lime-500
  },
  {
    id: "amphibia",
    name: "Amphibians",
    apiEndpoint: "class/Amphibia",
    estimatedDescribed: 8918,
    estimatedSource: IUCN_SOURCE,
    estimatedSourceUrl: IUCN_SOURCE_URL,
    dataFile: "redlist-amphibia.json",
    gbifDataFile: "gbif-amphibia.csv",
    gbifKingdomKey: 1,
    gbifClassKey: 131,
    color: "#14b8a6", // teal-500
  },
  {
    id: "fishes",
    name: "Fishes",
    apiEndpoint: "class/Actinopterygii",
    apiEndpoints: ["class/Actinopterygii", "class/Chondrichthyes"],
    estimatedDescribed: 37288,
    estimatedSource: IUCN_SOURCE,
    estimatedSourceUrl: IUCN_SOURCE_URL,
    dataFile: "redlist-fishes.json",
    dataFiles: ["redlist-actinopterygii.json", "redlist-chondrichthyes.json"],
    gbifDataFile: "gbif-fishes.csv",
    gbifKingdomKey: 1,
    // Combined: ray-finned fish orders + Elasmobranchii (121) + Holocephali (120)
    gbifOrderKeys: [389,391,427,428,446,494,495,496,497,498,499,537,538,547,548,549,550,587,588,589,590,696,708,742,752,753,772,773,774,781,836,848,857,860,861,888,889,890,898,929,975,976,1067,1153,1313],
    gbifClassKeys: [121, 120],
    color: "#06b6d4", // cyan-500
  },
  {
    id: "invertebrates",
    name: "Invertebrates",
    apiEndpoint: "class/Insecta",
    apiEndpoints: ["class/Insecta", "class/Arachnida", "class/Gastropoda", "class/Bivalvia", "class/Malacostraca", "class/Anthozoa"],
    estimatedDescribed: 1508442, // IUCN invertebrates subtotal
    estimatedSource: IUCN_SOURCE,
    estimatedSourceUrl: IUCN_SOURCE_URL,
    dataFile: "redlist-invertebrates.json",
    dataFiles: ["redlist-insecta.json", "redlist-arachnida.json", "redlist-gastropoda.json", "redlist-bivalvia.json", "redlist-malacostraca.json", "redlist-anthozoa.json"],
    gbifDataFile: "gbif-invertebrates.csv",
    gbifKingdomKey: 1,
    gbifClassKeys: [216, 367, 225, 137, 229, 206], // Insecta, Arachnida, Gastropoda, Bivalvia, Malacostraca, Anthozoa
    color: "#78716c", // stone-500
  },
  {
    id: "plantae",
    name: "Plants",
    apiEndpoint: "kingdom/Plantae",
    estimatedDescribed: 426132,
    estimatedSource: IUCN_SOURCE,
    estimatedSourceUrl: IUCN_SOURCE_URL,
    dataFile: "redlist-plantae.json",
    gbifDataFile: "gbif-plantae.csv",
    gbifKingdomKey: 6,
    color: "#22c55e", // green-500
  },
  {
    id: "fungi",
    name: "Fungi",
    apiEndpoint: "phylum/Ascomycota",
    apiEndpoints: ["phylum/Ascomycota", "phylum/Basidiomycota"],
    estimatedDescribed: 162653, // IUCN fungi & protists subtotal
    estimatedSource: IUCN_SOURCE,
    estimatedSourceUrl: IUCN_SOURCE_URL,
    dataFile: "redlist-fungi.json",
    dataFiles: ["redlist-ascomycota.json", "redlist-basidiomycota.json"],
    gbifDataFile: "gbif-fungi.csv",
    gbifKingdomKey: 5,
    color: "#d97706", // amber-600
  },
];

// Map for quick lookup by ID
export const TAXA_BY_ID: Record<string, TaxonConfig> = Object.fromEntries(
  TAXA.map((t) => [t.id, t])
);

// Get taxon by ID, with fallback to plantae
export function getTaxonConfig(id: string): TaxonConfig {
  return TAXA_BY_ID[id] || TAXA_BY_ID["plantae"];
}

// IUCN category colors (shared across all taxa)
export const CATEGORY_COLORS: Record<string, string> = {
  EX: "#000000",
  EW: "#542344",
  CR: "#d81e05",
  EN: "#fc7f3f",
  VU: "#f9e814",
  NT: "#cce226",
  LC: "#60c659",
  DD: "#6b7280",
};

// Category order for sorting (most threatened first)
export const CATEGORY_ORDER: Record<string, number> = {
  EX: 0,
  EW: 1,
  CR: 2,
  EN: 3,
  VU: 4,
  NT: 5,
  LC: 6,
  DD: 7,
};

// Category full names
export const CATEGORY_NAMES: Record<string, string> = {
  EX: "Extinct",
  EW: "Extinct in the Wild",
  CR: "Critically Endangered",
  EN: "Endangered",
  VU: "Vulnerable",
  NT: "Near Threatened",
  LC: "Least Concern",
  DD: "Data Deficient",
};

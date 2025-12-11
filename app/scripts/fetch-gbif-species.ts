/**
 * GBIF Species Occurrence Count Fetcher
 * =====================================
 *
 * Pre-computes species occurrence counts from GBIF for any taxon and saves to CSV.
 * Output includes scientific names for Red List matching (no separate enrichment step needed).
 *
 * Output format: species_key,occurrence_count,scientific_name
 *
 * ## Problem This Solves
 *
 * The raw GBIF occurrence facet API returns ALL speciesKeys with occurrences, but many
 * of these are not valid species. They include:
 * - Subspecies, varieties, forms (rank !== SPECIES)
 * - Synonyms, doubtful names, misapplied names (taxonomicStatus !== ACCEPTED)
 * - Fossil specimens and literature citations (basisOfRecord filtering)
 *
 * Without filtering, you get inflated counts (e.g., 15,000 "mammals" vs ~6,800 actual species).
 *
 * ## Solution: Two-Stage Filtering
 *
 * 1. **Basis of Record Filter** (during occurrence fetch):
 *    - Excludes FOSSIL_SPECIMEN and MATERIAL_CITATION
 *    - These often represent extinct species or unverified literature records
 *    - Applied via GBIF API query parameters
 *
 * 2. **Species Validation** (post-fetch, via GBIF Species API):
 *    - Fetches each speciesKey from https://api.gbif.org/v1/species/{key}
 *    - Keeps only records where: rank=SPECIES AND taxonomicStatus=ACCEPTED
 *    - Filters out genera, families, subspecies, synonyms, etc.
 *    - Also retrieves canonical names (used for Red List matching)
 *
 * ## Performance Notes
 *
 * - SPECIES_VALIDATION_BATCH_SIZE: Controls concurrent API requests (default: 500)
 * - SPECIES_VALIDATION_DELAY: Delay between batches to avoid rate limiting (default: 50ms)
 * - Large taxa (plants: 350k, invertebrates: 560k) take 5-10 minutes with current settings
 * - Uses loop-based push instead of spread operator to avoid stack overflow on large arrays
 *
 * ## Expected Results (validated species counts vs IUCN estimates)
 *
 * | Taxon         | GBIF Validated | IUCN Estimate | Coverage |
 * |---------------|----------------|---------------|----------|
 * | Mammals       | ~6,100         | 6,819         | ~90%     |
 * | Birds         | ~11,400        | 11,185        | ~102%    |
 * | Reptiles      | ~10,600        | 12,502        | ~85%     |
 * | Amphibians    | ~7,200         | 8,918         | ~81%     |
 * | Fishes        | ~33,500        | 37,288        | ~90%     |
 * | Plants        | ~335,000       | 426,132       | ~79%     |
 * | Fungi         | ~94,000        | 162,653       | ~58%     |
 * | Invertebrates | ~560,000       | 1,508,442     | ~37%     |
 *
 * Usage:
 *   npx tsx scripts/fetch-gbif-species.ts <taxon>
 *
 * Taxa (from src/config/taxa.ts):
 *   plantae, fungi, mammalia, aves, reptilia, amphibia, actinopterygii,
 *   chondrichthyes, insecta, arachnida, malacostraca, gastropoda,
 *   bivalvia, anthozoa, fishes, invertebrates
 *
 * Examples:
 *   npx tsx scripts/fetch-gbif-species.ts mammalia
 *   npx tsx scripts/fetch-gbif-species.ts plantae
 */

import * as fs from "fs";
import * as path from "path";

// Taxa configuration (inline to avoid import issues with tsx)
interface TaxonConfig {
  id: string;
  name: string;
  gbifDataFile: string;
  gbifKingdomKey: number;
  gbifClassKey?: number;
  gbifClassKeys?: number[]; // Multiple class keys (e.g., reptiles split into Squamata, Crocodylia, Testudines)
  gbifOrderKeys?: number[]; // Multiple order keys (e.g., fish have no class in GBIF)
}

const FISH_ORDER_KEYS = [389,391,427,428,446,494,495,496,497,498,499,537,538,547,548,549,550,587,588,589,590,696,708,742,752,753,772,773,774,781,836,848,857,860,861,888,889,890,898,929,975,976,1067,1153,1313];

const TAXA_CONFIG: Record<string, TaxonConfig> = {
  plantae: { id: "plantae", name: "Plants", gbifDataFile: "gbif-plantae.csv", gbifKingdomKey: 6 },
  fungi: { id: "fungi", name: "Fungi", gbifDataFile: "gbif-fungi.csv", gbifKingdomKey: 5 },
  mammalia: { id: "mammalia", name: "Mammals", gbifDataFile: "gbif-mammalia.csv", gbifKingdomKey: 1, gbifClassKey: 359 },
  aves: { id: "aves", name: "Birds", gbifDataFile: "gbif-aves.csv", gbifKingdomKey: 1, gbifClassKey: 212 },
  reptilia: { id: "reptilia", name: "Reptiles", gbifDataFile: "gbif-reptilia.csv", gbifKingdomKey: 1, gbifClassKeys: [11592253, 11493978, 11418114] },
  amphibia: { id: "amphibia", name: "Amphibians", gbifDataFile: "gbif-amphibia.csv", gbifKingdomKey: 1, gbifClassKey: 131 },
  // Combined taxa
  fishes: { id: "fishes", name: "Fishes", gbifDataFile: "gbif-fishes.csv", gbifKingdomKey: 1, gbifOrderKeys: FISH_ORDER_KEYS, gbifClassKeys: [121, 120] },
  mollusca: { id: "mollusca", name: "Molluscs", gbifDataFile: "gbif-mollusca.csv", gbifKingdomKey: 1, gbifClassKeys: [225, 137] },
  // Individual taxa (still available for separate fetching)
  actinopterygii: { id: "actinopterygii", name: "Ray-finned Fishes", gbifDataFile: "gbif-actinopterygii.csv", gbifKingdomKey: 1, gbifOrderKeys: FISH_ORDER_KEYS },
  chondrichthyes: { id: "chondrichthyes", name: "Sharks & Rays", gbifDataFile: "gbif-chondrichthyes.csv", gbifKingdomKey: 1, gbifClassKeys: [121, 120] },
  gastropoda: { id: "gastropoda", name: "Snails & Slugs", gbifDataFile: "gbif-gastropoda.csv", gbifKingdomKey: 1, gbifClassKey: 225 },
  bivalvia: { id: "bivalvia", name: "Bivalves", gbifDataFile: "gbif-bivalvia.csv", gbifKingdomKey: 1, gbifClassKey: 137 },
  // Other taxa
  arachnida: { id: "arachnida", name: "Arachnids", gbifDataFile: "gbif-arachnida.csv", gbifKingdomKey: 1, gbifClassKey: 367 },
  malacostraca: { id: "malacostraca", name: "Crustaceans", gbifDataFile: "gbif-malacostraca.csv", gbifKingdomKey: 1, gbifClassKey: 229 },
  anthozoa: { id: "anthozoa", name: "Corals & Anemones", gbifDataFile: "gbif-anthozoa.csv", gbifKingdomKey: 1, gbifClassKey: 206 },
  insecta: { id: "insecta", name: "Insects", gbifDataFile: "gbif-insecta.csv", gbifKingdomKey: 1, gbifClassKey: 216 },
  // Combined invertebrates
  invertebrates: { id: "invertebrates", name: "Invertebrates", gbifDataFile: "gbif-invertebrates.csv", gbifKingdomKey: 1, gbifClassKeys: [216, 367, 225, 137, 229, 206] },
};

// =============================================================================
// CONFIGURATION - Adjust these values to tune performance vs. API rate limits
// =============================================================================

// Max species keys returned per GBIF occurrence facet request (GBIF hard limit)
const FACET_LIMIT = 100000;

// Delay between occurrence API requests (ms) - lower = faster, but may hit rate limits
const REQUEST_DELAY = 200;

// Number of concurrent species validation requests per batch
// Higher = faster, but may hit rate limits or cause memory issues
// At 500 concurrent requests, can validate ~10k species/second
const SPECIES_VALIDATION_BATCH_SIZE = 500;

// Delay between validation batches (ms) - gives GBIF API breathing room
const SPECIES_VALIDATION_DELAY = 50;

// =============================================================================
// BASIS OF RECORD FILTER
// =============================================================================
// GBIF basisOfRecord types to INCLUDE in occurrence counts.
// We explicitly EXCLUDE:
//   - FOSSIL_SPECIMEN: Extinct species, not relevant for current biodiversity
//   - MATERIAL_CITATION: Literature references, often duplicate or unverified data
// Including these would inflate counts significantly.
const INCLUDED_BASIS_OF_RECORD = [
  "HUMAN_OBSERVATION",    // iNaturalist, eBird, etc.
  "MACHINE_OBSERVATION",  // Camera traps, acoustic sensors
  "PRESERVED_SPECIMEN",   // Museum specimens (current, not fossil)
  "OCCURRENCE",           // Generic occurrence record
  "MATERIAL_SAMPLE",      // DNA/tissue samples
  "OBSERVATION",          // Generic observation
  "LIVING_SPECIMEN",      // Zoo/garden specimens
];

interface SpeciesCount {
  speciesKey: number;
  count: number;
}

interface FacetResponse {
  count: number;
  facets: Array<{
    field: string;
    counts: Array<{
      name: string;
      count: number;
    }>;
  }>;
}

async function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

interface SpeciesInfo {
  key: number;
  rank: string;
  taxonomicStatus: string;
  scientificName: string;
  canonicalName: string;
}

interface ValidatedSpecies {
  key: number;
  canonicalName: string;
}

/**
 * Validate species keys against the GBIF Species API
 *
 * WHY THIS IS NECESSARY:
 * The GBIF Occurrence API's speciesKey facet returns ANY taxon with occurrences,
 * not just valid species. This includes:
 *
 * 1. Non-species ranks (genera, families, subspecies, varieties)
 *    Example: A record identified only to genus gets a speciesKey for that genus
 *
 * 2. Non-accepted taxonomic statuses (synonyms, doubtful, misapplied)
 *    Example: Old scientific names that are now synonyms still have speciesKeys
 *
 * This function queries https://api.gbif.org/v1/species/{key} for each key and
 * filters to keep only: rank=SPECIES AND taxonomicStatus=ACCEPTED
 *
 * TYPICAL FILTERING RATES:
 * - Mammals: ~1% filtered (mostly subspecies)
 * - Plants: ~4% filtered (synonyms + infraspecific taxa)
 * - Invertebrates: ~0.5% filtered (mostly data quality issues)
 *
 * @returns Map of valid species keys to their canonical names
 */
async function validateSpeciesKeys(speciesKeys: number[]): Promise<Map<number, string>> {
  const validSpecies = new Map<number, string>();
  const invalidKeys: Array<{ key: number; reason: string }> = [];

  console.log(`\nValidating ${speciesKeys.length} species keys...`);

  for (let i = 0; i < speciesKeys.length; i += SPECIES_VALIDATION_BATCH_SIZE) {
    const batch = speciesKeys.slice(i, i + SPECIES_VALIDATION_BATCH_SIZE);

    const results = await Promise.all(
      batch.map(async (key) => {
        try {
          const response = await fetch(`https://api.gbif.org/v1/species/${key}`);
          if (!response.ok) {
            return { key, rank: "UNKNOWN", taxonomicStatus: "UNKNOWN", scientificName: "Unknown", canonicalName: "" };
          }
          const data = await response.json();
          return {
            key,
            rank: data.rank || "UNKNOWN",
            taxonomicStatus: data.taxonomicStatus || "UNKNOWN",
            scientificName: data.scientificName || "Unknown",
            canonicalName: data.canonicalName || data.scientificName || "",
          };
        } catch {
          return { key, rank: "ERROR", taxonomicStatus: "ERROR", scientificName: "Error", canonicalName: "" };
        }
      })
    );

    for (const info of results) {
      if (info.rank === "SPECIES" && info.taxonomicStatus === "ACCEPTED") {
        validSpecies.set(info.key, info.canonicalName);
      } else {
        invalidKeys.push({
          key: info.key,
          reason: `rank=${info.rank}, status=${info.taxonomicStatus}`,
        });
      }
    }

    const progress = Math.min(i + SPECIES_VALIDATION_BATCH_SIZE, speciesKeys.length);
    process.stdout.write(`\r  Validated ${progress}/${speciesKeys.length} (${validSpecies.size} valid, ${invalidKeys.length} filtered)`);

    if (i + SPECIES_VALIDATION_BATCH_SIZE < speciesKeys.length) {
      await delay(SPECIES_VALIDATION_DELAY);
    }
  }

  console.log(`\n  Filtered out ${invalidKeys.length} non-species or non-accepted taxa`);

  // Log some examples of filtered taxa
  if (invalidKeys.length > 0) {
    console.log(`  Examples of filtered taxa:`);
    invalidKeys.slice(0, 5).forEach((item) => {
      console.log(`    - Key ${item.key}: ${item.reason}`);
    });
  }

  return validSpecies;
}

function buildGbifUrl(taxon: TaxonConfig, minCount?: number, maxCount?: number): string {
  const params = new URLSearchParams({
    hasCoordinate: "true",
    hasGeospatialIssue: "false",
    facet: "speciesKey",
    facetLimit: FACET_LIMIT.toString(),
    limit: "0",
  });

  // Add taxonomy filter
  if (taxon.gbifClassKey) {
    params.set("classKey", taxon.gbifClassKey.toString());
  } else {
    params.set("kingdomKey", taxon.gbifKingdomKey.toString());
  }

  // Add occurrence count range if specified
  if (minCount !== undefined || maxCount !== undefined) {
    // GBIF doesn't directly support filtering facets by count
    // We'll need to use a different approach - download facets and filter locally
  }

  return `https://api.gbif.org/v1/occurrence/search?${params}`;
}

async function fetchSpeciesCounts(taxon: TaxonConfig): Promise<SpeciesCount[]> {
  const url = buildGbifUrl(taxon);
  console.log(`Fetching from: ${url}`);

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`GBIF API error: ${response.statusText}`);
  }

  const data: FacetResponse = await response.json();
  const speciesFacet = data.facets.find((f) => f.field === "SPECIES_KEY");

  if (!speciesFacet) {
    return [];
  }

  const results: SpeciesCount[] = speciesFacet.counts.map((c) => ({
    speciesKey: parseInt(c.name, 10),
    count: c.count,
  }));

  // Sort by count descending
  results.sort((a, b) => b.count - a.count);

  return results;
}

function deduplicateAndSort(allResults: SpeciesCount[]): SpeciesCount[] {
  const seen = new Map<number, SpeciesCount>();
  for (const r of allResults) {
    if (!seen.has(r.speciesKey) || seen.get(r.speciesKey)!.count < r.count) {
      seen.set(r.speciesKey, r);
    }
  }
  const deduped = Array.from(seen.values());
  deduped.sort((a, b) => b.count - a.count);
  return deduped;
}

async function fetchForTaxonKey(keyType: "classKey" | "orderKey", keyValue: number, label: string): Promise<SpeciesCount[]> {
  const allResults: SpeciesCount[] = [];
  let offset = 0;
  let hasMore = true;

  while (hasMore) {
    const params = new URLSearchParams({
      hasCoordinate: "true",
      hasGeospatialIssue: "false",
      facet: "speciesKey",
      facetLimit: FACET_LIMIT.toString(),
      facetOffset: offset.toString(),
      limit: "0",
      [keyType]: keyValue.toString(),
    });

    // Add basisOfRecord filter to exclude fossils and literature citations
    INCLUDED_BASIS_OF_RECORD.forEach((bor) => params.append("basisOfRecord", bor));

    const url = `https://api.gbif.org/v1/occurrence/search?${params}`;
    console.log(`Fetching ${label} offset ${offset}...`);

    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`GBIF API error: ${response.statusText}`);
    }

    const data: FacetResponse = await response.json();
    const speciesFacet = data.facets.find((f) => f.field === "SPECIES_KEY");

    if (!speciesFacet || speciesFacet.counts.length === 0) {
      hasMore = false;
      break;
    }

    const results = speciesFacet.counts.map((c) => ({
      speciesKey: parseInt(c.name, 10),
      count: c.count,
    }));

    // IMPORTANT: Use loop-based push instead of allResults.push(...results)
    // The spread operator causes "RangeError: Maximum call stack size exceeded"
    // when arrays exceed ~100k elements (insecta has 400k+, plants have 350k+)
    for (const r of results) {
      allResults.push(r);
    }
    console.log(`  -> Got ${results.length} species (total: ${allResults.length})`);

    if (results.length < FACET_LIMIT) {
      hasMore = false;
    } else {
      offset += FACET_LIMIT;
      await delay(REQUEST_DELAY);
    }
  }

  return allResults;
}

async function fetchForClassKey(classKey: number, label: string): Promise<SpeciesCount[]> {
  const allResults: SpeciesCount[] = [];
  let offset = 0;
  let hasMore = true;

  while (hasMore) {
    const params = new URLSearchParams({
      hasCoordinate: "true",
      hasGeospatialIssue: "false",
      facet: "speciesKey",
      facetLimit: FACET_LIMIT.toString(),
      facetOffset: offset.toString(),
      limit: "0",
      classKey: classKey.toString(),
    });

    const url = `https://api.gbif.org/v1/occurrence/search?${params}`;
    console.log(`Fetching ${label} offset ${offset}...`);

    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`GBIF API error: ${response.statusText}`);
    }

    const data: FacetResponse = await response.json();
    const speciesFacet = data.facets.find((f) => f.field === "SPECIES_KEY");

    if (!speciesFacet || speciesFacet.counts.length === 0) {
      hasMore = false;
      break;
    }

    const results = speciesFacet.counts.map((c) => ({
      speciesKey: parseInt(c.name, 10),
      count: c.count,
    }));

    // Use loop-based push to avoid stack overflow with very large arrays
    for (const r of results) {
      allResults.push(r);
    }
    console.log(`  -> Got ${results.length} species (total: ${allResults.length})`);

    if (results.length < FACET_LIMIT) {
      hasMore = false;
    } else {
      offset += FACET_LIMIT;
      await delay(REQUEST_DELAY);
    }
  }

  return allResults;
}

async function fetchAllSpeciesCounts(taxon: TaxonConfig): Promise<SpeciesCount[]> {
  const allResults: SpeciesCount[] = [];

  // Handle multiple order keys (e.g., ray-finned fish have no class in GBIF)
  if (taxon.gbifOrderKeys && taxon.gbifOrderKeys.length > 0) {
    let orderIndex = 0;
    for (const orderKey of taxon.gbifOrderKeys) {
      orderIndex++;
      console.log(`\nFetching orderKey ${orderKey} (${orderIndex}/${taxon.gbifOrderKeys.length})...`);
      const results = await fetchForTaxonKey("orderKey", orderKey, `order ${orderKey}`);
      // Use loop-based push to avoid stack overflow with large arrays
      for (const r of results) {
        allResults.push(r);
      }
      await delay(REQUEST_DELAY);
    }
  }

  // Handle multiple class keys (e.g., reptiles, sharks/rays, molluscs)
  if (taxon.gbifClassKeys && taxon.gbifClassKeys.length > 0) {
    for (const classKey of taxon.gbifClassKeys) {
      console.log(`\nFetching classKey ${classKey}...`);
      const results = await fetchForTaxonKey("classKey", classKey, `class ${classKey}`);
      // Use loop-based push to avoid stack overflow with large arrays
      for (const r of results) {
        allResults.push(r);
      }
      await delay(REQUEST_DELAY);
    }
  }

  // If we fetched from order keys or class keys, deduplicate and return
  if (allResults.length > 0) {
    return deduplicateAndSort(allResults);
  }

  // Single class key or kingdom key
  let offset = 0;
  let hasMore = true;

  while (hasMore) {
    const params = new URLSearchParams({
      hasCoordinate: "true",
      hasGeospatialIssue: "false",
      facet: "speciesKey",
      facetLimit: FACET_LIMIT.toString(),
      facetOffset: offset.toString(),
      limit: "0",
    });

    // Add basisOfRecord filter to exclude fossils and literature citations
    INCLUDED_BASIS_OF_RECORD.forEach((bor) => params.append("basisOfRecord", bor));

    if (taxon.gbifClassKey) {
      params.set("classKey", taxon.gbifClassKey.toString());
    } else {
      params.set("kingdomKey", taxon.gbifKingdomKey.toString());
    }

    const url = `https://api.gbif.org/v1/occurrence/search?${params}`;
    console.log(`Fetching offset ${offset}...`);

    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`GBIF API error: ${response.statusText}`);
    }

    const data: FacetResponse = await response.json();
    const speciesFacet = data.facets.find((f) => f.field === "SPECIES_KEY");

    if (!speciesFacet || speciesFacet.counts.length === 0) {
      hasMore = false;
      break;
    }

    const results = speciesFacet.counts.map((c) => ({
      speciesKey: parseInt(c.name, 10),
      count: c.count,
    }));

    // Use loop-based push to avoid stack overflow with very large arrays
    for (const r of results) {
      allResults.push(r);
    }
    console.log(`  -> Got ${results.length} species (total: ${allResults.length})`);

    if (results.length < FACET_LIMIT) {
      hasMore = false;
    } else {
      offset += FACET_LIMIT;
      await delay(REQUEST_DELAY);
    }
  }

  allResults.sort((a, b) => b.count - a.count);
  return allResults;
}

interface SpeciesCountWithName extends SpeciesCount {
  scientificName: string;
}

function saveToCsv(results: SpeciesCountWithName[], outputFile: string): void {
  const header = "species_key,occurrence_count,scientific_name";
  const rows = results.map((r) => {
    // Escape scientific name if it contains commas (shouldn't happen, but safe)
    const safeName = r.scientificName.includes(",") ? `"${r.scientificName}"` : r.scientificName;
    return `${r.speciesKey},${r.count},${safeName}`;
  });
  const content = [header, ...rows].join("\n");
  fs.writeFileSync(outputFile, content);
}

async function main() {
  // Parse CLI arguments
  const args = process.argv.slice(2);
  const taxonId = args[0]?.toLowerCase();

  // Validate taxon
  if (!taxonId) {
    console.error("Usage: npx tsx scripts/fetch-gbif-species.ts <taxon>");
    console.error("\nAvailable taxa:");
    Object.entries(TAXA_CONFIG).forEach(([id, config]) => {
      console.error(`  ${id.padEnd(18)} - ${config.name}`);
    });
    process.exit(1);
  }

  const taxonConfig = TAXA_CONFIG[taxonId];
  if (!taxonConfig) {
    console.error(`Unknown taxon: ${taxonId}`);
    console.error("\nAvailable taxa:");
    Object.keys(TAXA_CONFIG).forEach((id) => console.error(`  ${id}`));
    process.exit(1);
  }

  const OUTPUT_FILE = path.join(process.cwd(), "data", taxonConfig.gbifDataFile);

  console.log(`GBIF Species Count Fetcher - ${taxonConfig.name}`);
  console.log("=".repeat(50));
  console.log(`Taxon: ${taxonConfig.name} (${taxonId})`);
  console.log(`Kingdom Key: ${taxonConfig.gbifKingdomKey}`);
  if (taxonConfig.gbifClassKey) {
    console.log(`Class Key: ${taxonConfig.gbifClassKey}`);
  }
  console.log(`Output: ${OUTPUT_FILE}`);
  console.log("");

  try {
    console.log("Fetching species occurrence counts from GBIF...");
    console.log("(Excluding FOSSIL_SPECIMEN and MATERIAL_CITATION records)");
    const rawResults = await fetchAllSpeciesCounts(taxonConfig);

    console.log(`\nRaw species count (before validation): ${rawResults.length}`);

    // Validate species keys to filter out non-species and non-accepted taxa
    // Also retrieves canonical names for Red List matching
    const speciesKeys = rawResults.map((r) => r.speciesKey);
    const validSpecies = await validateSpeciesKeys(speciesKeys);

    // Filter results to only include validated species, and add scientific names
    const results: SpeciesCountWithName[] = rawResults
      .filter((r) => validSpecies.has(r.speciesKey))
      .map((r) => ({
        ...r,
        scientificName: validSpecies.get(r.speciesKey) || "",
      }));

    console.log(`\nFinal species count (after validation): ${results.length}`);

    if (results.length > 0) {
      console.log(`Top 5 by occurrence count:`);
      results.slice(0, 5).forEach((r, i) => {
        console.log(`  ${i + 1}. Species ${r.speciesKey}: ${r.count.toLocaleString()} occurrences`);
      });

      const totalOccurrences = results.reduce((sum, r) => sum + r.count, 0);
      console.log(`\nTotal occurrences: ${totalOccurrences.toLocaleString()}`);
    }

    console.log(`\nSaving to ${OUTPUT_FILE}...`);
    saveToCsv(results, OUTPUT_FILE);

    const stats = fs.statSync(OUTPUT_FILE);
    const sizeMB = (stats.size / (1024 * 1024)).toFixed(2);
    console.log(`Done! File size: ${sizeMB} MB`);

  } catch (error) {
    console.error("Error:", error);
    process.exit(1);
  }
}

main().catch(console.error);

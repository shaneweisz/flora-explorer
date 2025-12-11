/**
 * GBIF Species Name Enricher
 * ==========================
 *
 * Enriches existing GBIF CSV files with scientific names by fetching from GBIF Species API.
 * This enables runtime matching between GBIF species and Red List assessments.
 *
 * Input format:  species_key,occurrence_count
 * Output format: species_key,occurrence_count,scientific_name
 *
 * Usage:
 *   npx tsx scripts/enrich-gbif-names.ts <taxon>
 *   npx tsx scripts/enrich-gbif-names.ts all
 *
 * Examples:
 *   npx tsx scripts/enrich-gbif-names.ts plantae
 *   npx tsx scripts/enrich-gbif-names.ts mammalia
 *   npx tsx scripts/enrich-gbif-names.ts all
 */

import * as fs from "fs";
import * as path from "path";

// Configuration
const BATCH_SIZE = 500; // Concurrent API requests per batch
const BATCH_DELAY = 50; // ms delay between batches

// Taxa configuration (matches fetch-gbif-species.ts)
const TAXA_FILES: Record<string, string> = {
  plantae: "gbif-plantae.csv",
  fungi: "gbif-fungi.csv",
  mammalia: "gbif-mammalia.csv",
  aves: "gbif-aves.csv",
  reptilia: "gbif-reptilia.csv",
  amphibia: "gbif-amphibia.csv",
  fishes: "gbif-fishes.csv",
  mollusca: "gbif-mollusca.csv",
  insecta: "gbif-insecta.csv",
  arachnida: "gbif-arachnida.csv",
  malacostraca: "gbif-malacostraca.csv",
  anthozoa: "gbif-anthozoa.csv",
};

interface SpeciesRecord {
  species_key: number;
  occurrence_count: number;
  scientific_name?: string;
}

async function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function fetchSpeciesName(speciesKey: number): Promise<string | null> {
  try {
    const response = await fetch(`https://api.gbif.org/v1/species/${speciesKey}`);
    if (!response.ok) return null;
    const data = await response.json();
    // Use canonicalName (without authorship) for cleaner matching
    return data.canonicalName || data.scientificName || null;
  } catch {
    return null;
  }
}

async function enrichFile(taxonId: string, fileName: string): Promise<void> {
  const filePath = path.join(process.cwd(), "data", fileName);

  // Check if file exists
  if (!fs.existsSync(filePath)) {
    console.log(`  Skipping ${taxonId}: file not found (${fileName})`);
    return;
  }

  // Read existing CSV
  const content = fs.readFileSync(filePath, "utf-8");
  const lines = content.trim().split("\n");
  const header = lines[0];

  // Check if already enriched
  if (header.includes("scientific_name")) {
    console.log(`  Skipping ${taxonId}: already enriched`);
    return;
  }

  // Parse data
  const records: SpeciesRecord[] = lines.slice(1).map((line) => {
    const [species_key, occurrence_count] = line.split(",");
    return {
      species_key: parseInt(species_key, 10),
      occurrence_count: parseInt(occurrence_count, 10),
    };
  });

  console.log(`  Enriching ${taxonId}: ${records.length} species...`);

  // Fetch names in batches
  for (let i = 0; i < records.length; i += BATCH_SIZE) {
    const batch = records.slice(i, i + BATCH_SIZE);

    const names = await Promise.all(
      batch.map((r) => fetchSpeciesName(r.species_key))
    );

    batch.forEach((record, idx) => {
      record.scientific_name = names[idx] || "";
    });

    const progress = Math.min(i + BATCH_SIZE, records.length);
    process.stdout.write(`\r    Progress: ${progress}/${records.length} (${((progress / records.length) * 100).toFixed(1)}%)`);

    if (i + BATCH_SIZE < records.length) {
      await delay(BATCH_DELAY);
    }
  }

  console.log(""); // New line after progress

  // Write enriched CSV
  const newHeader = "species_key,occurrence_count,scientific_name";
  const newRows = records.map((r) => {
    // Escape scientific name if it contains commas (shouldn't happen, but safe)
    const name = r.scientific_name || "";
    const safeName = name.includes(",") ? `"${name}"` : name;
    return `${r.species_key},${r.occurrence_count},${safeName}`;
  });

  const newContent = [newHeader, ...newRows].join("\n");
  fs.writeFileSync(filePath, newContent);

  const stats = fs.statSync(filePath);
  const sizeMB = (stats.size / (1024 * 1024)).toFixed(2);
  console.log(`    Saved: ${sizeMB} MB`);
}

async function main() {
  const args = process.argv.slice(2);
  const taxonId = args[0]?.toLowerCase();

  if (!taxonId) {
    console.error("Usage: npx tsx scripts/enrich-gbif-names.ts <taxon>");
    console.error("       npx tsx scripts/enrich-gbif-names.ts all");
    console.error("\nAvailable taxa:");
    Object.keys(TAXA_FILES).forEach((id) => console.error(`  ${id}`));
    process.exit(1);
  }

  console.log("GBIF Species Name Enricher");
  console.log("=".repeat(50));

  if (taxonId === "all") {
    // Process all taxa
    for (const [id, fileName] of Object.entries(TAXA_FILES)) {
      await enrichFile(id, fileName);
    }
  } else {
    // Process single taxon
    const fileName = TAXA_FILES[taxonId];
    if (!fileName) {
      console.error(`Unknown taxon: ${taxonId}`);
      console.error("\nAvailable taxa:");
      Object.keys(TAXA_FILES).forEach((id) => console.error(`  ${id}`));
      process.exit(1);
    }
    await enrichFile(taxonId, fileName);
  }

  console.log("\nDone!");
}

main().catch(console.error);

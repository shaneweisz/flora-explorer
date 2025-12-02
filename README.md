# Data-Deficient Plant Search

Find candidate locations for plant species using habitat similarity from geospatial embeddings.

## Why This Matters

GBIF has occurrence data for 354,357 plant species, but:
- **72.6%** have 100 or fewer occurrences
- **36.6%** have 10 or fewer occurrences
- **9.3%** have just 1 occurrence

This tool helps find where to look for rare plants by computing habitat similarity from known locations.

## How It Works

1. Fetch GBIF occurrences for a species in a region
2. Sample Tessera embeddings at those locations
3. Compute centroid of occurrence embeddings
4. Score every pixel by cosine similarity to centroid
5. Output high-similarity locations as candidates

Works with any number of samples, including just 1.

## Usage

```bash
uv run python run.py "Quercus robur" --region cambridge
uv run python run.py "Species name" --bbox 0.0,52.0,1.0,53.0
```

## Requirements

Pre-downloaded Tessera embeddings in `cache/2024/` (0.1° tiles).

## Output

Results in `output/{species}/`:
- `probability.tif` - Similarity heatmap
- `candidates.geojson` - High-similarity locations
- `occurrences.geojson` - GBIF records used

## Web App

```bash
cd app && npm install && npm run dev
```

Species with predictions show an "AI" badge at http://localhost:3000.

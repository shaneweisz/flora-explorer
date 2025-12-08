# Data-Deficient Plant Search

Find candidate locations for plant species using habitat classification from geospatial embeddings.

## Why This Matters

GBIF has occurrence data for 354,357 plant species, but:
- **72.6%** have 100 or fewer occurrences
- **36.6%** have 10 or fewer occurrences
- **9.3%** have just 1 occurrence

This tool helps find where to look for rare plants by learning habitat preferences from known locations.

## How It Works

1. Fetch GBIF occurrences for a species in a region
2. Sample Tessera embeddings at occurrence locations (positive samples)
3. Sample random background embeddings (negative samples)
4. Train a logistic regression classifier (positive vs background)
5. Score every pixel by classifier probability
6. Output high-probability locations as candidates

Validated to achieve ~87% AUC with 100 training samples (see `/experiment` page).

## Usage

```bash
uv run python run.py "Quercus robur" --region cambridge
uv run python run.py "Species name" --bbox 0.0,52.0,1.0,53.0
```

## Requirements

- Pre-downloaded Tessera embeddings in `cache/2024/` (0.1° tiles)
- At least 2 occurrences for the species in the region

## Output

Results in `output/{species}/`:
- `probability.tif` - Classifier probability heatmap
- `candidates.geojson` - High-probability locations
- `occurrences.geojson` - GBIF records used

## Web App

```bash
cd app && npm install && npm run dev
```

- Main explorer: http://localhost:3000
- Experiment validation: http://localhost:3000/experiment

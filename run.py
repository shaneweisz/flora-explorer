#!/usr/bin/env python3
"""
Species Candidate Location Finder

Find candidate locations for plant species using habitat similarity
computed from Tessera geospatial embeddings.

Usage:
    uv run python run.py "Quercus robur" --region cambridge
    uv run python run.py "Species name" --bbox 0.0,52.0,1.0,53.0
"""

import argparse
import logging
from pathlib import Path

from finder import find_candidates
from finder.pipeline import REGIONS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "output"
CACHE_DIR = PROJECT_ROOT / "cache"


def main():
    parser = argparse.ArgumentParser(
        description="Find candidate locations for a species using habitat similarity"
    )
    parser.add_argument("species", help="Scientific name of the species")
    parser.add_argument("--region", choices=list(REGIONS.keys()), help="Predefined region")
    parser.add_argument("--bbox", help="Bounding box: min_lon,min_lat,max_lon,max_lat")
    parser.add_argument("-o", "--output", help="Output directory")

    args = parser.parse_args()

    if args.region:
        bbox = REGIONS[args.region]["bbox"]
    elif args.bbox:
        bbox = tuple(map(float, args.bbox.split(",")))
    else:
        parser.error("Specify --region or --bbox")

    slug = args.species.lower().replace(" ", "_")
    output_dir = Path(args.output) if args.output else OUTPUT_DIR / slug

    result = find_candidates(
        species_name=args.species,
        bbox=bbox,
        cache_dir=CACHE_DIR,
        output_dir=output_dir,
    )

    print(f"\nOutput: {output_dir}/")
    print(f"  - probability.tif")
    print(f"  - candidates.geojson ({result.to_geojson()['metadata']['n_candidates']} points)")
    print(f"  - occurrences.geojson ({result.n_occurrences} GBIF records)")


if __name__ == "__main__":
    main()

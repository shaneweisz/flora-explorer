"""
GBIF API interactions for fetching species data and occurrences.
"""

import requests
from typing import Optional


def get_species_key(species_name: str) -> int:
    """Look up GBIF taxon key for a species name."""
    resp = requests.get(
        "https://api.gbif.org/v1/species/match",
        params={"name": species_name}
    )
    resp.raise_for_status()
    data = resp.json()
    key = data.get("usageKey")
    if not key:
        raise ValueError(f"Species not found: {species_name}")
    return key


def get_species_info(species_name: str) -> dict:
    """Get species information including taxon key and matched name."""
    resp = requests.get(
        "https://api.gbif.org/v1/species/match",
        params={"name": species_name}
    )
    resp.raise_for_status()
    data = resp.json()
    if not data.get("usageKey"):
        raise ValueError(f"Species not found: {species_name}")
    return {
        "taxon_key": data["usageKey"],
        "scientific_name": data.get("scientificName", species_name),
        "canonical_name": data.get("canonicalName", species_name),
        "rank": data.get("rank"),
        "confidence": data.get("confidence", 0),
    }


def fetch_occurrences(
    taxon_key: int,
    bbox: tuple[float, float, float, float],
    limit: Optional[int] = None
) -> list[tuple[float, float]]:
    """
    Fetch occurrence coordinates from GBIF.

    Args:
        taxon_key: GBIF taxon key
        bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat)
        limit: Maximum number of occurrences to fetch (None = all)

    Returns:
        List of (longitude, latitude) tuples
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    results = []
    offset = 0
    batch_size = 300

    while True:
        resp = requests.get(
            "https://api.gbif.org/v1/occurrence/search",
            params={
                "taxonKey": taxon_key,
                "hasCoordinate": "true",
                "hasGeospatialIssue": "false",
                "decimalLatitude": f"{min_lat},{max_lat}",
                "decimalLongitude": f"{min_lon},{max_lon}",
                "limit": batch_size,
                "offset": offset,
            }
        )
        resp.raise_for_status()
        data = resp.json()
        batch = data.get("results", [])

        if not batch:
            break

        for r in batch:
            if r.get("decimalLatitude") and r.get("decimalLongitude"):
                results.append((r["decimalLongitude"], r["decimalLatitude"]))

        if limit and len(results) >= limit:
            results = results[:limit]
            break

        if len(results) >= data.get("count", 0):
            break

        offset += batch_size

    return results

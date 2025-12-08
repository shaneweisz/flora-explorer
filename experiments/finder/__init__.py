"""
Species Candidate Location Finder

Find candidate locations for plant species using geospatial embeddings.
"""

from .gbif import get_species_key, get_species_info, fetch_occurrences
from .embeddings import EmbeddingMosaic
from .methods import ClassifierMethod
from .pipeline import find_candidates

__all__ = [
    "get_species_key",
    "get_species_info",
    "fetch_occurrences",
    "EmbeddingMosaic",
    "ClassifierMethod",
    "find_candidates",
]

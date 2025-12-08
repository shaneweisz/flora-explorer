"""
Tessera embedding mosaic loading and sampling.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.transform import Affine


class EmbeddingMosaic:
    """
    Manages loading and querying of Tessera embedding tiles.

    Tiles are stored as quantized numpy arrays with separate scale files.
    This class stitches them into a mosaic and provides coordinate-based access.
    """

    def __init__(
        self,
        cache_dir: Path,
        bbox: tuple[float, float, float, float],
        year: int = 2024,
        tile_size: float = 0.1,
    ):
        """
        Initialize the mosaic for a given bounding box.

        Args:
            cache_dir: Directory containing year subdirectories with tiles
            bbox: (min_lon, min_lat, max_lon, max_lat)
            year: Year of embeddings to load
            tile_size: Size of each tile in degrees (default 0.1°)
        """
        self.cache_dir = Path(cache_dir)
        self.bbox = bbox
        self.year = year
        self.tile_size = tile_size

        self._mosaic: Optional[np.ndarray] = None
        self._transform: Optional[Affine] = None
        self._tile_coords: list[tuple[float, float]] = []

    def load(self) -> None:
        """Load and stitch tiles covering the bounding box."""
        min_lon, min_lat, max_lon, max_lat = self.bbox
        tile_dir = self.cache_dir / str(self.year)

        # Calculate tile grid covering bbox
        step = self.tile_size
        # Tiles are named by their center, offset by half step
        half_step = step / 2

        tile_lons = np.arange(
            np.floor((min_lon + half_step) / step) * step - half_step,
            max_lon + step,
            step
        )
        tile_lats = np.arange(
            np.floor((min_lat + half_step) / step) * step - half_step,
            max_lat + step,
            step
        )

        # Load available tiles
        tiles: dict[tuple[float, float], np.ndarray] = {}
        for tlon in tile_lons:
            for tlat in tile_lats:
                tlon_r, tlat_r = round(tlon, 2), round(tlat, 2)
                name = f"grid_{tlon_r:.2f}_{tlat_r:.2f}"
                npy_path = tile_dir / name / f"{name}.npy"
                scales_path = tile_dir / name / f"{name}_scales.npy"

                if npy_path.exists() and scales_path.exists():
                    data = np.load(npy_path).astype(np.float32)
                    scales = np.load(scales_path)
                    # Dequantize: multiply by scales
                    tiles[(tlon_r, tlat_r)] = data * scales[:, :, np.newaxis]

        if not tiles:
            raise ValueError(f"No tiles found in {tile_dir} for bbox {self.bbox}")

        self._tile_coords = list(tiles.keys())

        # Get dimensions from first tile
        sample_tile = next(iter(tiles.values()))
        tile_h, tile_w, n_channels = sample_tile.shape

        # Sort coordinates for stitching
        unique_lons = sorted(set(t[0] for t in tiles.keys()))
        unique_lats = sorted(set(t[1] for t in tiles.keys()), reverse=True)

        # Create mosaic array
        mosaic_h = len(unique_lats) * tile_h
        mosaic_w = len(unique_lons) * tile_w
        self._mosaic = np.zeros((mosaic_h, mosaic_w, n_channels), dtype=np.float32)

        # Stitch tiles
        for i, tlat in enumerate(unique_lats):
            for j, tlon in enumerate(unique_lons):
                if (tlon, tlat) in tiles:
                    tile = tiles[(tlon, tlat)]
                    h, w = tile.shape[:2]
                    self._mosaic[i*tile_h:i*tile_h+h, j*tile_w:j*tile_w+w, :] = tile

        # Create geotransform
        mosaic_min_lon = min(unique_lons)
        mosaic_max_lat = max(unique_lats) + step
        self._transform = rasterio.transform.from_bounds(
            mosaic_min_lon,
            mosaic_max_lat - step * len(unique_lats),
            mosaic_min_lon + step * len(unique_lons),
            mosaic_max_lat,
            mosaic_w,
            mosaic_h
        )

    @property
    def mosaic(self) -> np.ndarray:
        """Get the loaded mosaic array (H, W, C)."""
        if self._mosaic is None:
            self.load()
        return self._mosaic

    @property
    def transform(self) -> Affine:
        """Get the geotransform for the mosaic."""
        if self._transform is None:
            self.load()
        return self._transform

    @property
    def shape(self) -> tuple[int, int, int]:
        """Get mosaic shape (height, width, channels)."""
        return self.mosaic.shape

    @property
    def n_pixels(self) -> int:
        """Total number of pixels in the mosaic."""
        h, w, _ = self.shape
        return h * w

    def sample_at_coords(
        self,
        coords: list[tuple[float, float]]
    ) -> tuple[np.ndarray, list[tuple[float, float]]]:
        """
        Sample embeddings at given coordinates.

        Args:
            coords: List of (longitude, latitude) tuples

        Returns:
            Tuple of (embeddings array, valid coordinates list)
        """
        mosaic = self.mosaic
        h, w = mosaic.shape[:2]

        embeddings = []
        valid_coords = []

        for lon, lat in coords:
            row, col = rasterio.transform.rowcol(self.transform, lon, lat)
            if 0 <= row < h and 0 <= col < w:
                embeddings.append(mosaic[row, col, :])
                valid_coords.append((lon, lat))

        return np.array(embeddings) if embeddings else np.array([]), valid_coords

    def get_all_embeddings(self) -> np.ndarray:
        """Get all embeddings as a flat array (N, C)."""
        mosaic = self.mosaic
        return mosaic.reshape(-1, mosaic.shape[-1])

    def pixel_to_coords(self, row: int, col: int) -> tuple[float, float]:
        """Convert pixel coordinates to geographic coordinates."""
        lon, lat = rasterio.transform.xy(self.transform, row, col)
        return lon, lat

    def coords_to_pixel(self, lon: float, lat: float) -> tuple[int, int]:
        """Convert geographic coordinates to pixel coordinates."""
        row, col = rasterio.transform.rowcol(self.transform, lon, lat)
        return row, col

"""Tile-boundary regression checks for the publication buffer grid."""
import unittest
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from modules.rice_overlay import merge_buffer


class RiceBufferTests(unittest.TestCase):
    def test_adjacent_tiles_cover_rounded_buffer_without_false_nodata_row(self):
        # Noninteger geographic bounds straddle the two sources. The rounded
        # mosaic edge extends past the requested bound, just as in the study.
        with MemoryFile() as upper, MemoryFile() as lower:
            common = dict(driver='GTiff', count=1, dtype='uint8', crs='EPSG:4326', width=10, height=10)
            with upper.open(**common, transform=from_origin(0, 2, .1, .1)) as u, \
                 lower.open(**common, transform=from_origin(0, 1, .1, .1)) as l:
                u.write(np.ones((10, 10), dtype=np.uint8), 1)
                l.write(np.zeros((10, 10), dtype=np.uint8), 1)
                mosaic, transform = merge_buffer([u, l], (.02, .43, .94, 1.42))
            self.assertEqual(mosaic.shape, (10, 9))
            self.assertFalse(np.any(mosaic == 255))
            self.assertTrue(np.isin(mosaic, [0, 1]).all())
            self.assertAlmostEqual(transform.f, 1.42)

    def test_unmapped_area_stays_missing_instead_of_non_rice(self):
        with MemoryFile() as memory:
            with memory.open(driver='GTiff', count=1, dtype='uint8', crs='EPSG:4326',
                             width=10, height=10, transform=from_origin(0, 1, .1, .1)) as tile:
                tile.write(np.ones((10, 10), dtype=np.uint8), 1)
                mosaic, _ = merge_buffer([tile], (0, 0, 1.2, 1))
            self.assertTrue(np.all(mosaic[:, -2:] == 255))


if __name__ == '__main__':
    unittest.main()

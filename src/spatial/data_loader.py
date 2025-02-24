import geopandas as gpd
from libpysal.weights import Queen
from typing import Tuple, List, Set
import os


def load_data_and_build_adjacency(shapefile_path: str) -> Tuple[gpd.GeoDataFrame, List[Set[int]]]:
    """
    Load shapefile and build adjacency using Queen contiguity.
    Attempt to link any island polygon to its nearest neighbor.
    """
    if not os.path.exists(shapefile_path):
        raise FileNotFoundError(f"Shapefile not found: {shapefile_path}")

    try:
        gdf = gpd.read_file(str(shapefile_path))
    except Exception as e:
        print(f"Error reading shapefile: {e}")
        raise

    try:
        w = Queen.from_dataframe(gdf, use_index=True)
        n = len(gdf)

        # Build adjacency list-of-sets
        adj = [set(w.neighbors[i]) for i in range(n)]

        # Identify islands and connect them
        islands = [i for i in range(n) if not adj[i]]
        if islands:
            print(f"Connecting {len(islands)} island(s) to nearest neighbors...")
            centroids = gdf.geometry.centroid
            for island in islands:
                distances = centroids.distance(centroids.iloc[island])
                closest = distances.nsmallest(2).index[1]  # skip self
                adj[island].add(closest)
                adj[closest].add(island)

        return gdf, adj

    except Exception as e:
        print(f"Error building adjacency: {e}")
        raise


def get_region_counts(gdf: gpd.GeoDataFrame) -> Tuple[int, int]:
    """
    Get the number of regions and total areas from the GeoDataFrame.
    """
    num_regions = gdf['COUNTYFP'].nunique()
    total_areas = len(gdf)
    return num_regions, total_areas
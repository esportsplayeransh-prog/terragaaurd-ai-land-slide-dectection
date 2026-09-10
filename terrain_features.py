import rasterio
import numpy as np

from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS


DEM_FILE = "darjeeling_dem.tif"

# Darjeeling is in UTM Zone 45N
TARGET_CRS = CRS.from_epsg(32645)


# ============================================================
# 1. REPROJECT DEM TO METRIC CRS
# ============================================================

with rasterio.open(DEM_FILE) as src:

    transform, width, height = calculate_default_transform(
        src.crs,
        TARGET_CRS,
        src.width,
        src.height,
        *src.bounds
    )

    elevation = np.empty(
        (height, width),
        dtype=np.float32
    )

    reproject(
        source=rasterio.band(src, 1),
        destination=elevation,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=transform,
        dst_crs=TARGET_CRS,
        resampling=Resampling.bilinear
    )

    # Pixel size is now in metres
    xres = abs(transform.a)
    yres = abs(transform.e)


# Remove invalid values
elevation[elevation < -100] = np.nan


print("==============================")
print("DARJEELING TERRAIN ANALYSIS")
print("==============================")

print(f"DEM resolution: {xres:.2f} m × {yres:.2f} m")


# ============================================================
# 2. ELEVATION
# ============================================================

center_row = elevation.shape[0] // 2
center_col = elevation.shape[1] // 2

center_elevation = elevation[
    center_row,
    center_col
]

print(f"Elevation: {center_elevation:.2f} m")


# ============================================================
# 3. SLOPE
# ============================================================

dz_dy, dz_dx = np.gradient(
    elevation,
    yres,
    xres
)

slope_rad = np.arctan(
    np.sqrt(
        dz_dx ** 2 +
        dz_dy ** 2
    )
)

slope_deg = np.degrees(slope_rad)

center_slope = slope_deg[
    center_row,
    center_col
]

print(f"Slope: {center_slope:.2f}°")


# ============================================================
# 4. ASPECT
# ============================================================

aspect_rad = np.arctan2(
    -dz_dx,
    dz_dy
)

aspect_deg = np.degrees(aspect_rad)

aspect_deg = (
    aspect_deg + 360
) % 360

center_aspect = aspect_deg[
    center_row,
    center_col
]

print(f"Aspect: {center_aspect:.2f}°")


# ============================================================
# 5. CURVATURE
# ============================================================

d2z_dx2 = np.gradient(
    dz_dx,
    xres,
    axis=1
)

d2z_dy2 = np.gradient(
    dz_dy,
    yres,
    axis=0
)

curvature = (
    d2z_dx2 +
    d2z_dy2
)

center_curvature = curvature[
    center_row,
    center_col
]

print(f"Curvature: {center_curvature:.6f}")


print("\n✅ Terrain feature extraction complete.")
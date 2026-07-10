"""
Panoramic strip mosaic generation from individual georeferenced GeoTIFFs.

Alignment is purely coordinate-based using each GeoTIFF's embedded CRS and
affine transform (as produced by geotiff_processor.py).  No feature matching
or image warping is performed — the georeferencing is assumed to be accurate
enough that the merge is seamless.
"""

import os
import glob
import traceback
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import rasterio
import rasterio.io
from rasterio.merge import merge


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _calculate_merge_dimensions(
    datasets: List[rasterio.DatasetReader],
    resolution: float,
) -> dict:
    """Calculate expected output dimensions for a merge operation.
    
    Returns dict with 'width', 'height', 'width_m', 'height_m'.
    """
    from rasterio.merge import merge
    
    # Get bounds of all datasets in their common CRS
    min_x, min_y, max_x, max_y = datasets[0].bounds
    
    # Track individual bounds for diagnostics
    all_bounds = []
    for ds in datasets:
        bounds = ds.bounds
        all_bounds.append({
            'file': os.path.basename(ds.name),
            'bounds': bounds,
            'width': bounds.right - bounds.left,
            'height': bounds.top - bounds.bottom,
        })
        min_x = min(min_x, bounds.left)
        min_y = min(min_y, bounds.bottom)
        max_x = max(max_x, bounds.right)
        max_y = max(max_y, bounds.top)
    
    # Calculate dimensions at the target resolution
    width_units = max_x - min_x
    height_units = max_y - min_y
    
    width_pixels = int(np.ceil(width_units / resolution))
    height_pixels = int(np.ceil(height_units / resolution))
    
    return {
        'width': width_pixels,
        'height': height_pixels,
        'width_m': width_units,
        'height_m': height_units,
        'bounds': (min_x, min_y, max_x, max_y),
        'individual_bounds': all_bounds,
    }





# ---------------------------------------------------------------------------
# Strip generation directly from original images (no pre-existing GeoTIFFs)
# ---------------------------------------------------------------------------

def create_panoramic_strips_from_images(
    image_list_csv: str,
    image_dir: str,
    output_dir: str,
    images_per_strip: int = 50,
    resolution_m: float = 0.005,
    strip_prefix: str = "strip",
    selected_images: Optional[List[str]] = None,
    lever_arm_x: float = 0.1044,
    lever_arm_y: float = 0.6246,
    lever_arm_z: float = 0.0826,
    pitch_offset: float = 0.010,
    roll_offset: float = 0.010,
    heading_offset: float = 0.000,
    utm_zone: Optional[int] = None,
    utm_hemisphere: Optional[str] = None,
    crop_top_pixels: int = 0,
    nodata_value: int = 0,
    merge_method: str = "first",
    # Minimum percent overlap (0-100). 100 disables overlap-based skipping.
    min_overlap_pct: float = 100.0,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Build panoramic strip mosaics directly from original images without
    writing intermediate GeoTIFFs to disk.

    Each image is orthorectified in-memory using the same homography
    pipeline as :func:`geotiff_processor.process_image_to_geotiff` and
    then contributed to the mosaic via ``rasterio.merge``.

    Parameters
    ----------
    image_list_csv : str
        Path to the (normalized) CSV with navigation data.
    image_dir : str
        Directory containing the source images.
    output_dir : str
        Directory where strip GeoTIFFs will be saved.
    images_per_strip : int
        Number of images per strip mosaic.
    resolution_m : float
        Output pixel size in metres.
    strip_prefix : str
        Filename prefix for output strips.
    selected_images : Optional[List[str]]
        Subset of image filenames to process (``None`` = all).
    lever_arm_x/y/z : float
        IMU-to-camera lever-arm offsets in the vehicle body frame.
    pitch_offset, roll_offset, heading_offset : float
        Angular calibration offsets in degrees.
    utm_zone : Optional[int]
        UTM zone (``None`` = auto from image lat/lon).
    utm_hemisphere : Optional[str]
        ``'N'`` or ``'S'`` (``None`` = auto from image lat/lon).
    crop_top_pixels : int
        Rows to remove from the top of each image before orthorectification.
        Improves along-track overlap quality.  0 = disabled.
    min_overlap_pct : float
        Minimum percent overlap between a new image and the previous included
        image required to skip the new image. 100 = disabled (no overlap filter).
    nodata_value : int
        No-data value written into the output GeoTIFFs (default 0).
    merge_method : str
        Rasterio merge strategy (``'first'`` or ``'last'``).
    progress_callback : Optional[Callable[[str], None]]
        Optional logging callback.

    Returns
    -------
    dict
        ``{'total_strips': int, 'success': int, 'failed': int,
           'total_images': int, 'skipped_images': int}``
    """
    # Import here to avoid circular imports at module load time
    from geotiff_processor import process_image_to_geotiff_memfile

    def log(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

    # ------------------------------------------------------------------
    # Load & filter image list
    # ------------------------------------------------------------------
    df = pd.read_csv(image_list_csv)
    if selected_images is not None:
        df = df[df['file_name'].isin(selected_images)]
    df = df.reset_index(drop=True)

    if df.empty:
        log("  No images to process.")
        return {"total_strips": 0, "success": 0, "failed": 0,
                "total_images": 0, "skipped_images": 0}

    log(f"  Processing {len(df)} image(s) directly from {image_dir}")
    if crop_top_pixels > 0:
        log(f"  Cropping top {crop_top_pixels} pixel row(s) from each image.")

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Partition rows into strips
    # ------------------------------------------------------------------
    total_images = len(df)
    strip_rows: List[pd.DataFrame] = [
        df.iloc[i : i + images_per_strip]
        for i in range(0, total_images, images_per_strip)
    ]
    total_strips = len(strip_rows)
    log(f"  Partitioned into {total_strips} strip(s) of up to {images_per_strip} image(s) each.")
    log(f"  Output resolution: {resolution_m * 100:.1f} cm/pixel  ({resolution_m} m/pixel)")

    stats = {
        "total_strips": total_strips,
        "success": 0,
        "failed": 0,
        "total_images": total_images,
        "skipped_images": 0,
    }

    for strip_idx, strip_df in enumerate(strip_rows, 1):
        strip_name = f"{strip_prefix}_{strip_idx:03d}.tif"
        strip_path = os.path.join(output_dir, strip_name)

        log(f"\n  Strip {strip_idx}/{total_strips}: "
            f"orthorectifying {len(strip_df)} image(s) → {strip_name}")

        memfiles: List[rasterio.io.MemoryFile] = []
        open_datasets: List[rasterio.DatasetReader] = []

        try:
            for row_idx, (_, img_row) in enumerate(strip_df.iterrows(), 1):
                fname = img_row['file_name']
                log(f"    [{row_idx}/{len(strip_df)}] {fname}")

                mf = process_image_to_geotiff_memfile(
                    image_row=img_row,
                    image_dir=image_dir,
                    lever_arm_x=lever_arm_x,
                    lever_arm_y=lever_arm_y,
                    lever_arm_z=lever_arm_z,
                    pitch_offset=pitch_offset,
                    roll_offset=roll_offset,
                    heading_offset=heading_offset,
                    utm_zone=utm_zone,
                    utm_hemisphere=utm_hemisphere,
                    crop_top_pixels=crop_top_pixels,
                    nodata_val=nodata_value,
                    progress_callback=progress_callback,
                )

                if mf is None:
                    log(f"    Skipped {fname} (orthorectification failed).")
                    stats["skipped_images"] += 1
                    continue

                # Open the in-memory dataset to inspect its bounds and
                # possibly skip it if it overlaps too much with the
                # previously-included image for this strip.
                ds_new = mf.open()

                try:
                    skip_due_to_overlap = False
                    # If we already have an included dataset, compute
                    # intersection area between the new image and the
                    # last included image. Skip if >=70% overlap of the
                    # new image's area.
                    if open_datasets:
                        ds_prev = open_datasets[-1]
                        b1 = ds_prev.bounds
                        b2 = ds_new.bounds

                        inter_w = max(0.0, min(b1.right, b2.right) - max(b1.left, b2.left))
                        inter_h = max(0.0, min(b1.top, b2.top) - max(b1.bottom, b2.bottom))
                        inter_area = inter_w * inter_h

                        area_new = (b2.right - b2.left) * (b2.top - b2.bottom)
                        overlap_ratio = inter_area / area_new if area_new > 0 else 0.0

                        # If user requested overlap filtering (min_overlap_pct < 100),
                        # compare against the user-specified threshold. Otherwise skip
                        # overlap-based filtering.
                        try:
                            threshold = float(min_overlap_pct) / 100.0
                        except Exception:
                            threshold = 1.0

                        if threshold < 1.0 and overlap_ratio >= threshold:
                            pct = overlap_ratio * 100.0
                            log(f"    Skipped {fname} — {pct:.1f}% overlap with previous image.")
                            stats["skipped_images"] += 1
                            skip_due_to_overlap = True

                    if skip_due_to_overlap:
                        # Close the opened dataset and memoryfile since we're not using it.
                        try:
                            ds_new.close()
                        except Exception:
                            pass
                        try:
                            mf.close()
                        except Exception:
                            pass
                        continue

                    # Otherwise keep the memoryfile/dataset for merging.
                    memfiles.append(mf)
                    open_datasets.append(ds_new)
                except Exception:
                    # Ensure we close resources on any unexpected error here.
                    try:
                        ds_new.close()
                    except Exception:
                        pass
                    try:
                        mf.close()
                    except Exception:
                        pass
                    raise

            if not open_datasets:
                raise RuntimeError("No images were successfully orthorectified for this strip.")

            log(f"    Merging {len(open_datasets)} orthorectified dataset(s)…")

            # Safety check: prevent absurdly large outputs
            expected = _calculate_merge_dimensions(open_datasets, resolution_m)
            total_px = expected['width'] * expected['height']
            max_px = 350_000_000
            if total_px > max_px:
                raise RuntimeError(
                    f"Merged output would be {total_px:,} pixels "
                    f"({total_px / 1e6:.1f} MP), exceeding safety limit of "
                    f"{max_px / 1e6:.1f} MP. Check georeferencing."
                )

            log(f"    Expected output: {expected['width']:,} x {expected['height']:,} pixels "
                f"({expected['width_m']:.1f} x {expected['height_m']:.1f} m)")

            mosaic, out_transform = merge(
                open_datasets,
                method=merge_method,
                nodata=nodata_value,
                res=(resolution_m, resolution_m),
            )

            out_meta = open_datasets[0].meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_transform,
                "compress": "lzw",
                "tiled": True,
                "blockxsize": 512,
                "blockysize": 512,
                "nodata": nodata_value,
                "photometric": "RGB",
            })

            with rasterio.open(strip_path, "w", **out_meta) as dst:
                dst.write(mosaic)
                dst.update_tags(
                    SOFTWARE="Python/Rasterio/OpenCV",
                    STRIP_IMAGE_COUNT=str(len(open_datasets)),
                    RESOLUTION_M=str(resolution_m),
                    MERGE_METHOD=merge_method,
                    CROP_TOP_PIXELS=str(crop_top_pixels),
                )

            log(f"    Saved: {strip_path}")
            stats["success"] += 1

        except Exception as exc:
            log(f"    ERROR on strip {strip_idx}: {exc}")
            log(traceback.format_exc())
            stats["failed"] += 1

        finally:
            for ds in open_datasets:
                try:
                    ds.close()
                except Exception:
                    pass
            for mf in memfiles:
                try:
                    mf.close()
                except Exception:
                    pass

    log(
        f"\n  Panoramic strip summary (from images): "
        f"{stats['success']} succeeded, {stats['failed']} failed, "
        f"{stats['skipped_images']} image(s) skipped out of {stats['total_images']} total."
    )
    return stats

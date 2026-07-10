"""
Unified Pipeline for NeRF / 3D Gaussian Splatting Data Preparation.
Colorizes a LAZ point cloud, identifies used images, and outputs 
COLMAP (sparse/0/) and Nerfstudio (transforms.json) workspaces.
"""

import os
import struct
import json
import shutil
import numpy as np
import pandas as pd
import cv2
from typing import Tuple, List



# ============================================================================
# CONFIGURATION
# ============================================================================
# Camera intrinsic parameters (from your existing calibration)
FOCAL_LENGTH_PX = 3801.37053
IMAGE_WIDTH = 4096
IMAGE_HEIGHT = 3008

K1, K2, P1, P2 = 0.0113579, -0.0143928, 0.0042688, -0.000244194

# Lever arm offsets (meters)
LEVER_ARM_X = 0.125813   # Forward
LEVER_ARM_Y = 0.945584   # Right
LEVER_ARM_Z = -0.213513  # Up

# Angular offsets (degrees)
PITCH_OFFSET = 0.010
ROLL_OFFSET = 0.010
HEADING_OFFSET = 0.000


# ============================================================================
# MATH & SPATIAL UTILITIES
# ============================================================================

def pixels_to_world_coordinates(
    distance_off_bottom: float,
    pitch: float,
    roll: float, 
    heading: float,
    image_path: str,
    camera_east: float = 0.0,
    camera_north: float = 0.0,
    focal_length_px: float = 3801.37053,
    k1: float = 0.0113579,
    k2: float = -0.0143928,
    p1: float = 0.0042688,
    p2: float = -0.000244194,
    image_width: int = 4096,
    image_height: int = 3008,
    downsample: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert image pixels to world coordinates in meters assuming a flat 2D plane.
    
    Parameters:
    -----------
    distance_off_bottom : float
        Distance from camera to seafloor in meters
    pitch : float
        Camera pitch angle in degrees (rotation about X-axis: positive = nose up)
    roll : float  
        Camera roll angle in degrees (rotation about Y-axis: positive = right bank)
    heading : float
        Camera heading angle in degrees (rotation about Z-axis: 0 = North, 90 = East)
    image_path : str
        Path to the image file (for validation/reference)
    camera_east, camera_north : float
        Camera position in ENU coordinates (meters)
    focal_length_px : float
        Focal length in pixels (default: 3801.37053)
    k1, k2, p1, p2 : float
        Radial and tangential distortion coefficients
    image_width, image_height : int
        Image dimensions in pixels
    downsample : int
        Downsampling factor (1=no downsampling, 2=half size, 10=1/10 size, etc.)
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        x_world, y_world arrays in meters for each pixel in absolute ENU coordinates
    """
    
    # make correction to pitch for downward looking camera
    pitch = -90 + pitch
    
    # Apply downsampling to image dimensions and camera parameters
    image_width_ds = image_width // downsample
    image_height_ds = image_height // downsample
    focal_length_ds = focal_length_px / downsample
    
    # Convert angles from degrees to radians
    pitch_rad = np.radians(pitch)
    roll_rad = np.radians(roll) 
    heading_rad = np.radians(heading)
    
    # Camera intrinsic matrix (using downsampled parameters)
    cx = image_width_ds / 2.0   # Principal point x
    cy = image_height_ds / 2.0  # Principal point y
    fx = fy = focal_length_ds  # Focal length in pixels
    
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy], 
        [0, 0, 1]
    ])
    
    # Distortion coefficients
    dist_coeffs = np.array([k1, k2, p1, p2, 0])
    
    # Create pixel coordinate grids (using downsampled dimensions)
    u_coords, v_coords = np.meshgrid(
        np.arange(image_width_ds, dtype=np.float32),
        np.arange(image_height_ds, dtype=np.float32)
    )
    
    # Stack coordinates for undistortion
    pixel_coords = np.stack([u_coords.ravel(), v_coords.ravel()], axis=1)
    pixel_coords = pixel_coords.reshape(-1, 1, 2)
    
    # Undistort pixel coordinates
    undistorted_coords = cv2.undistortPoints(
        pixel_coords, camera_matrix, dist_coeffs, P=camera_matrix
    )
    undistorted_coords = undistorted_coords.reshape(-1, 2)
    
    # Convert to normalized camera coordinates
    u_undist = undistorted_coords[:, 0]
    v_undist = undistorted_coords[:, 1]
    
    # Convert to normalized coordinates (subtract principal point, divide by focal length)
    x_norm = (u_undist - cx) / fx
    y_norm = (v_undist - cy) / fy
    
    # Create rays in camera coordinate system
    # Camera coordinates: X=right, Y=forward, Z=up
    # For a ray pointing through pixel (x_norm, y_norm), the ray direction in camera frame is:
    # X = x_norm (rightward displacement from optical axis)
    # Y = 1 (forward along optical axis - this should be the primary direction)  
    # Z = -y_norm (upward displacement from optical axis, negative because image Y is downward)
    rays_camera = np.column_stack([x_norm, np.ones(len(x_norm)), -y_norm])
    
    # Create rotation matrices for camera orientation
    # All rotations are in camera reference frame: X=right, Y=forward, Z=up
    
    # Pitch rotation (around X-axis in camera frame - nose up/down)
    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad), np.cos(pitch_rad)]
    ])
    
    # Roll rotation (around Y-axis in camera frame - left/right bank)
    R_roll = np.array([
        [np.cos(roll_rad), 0, np.sin(roll_rad)],
        [0, 1, 0],
        [-np.sin(roll_rad), 0, np.cos(roll_rad)]
    ])
    
    # Heading rotation (around Z-axis in camera frame - left/right turn)
    # Standard navigation convention: 0° = North, 90° = East
    # This rotates the camera's forward direction to align with the heading
    R_heading = np.array([
        [np.cos(heading_rad), np.sin(heading_rad), 0],
        [-np.sin(heading_rad), np.cos(heading_rad), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation: pitch, then roll, then heading (all in camera frame)
    # Order matters: we apply rotations in sequence
    R_camera_to_world = R_heading @ R_roll @ R_pitch
    
    # Transform rays to world coordinate system
    rays_world = (R_camera_to_world @ rays_camera.T).T
    
    # Extract ray components in world coordinates (ENU)
    ray_x = rays_world[:, 0]  # East component
    ray_y = rays_world[:, 1]  # North component  
    ray_z = rays_world[:, 2]  # Up component
    
    # Calculate intersection with seafloor plane (Z = 0)
    # Camera is at height +distance_off_bottom above the seafloor (seafloor at Z=0)
    # Ray equation: P = camera_position + t * ray_direction
    # Camera position: (camera_east, camera_north, distance_off_bottom)  
    # Seafloor plane: Z = 0
    # Solve: distance_off_bottom + t * ray_z = 0
    # Therefore: t = -distance_off_bottom / ray_z
    
    # Avoid division by zero for rays parallel to seafloor
    valid_rays = np.abs(ray_z) > 1e-10
    t = np.full(len(ray_z), np.nan)
    t[valid_rays] = -distance_off_bottom / ray_z[valid_rays]
    
    # Only keep rays that intersect the seafloor (positive t for downward-pointing rays)
    # For a camera above seafloor, we want rays with negative Z component (pointing down)
    # and positive t values (intersection in front of camera)
    valid_intersection = valid_rays & (t > 0) & (ray_z < 0)
    t[~valid_intersection] = np.nan
    
    # Calculate world coordinates relative to camera position
    x_relative = t * ray_x
    y_relative = t * ray_y
    
    # Add camera position to get absolute ENU coordinates
    x_world = x_relative + camera_east
    y_world = y_relative + camera_north
    
    # Reshape back to downsampled image dimensions
    x_world = x_world.reshape(image_height_ds, image_width_ds)
    y_world = y_world.reshape(image_height_ds, image_width_ds)
    
    return x_world, y_world

def imu_to_camera_enu(
    imu_east: float, imu_north: float, imu_up: float,
    lever_arm_x: float, lever_arm_y: float, lever_arm_z: float,
    pitch: float, roll: float, heading: float
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    
    pitch_rad, roll_rad, heading_rad = map(np.radians, [pitch, roll, heading])
    lever_arm_body = np.array([lever_arm_x, lever_arm_y, lever_arm_z])
    
    R_roll = np.array([[np.cos(roll_rad), 0, np.sin(roll_rad)], [0, 1, 0], [-np.sin(roll_rad), 0, np.cos(roll_rad)]])
    R_pitch = np.array([[1, 0, 0], [0, np.cos(pitch_rad), -np.sin(pitch_rad)], [0, np.sin(pitch_rad), np.cos(pitch_rad)]])
    R_heading = np.array([[np.cos(heading_rad), np.sin(heading_rad), 0], [-np.sin(heading_rad), np.cos(heading_rad), 0], [0, 0, 1]])
    
    R_body_to_enu = R_heading @ R_roll @ R_pitch
    lever_arm_enu = R_body_to_enu @ lever_arm_body
    
    camera_pos = (imu_east + lever_arm_enu[0], imu_north + lever_arm_enu[1], imu_up + lever_arm_enu[2])
    return camera_pos, (lever_arm_enu[0], lever_arm_enu[1], lever_arm_enu[2])

def build_camera_rotation_matrix(pitch: float, roll: float, heading: float) -> np.ndarray:
    pitch_rad, roll_rad, heading_rad = map(np.radians, [pitch, roll, heading])
    pitch_rad = np.radians(0) + pitch_rad
    # pitch_rad = np.radians(-90) + pitch_rad

    R_pitch = np.array([[1, 0, 0], [0, np.cos(pitch_rad), -np.sin(pitch_rad)], [0, np.sin(pitch_rad), np.cos(pitch_rad)]])
    R_roll = np.array([[np.cos(roll_rad), 0, np.sin(roll_rad)], [0, 1, 0], [-np.sin(roll_rad), 0, np.cos(roll_rad)]])
    R_heading = np.array([[np.cos(heading_rad), np.sin(heading_rad), 0], [-np.sin(heading_rad), np.cos(heading_rad), 0], [0, 0, 1]])
    
    R_camera_to_world = R_heading @ R_roll @ R_pitch
    return R_camera_to_world.T

def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w, x, y, z = 0.25 / s, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    quat = np.array([w, x, y, z])
    return quat / np.linalg.norm(quat)


# ============================================================================
# EXPORT UTILITIES (COLMAP, JSON, PLY)
# ============================================================================
def write_ply(filename: str, points: np.ndarray, colors: np.ndarray):
    """Writes a 3DGS-compatible binary PLY file."""
    print(f"Writing PLY point cloud to {filename}...")
    num_points = len(points)
    normals = np.zeros_like(points) # 3DGS expects normals, usually 0 is fine for init
    
    with open(filename, 'wb') as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {num_points}\n".encode('utf-8'))
        f.write(b"property double x\nproperty double y\nproperty double z\n")
        f.write(b"property double nx\nproperty double ny\nproperty double nz\n")
        f.write(b"property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(b"end_header\n")
        
        for i in range(num_points):
            f.write(struct.pack('<ddddddBBB', 
                points[i, 0], points[i, 1], points[i, 2],
                normals[i, 0], normals[i, 1], normals[i, 2],
                colors[i, 0], colors[i, 1], colors[i, 2]
            ))

def write_colmap_and_json(out_dir: str, df_used_images: pd.DataFrame, image_dir: str):
    """Generates COLMAP bins and Nerfstudio JSON for the used images."""
    colmap_dir = os.path.join(out_dir, "sparse", "0")
    os.makedirs(colmap_dir, exist_ok=True)
    
    # 1. Write cameras.bin
    with open(os.path.join(colmap_dir, 'cameras.bin'), 'wb') as f:
        f.write(struct.pack('<Q', 1)) # 1 camera (uint64)
        f.write(struct.pack('<i', 1)) # camera_id (int32) <--- FIXED from <Q
        f.write(struct.pack('<i', 3)) # OPENCV model_id (int32)
        f.write(struct.pack('<Q', IMAGE_WIDTH))
        f.write(struct.pack('<Q', IMAGE_HEIGHT))
        for p in [FOCAL_LENGTH_PX, FOCAL_LENGTH_PX, IMAGE_WIDTH/2.0, IMAGE_HEIGHT/2.0, K1, K2, P1, P2]:
            f.write(struct.pack('<d', p))

    # 2. Setup JSON dictionary
    ns_json = {
        "camera_model": "OPENCV",
        "fl_x": FOCAL_LENGTH_PX,
        "fl_y": FOCAL_LENGTH_PX,
        "cx": IMAGE_WIDTH / 2.0,
        "cy": IMAGE_HEIGHT / 2.0,
        "w": IMAGE_WIDTH,
        "h": IMAGE_HEIGHT,
        "k1": K1, "k2": K2, "p1": P1, "p2": P2,
        "ply_file_path": "sparse/0/points3D.ply",
        "frames": []
    }

    images_bin_path = os.path.join(colmap_dir, 'images.bin')
    with open(images_bin_path, 'wb') as f:
        f.write(struct.pack('<Q', len(df_used_images)))
        
        for idx, (_, row) in enumerate(df_used_images.iterrows(), 1):
            fname = str(row['filename']) 
            pitch, roll, heading = -float(row['pitch']), float(row['roll']), float(row['heading'])
            e, n, u = float(row['easting']), float(row['northing']), float(row['depth'])
            
            cam_pos, _ = imu_to_camera_enu(e, n, u, LEVER_ARM_X, LEVER_ARM_Y, LEVER_ARM_Z, pitch, roll, heading)
            
            R_w2c = build_camera_rotation_matrix(pitch + PITCH_OFFSET, roll + ROLL_OFFSET, heading + HEADING_OFFSET)
            quat = rotation_matrix_to_quaternion(R_w2c)
            
            # Write COLMAP images.bin
            f.write(struct.pack('<i', idx)) # image_id (int32) <--- FIXED from <Q
            for q in quat: f.write(struct.pack('<d', q))
            for pos in cam_pos: f.write(struct.pack('<d', pos))
            f.write(struct.pack('<i', 1)) # camera_id (int32) <--- FIXED from <Q
            f.write(fname.encode('utf-8') + b'\x00')
            f.write(struct.pack('<Q', 0)) # 0 points2D
            
            # Write Nerfstudio JSON
            R_c2w = R_w2c.T
            c2w_matrix = np.eye(4)
            c2w_matrix[:3, :3] = R_c2w
            c2w_matrix[:3, 3] = cam_pos
            
            ns_json["frames"].append({
                "file_path": f"images/{fname}",
                "transform_matrix": c2w_matrix.tolist()
            })
            
            # Copy image to the expected folders
            src_img = os.path.join(image_dir, fname)
            dst_img = os.path.join(out_dir, "images", fname)
            if os.path.exists(src_img) and not os.path.exists(dst_img):
                shutil.copy2(src_img, dst_img)

    with open(os.path.join(out_dir, 'transforms.json'), 'w') as f:
        json.dump(ns_json, f, indent=4)
        
    print(f"Exported COLMAP to {colmap_dir} and JSON to {out_dir}/transforms.json")

    with open(os.path.join(colmap_dir, 'points3D.bin'), 'wb') as f:
        f.write(struct.pack('<Q', 0)) # Write 0 to indicate zero points in the bin file


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def run_pipeline(
    laz_path: str, 
    csv_path: str, 
    image_dir: str, 
    out_dir: str,
    downsample: int = 3
):
    import laspy
    from scipy.spatial import cKDTree
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "sparse", "0"), exist_ok=True)

    print(f"Reading LAZ: {laz_path}")
    las = laspy.read(laz_path)
    las_e, las_n, las_z = np.array(las.x, dtype=np.float64), np.array(las.y, dtype=np.float64), np.array(las.z, dtype=np.float64)

    e_lo, e_hi = las_e.min(), las_e.max()
    n_lo, n_hi = las_n.min(), las_n.max()

    df_imgs = pd.read_csv(csv_path)
    used_images = set()

    print("Building KD-tree from point cloud...")
    las_tree = cKDTree(np.column_stack([las_e, las_n]))
    colors_rgb = np.full((len(las_e), 3), -1, dtype=np.int32)
    best_distances = np.full(len(las_e), np.inf)

    total = len(df_imgs)
    print(f"Processing {total} images for colorization...")

    for seq, (_, img_row) in enumerate(df_imgs.iterrows(), 1):
        fname = str(img_row['filename'])
        fpath = os.path.join(image_dir, fname)
        if not os.path.exists(fpath): continue

        pitch, roll, heading = -float(img_row['pitch']), float(img_row['roll']), float(img_row['heading'])
        e, n, u, alt = float(img_row['easting']), float(img_row['northing']), float(img_row['depth']), float(img_row['altitude'])

        cam_pos, shift = imu_to_camera_enu(e, n, u, LEVER_ARM_X, LEVER_ARM_Y, LEVER_ARM_Z, pitch, roll, heading)

        px_e, px_n = pixels_to_world_coordinates(
            distance_off_bottom=alt + shift[2],
            pitch=pitch + PITCH_OFFSET, roll=roll + ROLL_OFFSET, heading=heading + HEADING_OFFSET,
            image_path=fpath, camera_east=cam_pos[0], camera_north=cam_pos[1], downsample=downsample
        )

        img_bgr = cv2.imread(fpath)
        if img_bgr is None: continue
        if downsample > 1:
            img_bgr = cv2.resize(img_bgr, (img_bgr.shape[1] // downsample, img_bgr.shape[0] // downsample), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        e_flat, n_flat = px_e.flatten(), px_n.flatten()
        rgb_flat = img_rgb.reshape(-1, 3)

        mask = (~np.isnan(e_flat) & ~np.isnan(n_flat) &
                (e_flat >= e_lo) & (e_flat <= e_hi) &
                (n_flat >= n_lo) & (n_flat <= n_hi))
        if not mask.any():
            print(f"  [{seq}/{total}] Skipped  {fname} - no overlap with point cloud bbox")
            continue

        ev, nv, rgbv = e_flat[mask], n_flat[mask], rgb_flat[mask]

        distances, indices = las_tree.query(np.column_stack([ev, nv]))

        # Assign color from the closest projected pixel across all images
        update_mask = distances < best_distances[indices]
        update_indices = indices[update_mask]
        best_distances[update_indices] = distances[update_mask]
        colors_rgb[update_indices] = rgbv[update_mask]

        used_images.add(fname)
        print(f"  [{seq}/{total}] Processed {fname} - {int(update_mask.sum())} LAS pts colored")

    has_color = colors_rgb[:, 0] >= 0
    if has_color.any():
        points_3d = np.column_stack((las_e[has_color], las_n[has_color], las_z[has_color]))
        final_colors = colors_rgb[has_color].astype(np.uint8)

        print(f"Colorized {has_color.sum()} / {len(las_e)} points.")

        ply_path = os.path.join(out_dir, "sparse", "0", "points3D.ply")
        write_ply(ply_path, points_3d, final_colors)
    else:
        print("No valid points generated! Exiting.")
        return


    # Filter CSV to only include used images
    df_used = df_imgs[df_imgs['filename'].isin(used_images)].copy()
    print(f"Exporting camera poses for {len(df_used)} utilized images...")
    
    # Generate COLMAP and JSON files
    write_colmap_and_json(out_dir, df_used, image_dir)
    print("PIPELINE COMPLETE.")

if __name__ == "__main__":
    # SET YOUR PATHS HERE
    LAZ_FILE = r"E:\EN2501_LLS\DIVE032_Stokey\products\LLS\LLS_2025-08-31T102452.007600_2_V2.laz"
    IMAGE_CSV = r"E:\EN2501_LLS\DIVE032_Stokey\processing\image\image_file_list.csv"
    IMAGE_DIR = r"E:\EN2501_LLS\DIVE032_Stokey\processing\image\images"
    OUTPUT_WORKSPACE = r"E:\EN2501_LLS\DIVE032_Stokey\processing\image\nerf_workspace"
    
    run_pipeline(LAZ_FILE, IMAGE_CSV, IMAGE_DIR, OUTPUT_WORKSPACE)
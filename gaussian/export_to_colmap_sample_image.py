"""
Modified COLMAP Export Tool - Orientation Troubleshooting Version
"""

import os
import struct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple

# ============================================================================
# CONFIGURATION 
# ============================================================================

IMAGE_LIST_CSV = r"E:\EN2501_LLS\DIVE032_Stokey\processing\image\image_file_list.csv"
OUTPUT_DIR = r"E:\EN2501_LLS\DIVE032_Stokey\processing\image\colmap_workspace"

FOCAL_LENGTH_PX = 3801.37053  
IMAGE_WIDTH = 4096            
IMAGE_HEIGHT = 3008           

K1 = 0.0113579
K2 = -0.0143928
P1 = 0.0042688
P2 = -0.000244194

LEVER_ARM_X = 0.125813   
LEVER_ARM_Y = 0.945584   
LEVER_ARM_Z = -0.213513  

PITCH_OFFSET = 0.010
ROLL_OFFSET = 0.010
HEADING_OFFSET = 0.000

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def imu_to_camera_enu(imu_east, imu_north, imu_up, lever_arm_x, lever_arm_y, lever_arm_z, pitch, roll, heading):
    pitch_rad = np.radians(pitch)
    roll_rad = np.radians(roll)
    heading_rad = np.radians(heading)
    
    lever_arm_body = np.array([lever_arm_x, lever_arm_y, lever_arm_z])
    
    R_roll = np.array([
        [np.cos(roll_rad), 0, np.sin(roll_rad)],
        [0, 1, 0],
        [-np.sin(roll_rad), 0, np.cos(roll_rad)]
    ])
    
    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad), np.cos(pitch_rad)]
    ])
    
    R_heading = np.array([
        [np.cos(heading_rad), np.sin(heading_rad), 0],
        [-np.sin(heading_rad), np.cos(heading_rad), 0],
        [0, 0, 1]
    ])
    
    R_body_to_enu = R_heading @ R_roll @ R_pitch
    lever_arm_enu = R_body_to_enu @ lever_arm_body
    
    camera_east = imu_east + lever_arm_enu[0]
    camera_north = imu_north + lever_arm_enu[1] 
    camera_up = imu_up + lever_arm_enu[2]
    
    return (camera_east, camera_north, camera_up)

def rotation_matrix_to_quaternion(R):
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    quat = np.array([w, x, y, z])
    return quat / np.linalg.norm(quat)

def build_camera_rotation_matrix(pitch, roll, heading):
    pitch_rad = np.radians(pitch)
    roll_rad = np.radians(roll)
    heading_rad = np.radians(heading)
    
    # Current downward-looking adjustment
    pitch_rad = np.radians(0) + pitch_rad
    
    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad), np.cos(pitch_rad)]
    ])
    
    R_roll = np.array([
        [np.cos(roll_rad), 0, np.sin(roll_rad)],
        [0, 1, 0],
        [-np.sin(roll_rad), 0, np.cos(roll_rad)]
    ])
    
    R_heading = np.array([
        [np.cos(heading_rad), np.sin(heading_rad), 0],
        [-np.sin(heading_rad), np.cos(heading_rad), 0],
        [0, 0, 1]
    ])
    
    R_camera_to_world = R_heading @ R_roll @ R_pitch
    R_world_to_camera = R_camera_to_world.T
    
    return R_world_to_camera

def compute_camera_pose(image_row, lever_arm_x, lever_arm_y, lever_arm_z, pitch_offset, roll_offset, heading_offset):
    pitch = -image_row['pitch'] 
    roll = image_row['roll']
    heading = image_row['heading']
    imu_east = image_row['easting']
    imu_north = image_row['northing']
    imu_up = image_row['depth']  
    
    camera_pos = imu_to_camera_enu(
        imu_east, imu_north, imu_up,
        lever_arm_x, lever_arm_y, lever_arm_z,
        pitch, roll, heading
    )
    
    pitch_corrected = pitch + pitch_offset
    roll_corrected = roll + roll_offset
    heading_corrected = heading + heading_offset
    
    R_world_to_camera = build_camera_rotation_matrix(pitch_corrected, roll_corrected, heading_corrected)
    quat = rotation_matrix_to_quaternion(R_world_to_camera)
    position = np.array([camera_pos[0], camera_pos[1], camera_pos[2]])
    
    # We now return R_world_to_camera as well to make plotting easier
    return position, quat, R_world_to_camera


# ============================================================================
# TROUBLESHOOTING PLOTTER
# ============================================================================

def plot_troubleshooting_poses(poses):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Camera Poses Troubleshooting (COLMAP Convention)")
    
    for i, (position, quat, R_w2c) in enumerate(poses):
        c_x, c_y, c_z = position
        
        # Camera to World rotation (to define where the axes point in the world)
        R_c2w = R_w2c.T
        
        # Extract camera axes in the world frame
        x_axis = R_c2w[:, 0]  # Camera +X (COLMAP expects this to be RIGHT)
        y_axis = R_c2w[:, 1]  # Camera +Y (COLMAP expects this to be DOWN)
        z_axis = R_c2w[:, 2]  # Camera +Z (COLMAP expects this to be LOOK DIRECTION)
        
        # Plot camera center
        ax.scatter(c_x, c_y, c_z, color='black', s=50)
        ax.text(c_x, c_y, c_z, f' Cam {i+1}', size=10, zorder=1)
        
        # Draw coordinate axes for the camera
        # Scale determines how long the arrows are drawn (adjust if too big/small)
        scale = 0.5 
        ax.quiver(c_x, c_y, c_z, x_axis[0], x_axis[1], x_axis[2], color='r', length=scale, normalize=True, label='X (Right)' if i==0 else "")
        ax.quiver(c_x, c_y, c_z, y_axis[0], y_axis[1], y_axis[2], color='g', length=scale, normalize=True, label='Y (Down)' if i==0 else "")
        ax.quiver(c_x, c_y, c_z, z_axis[0], z_axis[1], z_axis[2], color='b', length=scale, normalize=True, label='Z (Look/Optical Axis)' if i==0 else "")

    ax.set_xlabel('Easting (X)')
    ax.set_ylabel('Northing (Y)')
    ax.set_zlabel('Up (Z)')
    
    # Set equal aspect ratio for realistic viewing
    ax.set_box_aspect([1,1,1])
    ax.legend()
    plt.show()


# ============================================================================
# COLMAP BINARY FILE WRITERS
# ============================================================================

def write_images_bin(output_path: str, df_images: pd.DataFrame, poses: list):
    camera_id = 1 
    with open(output_path, 'wb') as f:
        f.write(struct.pack('<Q', len(df_images)))
        
        for idx, ((_, img_row), (position, quat, R_w2c)) in enumerate(zip(df_images.iterrows(), poses), 1):
            
            # --- CRITICAL FIX ---
            # COLMAP expects translation `t` which is the projection of the world origin 
            # into the camera frame. t = -R * C
            t = -np.dot(R_w2c, position)
            
            f.write(struct.pack('<Q', idx))            
            f.write(struct.pack('<d', quat[0]))        
            f.write(struct.pack('<d', quat[1]))        
            f.write(struct.pack('<d', quat[2]))        
            f.write(struct.pack('<d', quat[3]))        
            
            # Write corrected translation
            f.write(struct.pack('<d', t[0]))     
            f.write(struct.pack('<d', t[1]))     
            f.write(struct.pack('<d', t[2]))     
            
            f.write(struct.pack('<Q', camera_id))
            f.write(img_row['filename'].encode('utf-8') + b'\x00')
            f.write(struct.pack('<Q', 0))

# [Note: write_cameras_bin and write_points3D_bin remain exactly the same as your original script]
def write_cameras_bin(output_path: str):
    with open(output_path, 'wb') as f:
        f.write(struct.pack('<Q', 1))
        f.write(struct.pack('<Q', 1))
        f.write(struct.pack('<i', 3))
        f.write(struct.pack('<Q', IMAGE_WIDTH))
        f.write(struct.pack('<Q', IMAGE_HEIGHT))
        for param in [FOCAL_LENGTH_PX, FOCAL_LENGTH_PX, IMAGE_WIDTH/2.0, IMAGE_HEIGHT/2.0, K1, K2, P1, P2]:
            f.write(struct.pack('<d', param))

def write_points3D_bin(output_path: str):
    with open(output_path, 'wb') as f:
        f.write(struct.pack('<Q', 0))

# ============================================================================
# MAIN PROCESSING
# ============================================================================

def main():
    print("Reading image navigation data...")
    if not os.path.exists(IMAGE_LIST_CSV):
        print(f"ERROR: Image list CSV not found: {IMAGE_LIST_CSV}")
        return
        
    df_images = pd.read_csv(IMAGE_LIST_CSV)
    
    # --- TROUBLESHOOTING: SUBSET TO 5 CONSECUTIVE IMAGES ---
    df_images = df_images.head(5)
    print(f"Subset to {len(df_images)} images for troubleshooting.")
    
    poses = []
    for idx, (_, img_row) in enumerate(df_images.iterrows(), 1):
        position, quat, R_w2c = compute_camera_pose(
            img_row, LEVER_ARM_X, LEVER_ARM_Y, LEVER_ARM_Z,
            PITCH_OFFSET, ROLL_OFFSET, HEADING_OFFSET
        )
        poses.append((position, quat, R_w2c))
    
    # Plot the poses to troubleshoot orientation visually
    print("Launching troubleshooting plot. Close the plot window to continue export...")
    plot_troubleshooting_poses(poses)
    
    print("Writing COLMAP binary files...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    write_cameras_bin(os.path.join(OUTPUT_DIR, 'cameras.bin'))
    write_images_bin(os.path.join(OUTPUT_DIR, 'images.bin'), df_images, poses)
    write_points3D_bin(os.path.join(OUTPUT_DIR, 'points3D.bin'))
    print("Export Complete.")

if __name__ == "__main__":
    main()
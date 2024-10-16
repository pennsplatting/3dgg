import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import numpy as np
# import open3d as o3d
import cv2

output_dir = "/home/zuoxy/cg/3dgp/debug/"
def visualize_rays(ray_origins, ray_directions, scale=1.0):
    """
    Visualize the ray origins and directions in a 3D plot.
    :param ray_origins: Tensor of shape [num_rays, 3], the 3D origin of each ray.
    :param ray_directions: Tensor of shape [num_rays, 3], the 3D direction of each ray.
    :param scale: Float, controls the length of the direction vectors.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the origins
    ax.scatter(ray_origins[:, 0].cpu().numpy(), 
               ray_origins[:, 1].cpu().numpy(), 
               ray_origins[:, 2].cpu().numpy(), 
               c='r', marker='o', label='Ray Origins')

    # Plot the directions
    ax.quiver(ray_origins[:, 0].cpu().numpy(),
              ray_origins[:, 1].cpu().numpy(),
              ray_origins[:, 2].cpu().numpy(),
              ray_directions[:, 0].cpu().numpy(),
              ray_directions[:, 1].cpu().numpy(),
              ray_directions[:, 2].cpu().numpy(),
              length=scale, color='b', label='Ray Directions')

    # # Labels and title
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_title('Ray Origins and Directions')
    # plt.legend()
    # plt.savefig(output_dir + "ray_origins_and_directions.png")

# Function to visualize and save depth map
def save_depth_map(depth_map, output_path=output_dir+'depth_map.png'):
    depth_map = np.squeeze(depth_map)  # Remove the last dimension

    # Normalize depth map for visualization
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_normalized = depth_map_normalized.astype(np.uint8)
    
    # Save the depth map image without displaying it
    cv2.imwrite(output_path, depth_map_normalized)

# Function to visualize and save XYZ point cloud as an image
# def save_point_cloud(XYZ_points, output_path=output_dir+'point_cloud.png'):
#     """
#     Save the point cloud visualization as an image.
#     XYZ_points should be of shape (H, W, 3).
#     """
#     # Reshape to (n, 3) where n = H * W
#     H, W, _ = XYZ_points.shape
#     XYZ_points_flat = XYZ_points.reshape(-1, 3)

#     # Create open3d point cloud object
#     point_cloud = o3d.geometry.PointCloud()
#     point_cloud.points = o3d.utility.Vector3dVector(XYZ_points_flat)
    
#     # Create a visualizer object
#     vis = o3d.visualization.Visualizer()
#     vis.create_window(visible=False)  # Set visible to False for cluster usage
#     vis.add_geometry(point_cloud)
    
#     # Capture the point cloud image
#     vis.poll_events()
#     vis.update_renderer()
#     vis.capture_screen_image(output_path)
#     vis.destroy_window()

# Function to visualize and save XYZ point cloud as an image using matplotlib
def save_point_cloud(XYZ_points, output_path=output_dir+'point_cloud.png'):
    """
    Save the point cloud visualization as an image using matplotlib.
    XYZ_points should be of shape (H, W, 3).
    """
    # Reshape to (n, 3) where n = H * W
    # H, W, _ = XYZ_points.shape
    # XYZ_points_flat = XYZ_points.reshape(-1, 3)

    # Extract x, y, z coordinates
    x = XYZ_points[:, 0]
    y = XYZ_points[:, 1]
    z = XYZ_points[:, 2]

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the point cloud with a colormap for depth (z-axis)
    scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=1)

    # Add color bar for depth
    fig.colorbar(scatter, ax=ax, label='Depth (z-axis)')

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Save the plot as an image file
    plt.savefig(output_path, dpi=300)
    plt.close(fig)  # Close the figure to release memory
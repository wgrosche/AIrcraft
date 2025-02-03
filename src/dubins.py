import numpy as np
import dubins

def normalize(v):
    """ Normalize a vector. """
    return v / np.linalg.norm(v)

def fit_plane(p1, p2, p3):
    """ Fit a plane to three points and return its normal vector and a point on the plane. """
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p1)
    normal = np.cross(v1, v2)  # Compute normal vector
    normal = normalize(normal)  # Normalize the normal vector
    return normal, np.array(p1)

def project_to_plane(point, plane_normal, plane_point):
    """ Project a point onto a plane defined by a normal and a reference point. """
    vec = np.array(point) - plane_point
    distance = np.dot(vec, plane_normal)  # Distance from plane
    projected_point = np.array(point) - distance * plane_normal  # Projection formula
    return projected_point

def generate_3d_dubins_path(waypoints, r_min, sample_dist=0.1):
    """
    Generates a 3D Dubins-like path by computing the shortest trajectory on a fitted plane.

    :param waypoints: List of waypoints [(x, y, z, theta)] where theta is heading in radians.
    :param r_min: Minimum turning radius.
    :param sample_dist: Sampling distance for the trajectory.
    :return: List of (x, y, z) points forming the 3D Dubins path.
    """
    path_points = []

    for i in range(len(waypoints) - 2):  # Use three consecutive waypoints to fit a plane
        (x1, y1, z1, theta1) = waypoints[i]
        (x2, y2, z2, theta2) = waypoints[i + 1]
        (x3, y3, z3, theta3) = waypoints[i + 2]

        # Fit a plane using three points
        normal, plane_point = fit_plane((x1, y1, z1), (x2, y2, z2), (x3, y3, z3))

        # Project start and end points onto the fitted plane
        p1_proj = project_to_plane((x1, y1, z1), normal, plane_point)
        p2_proj = project_to_plane((x2, y2, z2), normal, plane_point)

        # Compute the shortest Dubins path on the plane
        start_2d = (p1_proj[0], p1_proj[1], theta1)
        end_2d = (p2_proj[0], p2_proj[1], theta2)
        path_2d, _ = dubins.shortest_path(start_2d, end_2d, r_min).sample_many(sample_dist)

        # Convert 2D path points back to 3D by projecting them onto the plane
        path_segment = []
        for (x, y) in path_2d:
            proj_point = project_to_plane((x, y, z1), normal, plane_point)
            path_segment.append(tuple(proj_point))  # Store (x, y, z)

        path_points.extend(path_segment)

    return path_points

# Example usage:
waypoints_3d = [
    (0, 0, 0, 0), 
    (10, 10, 5, np.pi/4), 
    (20, 5, 10, np.pi/2), 
    (30, 15, 15, np.pi/3)
]  # (x, y, z, heading)
r_min = 2.0  # Minimum turn radius
trajectory_3d = generate_3d_dubins_path(waypoints_3d, r_min)

# Print trajectory points
for point in trajectory_3d:
    print(point)

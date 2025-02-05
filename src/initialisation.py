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

def transform_to_plane_coordinates(point, plane_normal, plane_point):
    """ Transform a 3D point to 2D coordinates in the plane's local (u, v) coordinate system. """
    # Create a basis for the plane
    plane_normal = normalize(plane_normal)
    u_axis = normalize(np.cross(plane_normal, [0, 0, 1]) if np.abs(plane_normal[2]) < 0.9 else np.cross(plane_normal, [1, 0, 0]))
    v_axis = np.cross(plane_normal, u_axis)

    # Compute the vector from the reference point on the plane to the input point
    vec = np.array(point) - plane_point

    # Project the vector onto the (u, v) basis
    u_coord = np.dot(vec, u_axis)
    v_coord = np.dot(vec, v_axis)

    return u_coord, v_coord

def transform_heading_to_plane(theta, plane_normal, u_axis):
    """ 
    Transform the 3D heading angle to the corresponding 2D heading in the plane's (u, v) coordinates.
    :param theta: 3D heading angle (in radians)
    :param plane_normal: Normal vector of the plane
    :param u_axis: The u-axis of the plane's coordinate system
    :return: Transformed 2D heading angle (in radians)
    """
    # Compute the 3D heading vector
    heading_vector = np.array([np.cos(theta), np.sin(theta), 0])

    # Project the heading vector onto the plane
    heading_vector_on_plane = heading_vector - np.dot(heading_vector, plane_normal) * plane_normal
    heading_vector_on_plane = normalize(heading_vector_on_plane)

    # Compute the 2D heading angle relative to the u-axis
    u_angle = np.arctan2(np.dot(heading_vector_on_plane, np.cross(plane_normal, u_axis)),
                        np.dot(heading_vector_on_plane, u_axis))
    return u_angle


def generate_3d_dubins_path(waypoints, r_min, sample_dist=0.01):
    path_points = []

    for i in range(len(waypoints) - 1):
        (x1, y1, z1, theta1) = waypoints[i]
        (x2, y2, z2, theta2) = waypoints[i + 1]


        if i == len(waypoints) - 2:
            normal, plane_point = fit_plane((x1, y1, z1), (x2, y2, z2), waypoints[i - 1][:3])
        else:
            # Fit a plane using three points
            normal, plane_point = fit_plane((x1, y1, z1), (x2, y2, z2), waypoints[i + 2][:3])

        plane_normal = normalize(normal)
        u_axis = normalize(np.cross(plane_normal, [0, 0, 1]) if np.abs(plane_normal[2]) < 0.9 else np.cross(plane_normal, [1, 0, 0]))
        v_axis = np.cross(plane_normal, u_axis)

        # Transform points to 2D plane coordinates
        start_2d = (*transform_to_plane_coordinates((x1, y1, z1), normal, plane_point), transform_heading_to_plane(theta1, normal, u_axis))
        end_2d = (*transform_to_plane_coordinates((x2, y2, z2), normal, plane_point), transform_heading_to_plane(theta2, normal, u_axis))

        # Compute the 2D Dubins path
        path_2d, _ = dubins.shortest_path(start_2d, end_2d, r_min).sample_many(sample_dist)

        # Convert back to 3D in the plane's coordinate system
        path_segment = []
        for u, v, _ in path_2d:
            point_3d = plane_point + u * u_axis + v * v_axis
            path_segment.append(tuple(point_3d))

        path_points.extend(path_segment)

    return path_points

def setup_waypoints(x_initial, waypoints):
    """
    Sets up waypoints for the 3D Dubins path generation.

    :param waypoints: List of waypoints [(x, y, z)]

    returns

    waypoints_with_dubins: List of waypoints with Dubins-like headings [(x, y, z, theta)]
    where theta points to the next waypoint
    """

    p_initial, v_initial, _, _ = x_initial

    initial_heading = np.arctan2(v_initial[1], v_initial[0])

    # waypoints.insert(0, (p_initial[0], p_initial[1], p_initial[2]))

    waypoints_with_dubins = [(p_initial[0], p_initial[1], p_initial[2], initial_heading)]

    for i in range(len(waypoints) - 1):
        x1, y1, z1 = waypoints_with_dubins[i]
        x2, y2, z2 = waypoints[i]
        x3, y3, z3 = waypoints[i + 1]
        theta = np.arctan2(y2 - y1, x2 - x1)
        waypoints_with_dubins.append((x1, y1, z1, theta))
        # waypoints_with_dubins.append((x2, y2, z2, theta))


# Example usage:
waypoints_3d = [
    (0, 0, 0, 0), 
    (10, 10, 5, np.pi/4), 
    (20, 5, 10, np.pi/2), 
    (30, 15, 15, np.pi/3),

    (40, 15, 15, 0)
]  # (x, y, z, heading)
r_min = 5.0  # Minimum turn radius
trajectory_3d = generate_3d_dubins_path(waypoints_3d, r_min)

# Print trajectory points
for point in trajectory_3d:
    print(point)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_3d_dubins_path(waypoints, trajectory):
    """
    Visualizes the 3D Dubins-like path and waypoints.
    
    :param waypoints: List of waypoints [(x, y, z, theta)].
    :param trajectory: List of (x, y, z) points representing the 3D Dubins path.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the waypoints
    waypoints_x = [p[0] for p in waypoints]
    waypoints_y = [p[1] for p in waypoints]
    waypoints_z = [p[2] for p in waypoints]
    ax.scatter(waypoints_x, waypoints_y, waypoints_z, color='red', label='Waypoints', s=50)

    # Plot the trajectory
    traj_x = [p[0] for p in trajectory]
    traj_y = [p[1] for p in trajectory]
    traj_z = [p[2] for p in trajectory]
    ax.plot(traj_x, traj_y, traj_z, color='blue', label='3D Dubins Path', linewidth=2)

    # Customize the plot
    ax.set_title('3D Dubins-like Path Visualization')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.grid(True)
    plt.show()

# Visualize the generated 3D Dubins path
visualize_3d_dubins_path(waypoints_3d, trajectory_3d)


import numpy as np
import dubins  # Install with `pip install dubins`

def sample_dubins_path(path, min_interval=0.1, max_interval=1.0, curvature_factor=1.0):
    """Sample a Dubins path at intervals based on curvature.
    
    Args:
        path: Dubins path object from `dubins` library.
        min_interval: Minimum sampling interval.
        max_interval: Maximum sampling interval.
        curvature_factor: Scaling factor for curvature sensitivity.
    
    Returns:
        List of (x, y, theta) samples.
    """
    def compute_curvature(segment_type, radius):
        if segment_type in ['L', 'R']:  # Left or Right turns
            return 1 / radius
        else:  # Straight segment
            return 0

    samples = []
    total_length = path.path_length()
    s = 0

    while s < total_length:
        (x, y, theta) = path.sample(s)
        segment_type, segment_length, radius = path.segment_info(s)
        
        # Compute curvature at this segment
        curvature = compute_curvature(segment_type, radius)
        
        # Determine the next sampling interval
        interval = curvature_factor / (1 + abs(curvature))
        interval = np.clip(interval, min_interval, max_interval)
        
        samples.append((x, y, theta))
        s += interval

    return samples


import numpy as np
from scipy.spatial.transform import Rotation as R

def compute_angular_velocity(quaternions, time_intervals):
    """Compute angular velocity from a list of quaternions and time intervals.
    
    Args:
        quaternions: List of scipy Rotation objects (quaternions).
        time_intervals: List of time intervals between consecutive quaternions.
    
    Returns:
        List of angular velocity vectors (in radians/sec) for each interval.
    """
    angular_velocities = []

    for i in range(1, len(quaternions)):
        q1 = quaternions[i - 1]
        q2 = quaternions[i]

        # Compute relative rotation (q2 * q1^-1)
        delta_q = q2 * q1.inv()
        axis, angle = delta_q.as_rotvec(), np.linalg.norm(delta_q.as_rotvec())

        # Angular velocity (omega = angle / delta_t * axis)
        delta_t = time_intervals[i - 1]
        omega = (angle / delta_t) * axis / np.linalg.norm(axis)
        angular_velocities.append(omega)

    return angular_velocities

    import numpy as np
from scipy.spatial.transform import Rotation as R

def compute_roll_angle(curvature, velocity, g=9.81):
    """Compute the roll angle for a coordinated turn.
    
    Args:
        curvature: Curvature of the path at the current point (1/m).
        velocity: Forward velocity of the drone (m/s).
        g: Gravitational acceleration (m/s^2).
    
    Returns:
        Roll angle in radians.
    """
    return np.arctan(curvature * velocity**2 / g)

def apply_roll_to_quaternion(base_quaternion, velocity_vector, roll_angle):
    """Apply a roll angle to a quaternion about the velocity vector.
    
    Args:
        base_quaternion: Original orientation as a scipy Rotation object.
        velocity_vector: 3D velocity vector (unit vector).
        roll_angle: Roll angle to apply (in radians).
    
    Returns:
        New quaternion with the applied roll.
    """
    # Create a rotation about the velocity vector by the roll angle
    roll_rotation = R.from_rotvec(roll_angle * velocity_vector)
    
    # Apply the roll rotation to the base quaternion
    new_orientation = roll_rotation * base_quaternion
    return new_orientation

# Assume we have a list of positions and quaternions for a 3D Dubins path
positions = np.array([...])  # List of (x, y, z) positions
quaternions = [R.from_quat([...]) for _ in range(len(positions))]  # List of quaternions
velocity = 15.0  # m/s

roll_angles = []
new_orientations = []

for i in range(1, len(positions) - 1):
    # Approximate curvature using finite differences
    r1 = positions[i - 1]
    r2 = positions[i]
    r3 = positions[i + 1]
    curvature = np.linalg.norm(np.cross(r2 - r1, r3 - r2)) / np.linalg.norm(r2 - r1)**3

    # Compute roll angle for the current sample
    roll_angle = compute_roll_angle(curvature, velocity)
    roll_angles.append(roll_angle)

    # Apply roll to the current quaternion
    velocity_vector = (r3 - r1) / np.linalg.norm(r3 - r1)  # Approximate velocity direction
    new_orientation = apply_roll_to_quaternion(quaternions[i], velocity_vector, roll_angle)
    new_orientations.append(new_orientation)

import numpy as np
import dubins # https://github.com/AgRoboticsResearch/pydubins.git
from aircraft.utils.utils import TrajectoryConfiguration

import numpy as np
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""


"""

def cumulative_distances(waypoints:np.ndarray, VERBOSE:bool = False):
    """
    Given a set of waypoints, calculate the distance between each waypoint.

    Parameters
    ----------
    waypoints : np.array
        Array of waypoints. (d x n) where d is the dimension of the waypoints 
        and n is the number of waypoints. The first waypoint is taken as the 
        initial position.

    Returns
    -------
    distance : np.array
        Cumulative distance between waypoints.
    
    """
    differences = np.diff(waypoints, axis=0)
    distances = np.linalg.norm(differences, axis=1)
    distance = np.cumsum(distances)

    if VERBOSE: 
        print("Cumulative waypoint distances: ", distance)
    return distance

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

def sample_dubins_path(path, min_interval=0.1, max_interval=10.0, curvature_factor=1.0, r_min=10.0, vel=30):
    """Sample a Dubins path at intervals based on curvature."""
    # Path type mapping
    PATH_TYPES = {
        0: 'LSL',
        1: 'LSR', 
        2: 'RSL',
        3: 'RSR',
        4: 'RLR',
        5: 'LRL'
    }
    
    path_type = PATH_TYPES[path.path_type()]
    samples = []
    time_intervals = []
    total_length = path.path_length()
    s = 0
    print(dir(path))
    def compute_curvature(s):
        segment_length = 0
        current_s = s
        for i, segment in enumerate(path_type):
            segment_length = path.segment_length(i)
            if current_s <= segment_length:
                if segment in ['L', 'R']:
                    return 1.0 / r_min
                return 0.0
            current_s -= segment_length
        return 0.0
    
    while s < total_length:
        point = path.sample(s)
        samples.append(point)
        
        curvature = compute_curvature(s)
        interval = curvature_factor / (1 + abs(curvature))
        interval = np.clip(interval, min_interval, max_interval)
        
        s += interval
        time_intervals.append(interval / vel)
    return np.array(samples), time_intervals

def generate_3d_dubins_path(waypoints, r_min, sample_dist=0.01):
    path_points = []
    all_time_intervals = []
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
        # path_2d, _ = dubins.shortest_path(start_2d, end_2d, r_min).sample_many(sample_dist)

        path_2d, time_intervals = sample_dubins_path(dubins.shortest_path(start_2d, end_2d, r_min))

        # Convert back to 3D in the plane's coordinate system
        path_segment = []
        for u, v, _ in path_2d:
            point_3d = plane_point + u * u_axis + v * v_axis
            path_segment.append(tuple(point_3d))

        path_points.extend(path_segment)
        all_time_intervals.extend(time_intervals)

    print("!!!", len(path_points), len(time_intervals))

    return path_points, all_time_intervals

def setup_waypoints(x_initial, waypoints):
    """
    Sets up waypoints for the 3D Dubins path generation.

    Parameters
    ----------
    x_initial : tuple
        Initial state containing (position, velocity, orientation, angular_velocity)
    waypoints : list
        List of waypoints [(x, y, z)]

    Returns
    -------
    waypoints_with_dubins : list
        List of waypoints with Dubins-like headings [(x, y, z, theta)]
        where theta points to the next waypoint
    """
    p_initial, v_initial= x_initial[:3], x_initial[3:6]
    initial_heading = np.arctan2(v_initial[1], v_initial[0])
    
    # Initialize list with initial position and heading
    waypoints_with_dubins = [(p_initial[0], p_initial[1], p_initial[2], initial_heading)]
    
    # Add waypoints with propagated headings
    for i in range(1, len(waypoints)):
        # Calculate heading to next waypoint
        if i < len(waypoints) - 1:
            dx = waypoints[i+1][0] - waypoints[i][0]
            dy = waypoints[i+1][1] - waypoints[i][1]
            heading = np.arctan2(dy, dx)
        else:
            # For last waypoint, keep the heading from previous segment
            heading = waypoints_with_dubins[-1][3]
            
        waypoints_with_dubins.append((
            waypoints[i][0],
            waypoints[i][1], 
            waypoints[i][2],
            heading
        ))
    
    return waypoints_with_dubins



# def visualize_3d_dubins_path(waypoints, trajectory):
#     """
#     Visualizes the 3D Dubins-like path and waypoints.
    
#     :param waypoints: List of waypoints [(x, y, z, theta)].
#     :param trajectory: List of (x, y, z) points representing the 3D Dubins path.
#     """
#     fig = plt.figure(figsize=(12, 8))
#     ax = fig.add_subplot(111, projection='3d')

#     # Plot the waypoints
#     waypoints_x = [p[0] for p in waypoints]
#     waypoints_y = [p[1] for p in waypoints]
#     waypoints_z = [p[2] for p in waypoints]
#     ax.scatter(waypoints_x, waypoints_y, waypoints_z, color='red', label='Waypoints', s=50)

#     # Plot the trajectory
#     traj_x = [p[0] for p in trajectory]
#     traj_y = [p[1] for p in trajectory]
#     traj_z = [p[2] for p in trajectory]
#     ax.plot(traj_x, traj_y, traj_z, color='blue', label='3D Dubins Path', linewidth=2)

#     # Customize the plot
#     ax.set_title('3D Dubins-like Path Visualization')
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.legend()
#     ax.grid(True)
#     plt.show()

def visualize_3d_dubins_path(waypoints, trajectory, orientations=None):
    """
    Visualizes the 3D Dubins-like path and waypoints.
    
    :param waypoints: List of waypoints [(x, y, z, theta)].
    :param trajectory: List of (x, y, z) points representing the 3D Dubins path.
    :param orientations: List of orientation vectors to display as quivers
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
    x_axes = [R.from_quat(orientation).apply([1,0,0]) for orientation in orientations]
    y_axes = [R.from_quat(orientation).apply([0,1,0]) for orientation in orientations]
    z_axes = [R.from_quat(orientation).apply([0,0,1]) for orientation in orientations]
    # Plot orientation quivers if provided
    if orientations is not None:
        # Sample points along trajectory for quivers
        sample_indices = np.linspace(1, len(trajectory)-2, 20, dtype=int)
        ax.quiver(
            [trajectory[i][0] for i in sample_indices],
            [trajectory[i][1] for i in sample_indices], 
            [trajectory[i][2] for i in sample_indices],
            [x_axes[i-1][0] for i in sample_indices],
            [x_axes[i-1][1] for i in sample_indices],
            [x_axes[i-1][2] for i in sample_indices],
            color='green', length=10.0, normalize=True
        )

    # Customize the plot
    ax.set_title('3D Dubins-like Path Visualization')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.grid(True)
    plt.show()


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

# # Assume we have a list of positions and quaternions for a 3D Dubins path
# positions = np.array([...])  # List of (x, y, z) positions
# quaternions = [R.from_quat([...]) for _ in range(len(positions))]  # List of quaternions
# velocity = 15.0  # m/s



def get_velocity_directions(path_points):
    """Get velocity directions from 3D path points"""
    velocity_vectors = []
    for i in range(len(path_points)-1):
        # Get direction vector between current and next point
        direction = np.array(path_points[i+1]) - np.array(path_points[i])
        # Normalize to unit vector
        velocity_vectors.append(direction / np.linalg.norm(direction))
    # Add final velocity (same as last segment)
    velocity_vectors.append(velocity_vectors[-1])
    return velocity_vectors


class DubinsInitialiser:
    def __init__(self, trajectory:TrajectoryConfiguration):
        self.waypoints = trajectory.waypoints.waypoints
        initial_state = trajectory.waypoints.initial_state
        self.cumulative_distances = cumulative_distances(self.waypoints)
        print(self.cumulative_distances )
        print(self.waypoints)

        if len(trajectory.waypoints.waypoint_indices) < 3:
            for i, waypoint in enumerate(self.waypoints[1:]):
                waypoint[2] = initial_state[2] + self.cumulative_distances[i] / trajectory.aircraft.glide_ratio

        self.dubins_waypoints = setup_waypoints(initial_state, self.waypoints)
        print(self.dubins_waypoints)
        self.dubins_path, time_intervals = generate_3d_dubins_path(self.dubins_waypoints, trajectory.aircraft.r_min)
        print(len(self.dubins_path))
        vel_directions = get_velocity_directions(self.dubins_path)
        rotations = [R.align_vectors(vel_dir, [1, 0, 0])[0] for vel_dir in vel_directions]
        roll_angles = []
        new_orientations = []
        for i in range(1, len(self.dubins_path) - 1):
            # Approximate curvature using finite differences
            r1 = np.array(self.dubins_path[i - 1])
            r2 = np.array(self.dubins_path[i])
            r3 = np.array(self.dubins_path[i + 1])
            curvature = np.linalg.norm(np.cross(r2 - r1, r3 - r2)) / np.linalg.norm(r2 - r1)**3

            # Compute roll angle for the current sample
            roll_angle = compute_roll_angle(curvature, trajectory.waypoints.default_velocity)
            roll_angles.append(roll_angle)

            # Apply roll to the current quaternion
            velocity_vector = (r3 - r1) / np.linalg.norm(r3 - r1)  # Approximate velocity direction
            new_orientation = apply_roll_to_quaternion(rotations[i], velocity_vector, roll_angle)
            new_orientations.append(new_orientation)
        print(len(time_intervals))
        print(len(new_orientations))
        angular_velocities = compute_angular_velocity(new_orientations, time_intervals)


        # print("Angular Velocities: ", angular_velocities)
        # print("Orientations: ", new_orientations)
        self.orientations = [orientation.as_quat() for orientation in new_orientations]

        print(len(self.orientations))
        print(len(self.dubins_path))
        # compute orientations, velocities and angular velocities

    @property
    def trajectory(self):
        pass


    def visualise(self):
        # Visualize the generated 3D Dubins path
        visualize_3d_dubins_path(self.dubins_waypoints, self.dubins_path, orientations = self.orientations)

import json
traj_dict = json.load(open('data/glider/problem_definition.json'),)
config = TrajectoryConfiguration(traj_dict)
dubins_init = DubinsInitialiser(config)
dubins_init.visualise()


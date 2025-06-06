import numpy as np
import dubins # https://github.com/AgRoboticsResearch/pydubins.git
from aircraft.utils.utils import TrajectoryConfiguration
from scipy.interpolate import CubicSpline
from dubins import _DubinsPath
import numpy as np
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Union, Type, Tuple, cast

Point = np.ndarray
Points = list[Point] | np.ndarray
DubinsPoints = list[tuple[Point, float]]
Vector = np.ndarray
Point2d = np.ndarray

def cumulative_distances(waypoints: Points, verbose: bool = False) -> np.ndarray:
    """
    Given a set of waypoints, calculate the cumulative distance along the path.

    Parameters
    ----------
    waypoints : Points
        List of waypoints. Each waypoint is a (3,) ndarray.

    Returns
    -------
    distance : np.ndarray
        Cumulative distance between waypoints (length n).
    """
    if not isinstance(waypoints, np.ndarray):
        waypoint_array = np.stack(waypoints, axis=1)  # shape (3, n)
    else:
        waypoint_array = waypoints
    diffs = np.diff(waypoint_array, axis=1)       # shape (3, n-1)
    distances = np.linalg.norm(diffs, axis=0)
    cumulative = np.concatenate([[0.0], np.cumsum(distances)])
    
    if verbose:
        for i, d in enumerate(cumulative):
            print(f"Waypoint {i}: {d:.3f} meters")

    return cumulative

def normalize(v:Vector) -> Vector:
    """ Normalize a vector. """
    if isinstance(v, list):
        v = np.array(v)
    return v / np.linalg.norm(v)

def fit_plane(p1:Point, p2:Point, p3:Point
              ) -> tuple[Vector, Point]:
    """ Fit a plane to three points and return its normal vector and a point on the plane. """
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p1)
    normal = np.cross(v1, v2)  # Compute normal vector
    normal = normalize(normal)  # Normalize the normal vector
    return normal, np.array(p1)

def project_to_plane(point:Point, plane_normal:Vector, 
                     plane_point:Point) -> Point:
    """ Project a point onto a plane defined by a normal and a reference point. """
    vec = np.array(point) - plane_point
    distance = np.dot(vec, plane_normal)  # Distance from plane
    projected_point = np.array(point) - distance * plane_normal  # Projection formula
    return projected_point

def transform_to_plane_coordinates(
        point:Point, 
        plane_normal:Vector, 
        plane_point:Point
        ) -> Point2d:
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

    return np.array([u_coord, v_coord])

def transform_heading_to_plane(theta:float, plane_normal:Vector, u_axis:Vector) -> float:
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

def sample_dubins_path(
        path:_DubinsPath, 
        min_interval:float=0.001, 
        max_interval:float=0.1, 
        curvature_factor:float=1.0, 
        r_min:float=10.0, 
        vel:float=30.0
        ) -> tuple[Points, list[float]]:
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
        samples.append(np.array(point))
        
        curvature = compute_curvature(s)
        interval = curvature_factor / (1 + abs(curvature))
        interval = np.clip(interval, min_interval, max_interval)
        
        s += interval
        time_intervals.append(interval / vel)
    return samples, time_intervals

def generate_3d_dubins_path(
        waypoints:DubinsPoints, 
        r_min:float
        ) -> tuple[Points, list[float]]:
    
    path_points = []
    all_time_intervals = []
    for i in range(len(waypoints) - 1):
        print(waypoints[i])
        point1, theta1 = waypoints[i]
        point2, theta2 = waypoints[i + 1]


        if i == len(waypoints) - 2:
            normal, plane_point = fit_plane(point1, point2, waypoints[i - 1][0])
        else:
            # Fit a plane using three points
            normal, plane_point = fit_plane(point1, point2, waypoints[i + 2][0])

        plane_normal = normalize(normal)
        u_axis = normalize(np.cross(plane_normal, [0, 0, 1]) if np.abs(plane_normal[2]) < 0.9 else np.cross(plane_normal, [1, 0, 0]))
        v_axis = np.cross(plane_normal, u_axis)

        # Transform points to 2D plane coordinates
        start_2d = (*transform_to_plane_coordinates(point1, normal, plane_point), transform_heading_to_plane(theta1, normal, u_axis))
        end_2d = (*transform_to_plane_coordinates(point2, normal, plane_point), transform_heading_to_plane(theta2, normal, u_axis))

        # Compute the 2D Dubins path
        # path_2d, _ = dubins.shortest_path(start_2d, end_2d, r_min).sample_many(sample_dist)

        path_2d, time_intervals = sample_dubins_path(dubins.shortest_path(start_2d, end_2d, r_min))
        
        # Convert back to 3D in the plane's coordinate system
        path_segment = []
        for u, v, _ in path_2d:
            point_3d = plane_point + float(u) * np.array(u_axis) + v * v_axis
            path_segment.append(tuple(point_3d))

        path_points.extend(path_segment)
        all_time_intervals.extend(time_intervals)

    return path_points, all_time_intervals

def setup_waypoints(x_initial:Point, waypoints:Points) -> DubinsPoints:
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
    waypoints_with_dubins = [[(p_initial[0], p_initial[1], p_initial[2]), initial_heading]]
    
    # Add waypoints with propagated headings
    for i in range(1, len(waypoints)):
        # Calculate heading to next waypoint
        if i < len(waypoints) - 1:
            dx = waypoints[i+1][0] - waypoints[i][0]
            dy = waypoints[i+1][1] - waypoints[i][1]
            heading = np.arctan2(dy, dx)
        else:
            # For last waypoint, keep the heading from previous segment
            heading = waypoints_with_dubins[-1][1]
            
        waypoints_with_dubins.append((
            ([waypoints[i][0],
            waypoints[i][1], 
            waypoints[i][2]],
            heading)
        ))
    
    return waypoints_with_dubins

def visualize_3d_dubins_path(waypoints:Points, trajectory, orientations:Optional[np.ndarray]=None):
    """
    Visualizes the 3D Dubins-like path and waypoints.
    
    :param waypoints: List of waypoints [(x, y, z, theta)].
    :param trajectory: List of (x, y, z) points representing the 3D Dubins path.
    :param orientations: List of orientation vectors to display as quivers
    """
    fig = plt.figure(figsize=(12, 8))
    ax = cast(Axes3D, fig.add_subplot(111, projection='3d'))  # cast to Axes3D


    # Plot the waypoints
    waypoints_x = [p[0] for p, heading in waypoints]
    waypoints_y = [p[1] for p, heading in waypoints]
    waypoints_z = [-p[2] for p, heading in waypoints]

    ax.scatter(waypoints_x, waypoints_y, waypoints_z, color='red', label='Waypoints', s=50) # type: ignore

    # Plot the trajectory
    traj_x = [p[0] for p in trajectory]
    traj_y = [p[1] for p in trajectory]
    traj_z = [-p[2] for p in trajectory]
    ax.plot(traj_x, traj_y, traj_z, color='blue', label='3D Dubins Path', linewidth=2)
    if orientations is not None:
        x_axes = [R.from_quat(orientation).apply([1,0,0]) for orientation in orientations]
        y_axes = [R.from_quat(orientation).apply([0,1,0]) for orientation in orientations]
        z_axes = [R.from_quat(orientation).apply([0,0,1]) for orientation in orientations]
        sample_indices = np.linspace(1, len(trajectory)-2, 20, dtype=int)
        ax.quiver(
            [trajectory[i][0] for i in sample_indices],
            [trajectory[i][1] for i in sample_indices], 
            [-trajectory[i][2] for i in sample_indices],
            [x_axes[i-1][0] for i in sample_indices],
            [x_axes[i-1][1] for i in sample_indices],
            [-x_axes[i-1][2] for i in sample_indices],
            color='green', length=1, normalize=True
        )

    # Customize the plot
    ax.set_title('3D Dubins-like Path Visualization')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    if hasattr(ax, 'set_zlabel'):
        ax.set_zlabel('Z')
    ax.legend()
    ax.grid(True)
    plt.show(block = True)

def compute_angular_velocity(quaternions:list[R], time_intervals:list[float]) -> list[Vector]:
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

        # Rotation vector: axis * angle (in radians)
        rotvec = delta_q.as_rotvec()

        # Divide by time interval to get angular velocity
        delta_t = time_intervals[i - 1]
        omega = rotvec / delta_t

        angular_velocities.append(omega)

    return angular_velocities


def compute_roll_angle(curvature:float, speed:float, g:float=9.81) -> float:
    """Compute the roll angle for a coordinated turn.
    
    Args:
        curvature: Curvature of the path at the current point (1/m).
        speed: Forward velocity of the drone (m/s).
        g: Gravitational acceleration (m/s^2).
    
    Returns:
        Roll angle in radians.
    """
    return np.arctan(curvature * speed**2 / g)

def apply_roll_to_quaternion(base_rotation:R, velocity_vector:Vector, roll_angle:float):
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
    new_orientation = roll_rotation * base_rotation
    return new_orientation

def get_velocity_directions(path_points:Points):
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
    """
    TODO:
    Initialises a waypoint traversal trajectory with the dubins shortest path.

    implements the initialise method which returns a trajectory "guess" containing
    guesses for state and control as well as (optionally) the waypoint variables lambda, mu and nu.

    Control initialisation assumes bang-bang type control for turns and neutral controls otherwise.
    
    """
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
        rotations = []
        for vel_dir in vel_directions:
            rot = R.align_vectors([vel_dir], [[1, 0, 0]])[0]
            rotations.append(rot)
        rotations = R.from_quat([r.as_quat() for r in rotations])

        velocities = np.array(vel_directions) * np.linalg.norm(trajectory.waypoints.initial_state[3:6])
        roll_angles = [0]
        new_orientations = [R.from_quat(trajectory.waypoints.initial_state[6:10])]
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

        new_orientations.append(new_orientations[-1]) # assume we don't need to change orientation for last waypoint

        print(len(time_intervals))
        self.time_intervals = time_intervals
        print(len(new_orientations))
        angular_velocities = compute_angular_velocity(new_orientations, time_intervals)


        # print("Angular Velocities: ", angular_velocities)
        # print("Orientations: ", new_orientations)
        self.orientations = [orientation.as_quat() for orientation in new_orientations]

        print("final", len(self.orientations))
        print("final", len(self.dubins_path))
        # compute orientations, velocities and angular velocities
        # convert to arrays
        orientations = np.array(self.orientations)
        # velocities = np.array(velocities)
        angular_velocities = np.zeros_like(velocities)
        positions = np.array(self.dubins_path)
        print(positions.shape)
        print(velocities.shape)
        print(orientations.shape)
        print(angular_velocities.shape)
        self.guess = np.concatenate((positions, velocities, orientations, angular_velocities), axis = 1)
        print(self.guess.shape)

    def waypoint_variable_guess(self):

        num_waypoints = self.num_waypoints

        lambda_guess = np.zeros((num_waypoints, self.num_nodes + 1))
        mu_guess = np.zeros((num_waypoints, self.num_nodes))
        nu_guess = np.zeros((num_waypoints, self.num_nodes))

        i_wp = 0
        for i in range(1, self.num_nodes):
            if i > self.switch_var[i_wp]:
                i_wp += 1

            if ((i_wp == 0) and (i + 1 >= self.switch_var[0])) or i + 1 - self.switch_var[i_wp-1] >= self.switch_var[i_wp]:
                mu_guess[i_wp, i] = 1

            for j in range(num_waypoints):
                if i + 1 >= self.switch_var[j]:
                    lambda_guess[j, i] = 1

        return (lambda_guess, mu_guess, nu_guess)

    # def trajectory(self, s):
    #     """
    #     Smooth trajectory interpolated from dubins path
    #     """
    #     s_values = np.linspace(0, 1, num=len(self.dubins_path))
    #     x_values = [i[0] for i in self.dubins_path]
    #     y_values = [i[1] for i in self.dubins_path]
    #     z_values = [i[2] for i in self.dubins_path]
    #     # Fit cubic splines for each coordinate
    #     Px = CubicSpline(s_values, x_values, bc_type='clamped')
    #     Py = CubicSpline(s_values, y_values, bc_type='clamped')
    #     Pz = CubicSpline(s_values, z_values, bc_type='clamped')

    #     return np.array([Px(s), Py(s), Pz(s)])
    
    def linear_interp(self, s, s_vals, y_vals):
        """Returns a CasADi SX symbolic expression for piecewise linear interpolation."""
        import casadi as ca
        
        expr = 0
        for i in range(len(s_vals) - 1):
            # Use CasADi's logical operators instead of Python's and/or
            condition = ca.logic_and(s >= s_vals[i], s <= s_vals[i+1])
            
            # Calculate the weight when the condition is true
            weight = (s_vals[i+1] - s) / (s_vals[i+1] - s_vals[i])
            
            # Use if_else to apply the weight only when the condition is true
            w = ca.if_else(condition, weight, 0)
            
            # Add the weighted contribution to the expression
            expr += w * y_vals[i] + (1 - w) * y_vals[i+1]
        
        return expr

    def length(self, N=100):
        import casadi as ca

        s = ca.MX.sym("s")
        pos = self.trajectory(s)
        dpos_ds = ca.jacobian(pos, s)  # ∂trajectory/∂s
        speed = ca.norm_2(dpos_ds)

        # Make sure to use SX or MX version of s depending on eval context
        integrand = ca.Function('integrand', [s], [speed])

        s_grid = np.linspace(0, 1, N)
        ds = 1 / (N - 1)
        length = 0.0

        for i in range(N - 1):
            si = float(s_grid[i])
            si1 = float(s_grid[i + 1])
            # Evaluate with .evalf()
            vi = integrand(si).full().item()
            vi1 = integrand(si1).full().item()
            length += 0.5 * ds * (vi + vi1)

        return length




    def sx_linear_interpolator(self, s_vals, y_vals):
        import casadi as ca
        import numpy as np
        assert len(s_vals) == len(y_vals)
        n = len(s_vals)

        def interp(s):
            expr = 0
            for i in range(n - 1):
                s0, s1 = s_vals[i], s_vals[i + 1]
                y0, y1 = y_vals[i], y_vals[i + 1]
                w = (s - s0) / (s1 - s0)
                cond = ca.logic_and(s >= s0, s <= s1)
                expr += ca.if_else(cond, y0 * (1 - w) + y1 * w, 0)
            # Extrapolation (optional)
            expr += ca.if_else(s < s_vals[0], y_vals[0], 0)
            expr += ca.if_else(s > s_vals[-1], y_vals[-1], 0)
            return expr

        return interp

    @property
    def trajectory(self):
        import casadi as ca
        s = ca.MX.sym("s")
        s_values = np.linspace(0, 1, num=len(self.dubins_path))
        x_values = [i[0] for i in self.dubins_path]
        y_values = [i[1] for i in self.dubins_path]
        z_values = [i[2] for i in self.dubins_path]

        # Px = ca.interpolant('Px', 'bspline', [s_values], x_values)
        # Py = ca.interpolant('Py', 'bspline', [s_values], y_values)
        # Pz = ca.interpolant('Pz', 'bspline', [s_values], z_values)

        interp_x = self.sx_linear_interpolator(s_values, x_values)
        interp_y = self.sx_linear_interpolator(s_values, y_values)
        interp_z = self.sx_linear_interpolator(s_values, z_values)

        traj = ca.vertcat(interp_x(s), interp_y(s), interp_z(s))

        return ca.Function('traj', [s], [traj])

    # def trajectory(self, s):
    #     import casadi as ca
        
    #     s_values = np.linspace(0, 1, num=len(self.dubins_path))
    #     x_values = [i[0] for i in self.dubins_path]
    #     y_values = [i[1] for i in self.dubins_path]
    #     z_values = [i[2] for i in self.dubins_path]
    #     # # Fit cubic splines for each coordinate
    #     # Px = ca.interpolant('Px', 'bspline', [s_values], x_values)
    #     # Py = ca.interpolant('Py', 'bspline', [s_values], y_values)
    #     # Pz = ca.interpolant('Pz', 'bspline', [s_values], z_values)

    #     return ca.vertcat(
    #             self.linear_interp(s, s_values, x_values),
    #             self.linear_interp(s, s_values, y_values),
    #             self.linear_interp(s, s_values, z_values)
    #         )

    def visualise(self):
        # Visualize the generated 3D Dubins path
        visualize_3d_dubins_path(self.dubins_waypoints, self.dubins_path, orientations = self.orientations)

    def state_guess(self, trajectory:TrajectoryConfiguration):
        """
        Initial guess for the state variables.
        """
        state_dim = self.aircraft.num_states
        initial_pos = trajectory.waypoints.initial_position
        initial_orientation = trajectory.waypoints.initial_state[6:10]
        velocity_guess = trajectory.waypoints.default_velocity
        waypoints = self.waypoints[1:, :]
        
        x_guess = np.zeros((state_dim, self.num_nodes + 1))
        distance = self.distances
    
        self.r_glide = 10
        
        direction_guess = (waypoints[0, :] - initial_pos)
        vel_guess = velocity_guess *  direction_guess / np.linalg.norm(direction_guess)

        if self.VERBOSE:
            print("Cumulative Waypoint Distances: ", distance)
            print("Predicted Switching Nodes: ", self.switch_var)
            print("Direction Guess: ", direction_guess)
            print("Velocity Guess: ", vel_guess)
            print("Initial Position: ", initial_pos)
            print("Waypoints: ", waypoints)

        x_guess[:3, 0] = initial_pos
        x_guess[3:6, 0] = vel_guess


        rotation, _ = R.align_vectors(np.array(direction_guess).reshape(1, -1), [[1, 0, 0]])

        # Check if the aircraft is moving in the opposite direction
        if np.dot(direction_guess.T, [1, 0, 0]) < 0:
            flip_y = R.from_euler('y', 180, degrees=True)
            rotation = rotation * flip_y

        # Get the euler angles
        euler = rotation.as_euler('xyz')
        print("Euler: ", euler)
        # If roll is close to 180, apply correction
        # if abs(euler[0]) >= np.pi/2: 
            # Create rotation around x-axis by 180 degrees
        roll_correction = R.from_euler('x', 180, degrees=True)
        
        x_guess[6:10, 0] = (rotation).as_quat()

        # z_flip = R.from_euler('x', 180, degrees=True)

        for i, waypoint in enumerate(waypoints):
            if len(self.trajectory.waypoints.waypoint_indices) < 3:
                    waypoint[2] = initial_pos[2] + self.distances[i] / self.r_glide
        i_wp = 0
        for i in range(self.num_nodes):
            # switch condition
            if i > self.switch_var[i_wp]:
                i_wp += 1
                
            if i_wp == 0:
                wp_last = initial_pos
            else:
                wp_last = waypoints[i_wp-1, :]
            wp_next = waypoints[i_wp, :]

            if i_wp > 0:
                interpolation = (i - self.switch_var[i_wp-1]) / (self.switch_var[i_wp] - self.switch_var[i_wp-1])
            else:
                interpolation = i / self.switch_var[0]

            

            # extend position guess
            pos_guess = (1 - interpolation) * wp_last + interpolation * wp_next

            x_guess[:3, i + 1] = np.reshape(pos_guess, (3,))
            

            direction = (wp_next - wp_last) / ca.norm_2(wp_next - wp_last)
            vel_guess = velocity_guess * direction
            x_guess[3:6, i + 1] = np.reshape(velocity_guess * direction, (3,))

            rotation, _ = R.align_vectors(np.array(direction).reshape(1, -1), [[1, 0, 0]])

            # Check if the aircraft is moving in the opposite direction
            if np.dot(direction.T, [1, 0, 0]) < 0:
                flip_y = R.from_euler('y', 180, degrees=True)
                rotation = rotation * flip_y

            # Get the euler angles
            euler = rotation.as_euler('xyz')
            # print("Euler: ", euler)
            # If roll is close to 180, apply correction
            # if abs(euler[0]) >= np.pi/2: 
                # Create rotation around x-axis by 180 degrees
            # roll_correction = R.from_euler('x', 180, degrees=True)
                # Apply correction
            # rotation = rotation * roll_correction


            x_guess[6:10, i + 1] = (rotation).as_quat()

        # x_guess = self.smooth_trajectory(x_guess)

        time_guess = distance[-1] / velocity_guess
        #
        
        
        return x_guess, time_guess
    
    def render(self):
        from aircraft.plotting.plotting import TrajectoryData, TrajectoryPlotter
        if self.VERBOSE:
            print("State Guess: ", x_guess)
            plotter = TrajectoryPlotter(self.aircraft)
            trajectory_data = TrajectoryData(
                state = np.array(x_guess),
                # time = np.array(time_guess)
            )
            
            plotter.plot(trajectory_data = trajectory_data)
            plt.pause(0.001)
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection = '3d')
            # ax.plot(x_guess[4, :], x_guess[5, :], x_guess[6, :])
            
            plt.show(block = True)
    

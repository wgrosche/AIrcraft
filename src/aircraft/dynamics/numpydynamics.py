import numpy as np
from abc import ABC, abstractmethod
from scipy.spatial.transform import Rotation as R


class SixDOF(ABC):
    """
    Implements reference frames and transformations in a 6DOF quaternion formulation.
    
    State:
    - Position (x, y, z) in the inertial frame
    - Velocity (vx, vy, vz) in the inertial frame
    - Attitude (quaternion w, x, y, z) representing rotation from body to inertial frame
    - Angular velocity (p, q, r) in the body frame
    
    Aerodynamic Quantities:
    - Airspeed: Magnitude of body-relative velocity
    - Alpha: Angle of attack
    - Beta: Sideslip angle
    - q_bar: Dynamic pressure
    """
    def __init__(self, rho=1.225, omega_earth=np.array([7.2921150e-5, 0, 0]), gravity = np.array([0, 0, -9.81]), integrator=None, mass = 1.0):
        self._state = np.zeros(13)
        self._state[6] = 1.0  # Default quaternion: [1, 0, 0, 0]
        self.rho = rho
        self._cached_euler = None  # Cache for Euler angles
        self.omega_earth = omega_earth
        self.gravity = gravity
        self.mass = mass

    @property
    def state(self):
        return self._state
    
    @property
    def inertia(self):
        return np.diag([1, 1, 1])
    
    @property
    @abstractmethod
    def controls(self):
        ...

    @property
    def inertial_position(self):
        return self._state[:3]
    
    @inertial_position.setter
    def inertial_position(self, value):
        if len(value) != 3:
            raise ValueError("Position must be a 3D vector.")
        self._state[:3] = value

    @property
    def inertial_velocity(self):
        return self._state[3:6]
    
    @inertial_velocity.setter
    def inertial_velocity(self, value):
        if len(value) != 3:
            raise ValueError("Velocity must be a 3D vector.")
        self._state[3:6] = value

    @property
    def inertial_attitude(self):
        quat = self._state[6:10]
        return quat / np.linalg.norm(quat)
    
    @inertial_attitude.setter
    def inertial_attitude(self, value):
        if len(value) != 4:
            raise ValueError("Attitude must be a 4D quaternion.")
        self._state[6:10] = value / np.linalg.norm(value)

    @property
    def body_angular_velocity(self):
        return self._state[10:13]

    def from_inertial_to_body(self, vector):
        return R.from_quat(self.inertial_attitude).apply(vector)
    
    def from_body_to_inertial(self, vector):
        return R.from_quat(self.inertial_attitude).inv().apply(vector)
    
    @property
    def body_velocity(self):
        return self.from_inertial_to_body(self.inertial_velocity)
    
    @property
    def airspeed(self):
        return np.linalg.norm(self.body_velocity) + 1e-8  # Prevent division by zero

    @property
    def alpha(self):
        return np.arctan2(self.body_velocity[2], self.body_velocity[0])

    @property
    def beta(self):
        return np.arcsin(self.body_velocity[1] / self.airspeed)
    
    @property
    def q_bar(self):
        return 0.5 * self.rho * self.airspeed**2

    def _compute_euler_angles(self):
        """Compute Euler angles from the quaternion and cache them."""
        w, x, y, z = self.inertial_attitude
        phi = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
        theta = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))  # Clip for safety
        psi = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        self._cached_euler = (phi, theta, psi)

    @property
    def phi(self):
        if self._cached_euler is None:
            self._compute_euler_angles()
        assert isinstance(self._cached_euler, tuple)
        return self._cached_euler[0]
    
    @property
    def theta(self):
        if self._cached_euler is None:
            self._compute_euler_angles()
        assert isinstance(self._cached_euler, tuple)
        return self._cached_euler[1]
    
    @property
    def psi(self):
        if self._cached_euler is None:
            self._compute_euler_angles()
        assert isinstance(self._cached_euler, tuple)
        return self._cached_euler[2]
    
    @property
    def phi_dot(self):
        p, q, r = self.body_angular_velocity
        return p + np.sin(self.phi) * np.tan(self.theta) * q + np.cos(self.phi) * np.tan(self.theta) * r
    
    @property
    def theta_dot(self):
        _, q, r = self.body_angular_velocity
        return q * np.cos(self.phi) - r * np.sin(self.phi)
    
    @property
    def psi_dot(self):
        _, q, r = self.body_angular_velocity
        return (q * np.sin(self.phi) + r * np.cos(self.phi)) / np.cos(self.theta)

    @property
    @abstractmethod
    def body_forces(self):
        pass
    
    @property
    @abstractmethod
    def body_moments(self):
        pass
    
    @property
    def inertial_forces(self):
        """
        Computes the inertial forces from the body forces. Adds contributions due to gravity,
        centrifugal force, and Coriolis force due to Earth's rotation.
        """
        # Transform body forces to the inertial frame
        inertial_forces = self.from_body_to_inertial(self.body_forces)
        
        # Add gravitational force (assumes a constant gravity vector in the inertial frame)
        inertial_forces += self.gravity

        # Add centrifugal force due to Earth's rotation
        r = self.inertial_position  # Position vector in the inertial frame (ECEF origin)
        omega = np.array([0, 0, self.omega_earth])  # Earth's angular velocity vector (rad/s)
        centrifugal_force = np.cross(omega, np.cross(omega, r))
        inertial_forces += centrifugal_force

        # Add Coriolis force due to Earth's rotation
        v_inertial = self.inertial_velocity  # Velocity in the inertial frame
        coriolis_force = 2 * np.cross(omega, v_inertial)
        inertial_forces += coriolis_force

        return inertial_forces
    
    @property
    def inertial_moments(self):
        """
        Computes the inertial moments from the body moments. Adds contributions due to
        fictitious forces (Coriolis and centrifugal effects) and transforms the moments
        to the inertial frame.
        """
        # Transform body moments to the inertial frame
        inertial_moments = self.from_body_to_inertial(self.body_moments)

        # Compute additional contributions
        omega = np.array([0, 0, self.omega_earth])  # Earth's angular velocity (rad/s)
        angular_velocity_body = self.body_angular_velocity
        angular_velocity_inertial = self.from_body_to_inertial(angular_velocity_body)

        # Compute fictitious moment contributions
        # Coriolis moment: cross product of angular velocity and inertial moments of inertia
        coriolis_moment = 2 * np.cross(omega, angular_velocity_inertial)

        # Centrifugal effects (if applicable, add any additional terms based on your specific setup)
        centrifugal_moment = np.cross(omega, np.cross(omega, angular_velocity_inertial))

        # Add contributions to the inertial moments
        inertial_moments += coriolis_moment
        inertial_moments += centrifugal_moment

        return inertial_moments

    
    @property
    def inertial_position_dot(self):
        return self.inertial_velocity
    
    @property
    def inertial_velocity_dot(self):
        return self.inertial_forces / self.mass
    
    @property
    def inertial_attitude_dot(self):
        """
        Compute the time evolution of the quaternion given angular velocity.
        """
        w, x, y, z = self.inertial_attitude
        p, q, r = self.body_angular_velocity

        omega_matrix = 0.5 * np.array([
            [0, -p, -q, -r],
            [p,  0,  r, -q],
            [q, -r,  0,  p],
            [r,  q, -p,  0],
        ])
        quat_dot = omega_matrix @ np.array([w, x, y, z])
        return quat_dot
    
    @property
    def body_angular_velocity_dot(self):
        """
        Compute the time evolution of the angular velocity given the body forces and moments.
        """
        I = self.inertia

        return np.linalg.inv(I) @ (self.body_moments - np.cross(self.body_angular_velocity, I @ self.body_angular_velocity))
    
    @property
    def state_dot(self):
        """
        Compute the time derivative of the state vector.
        """
        return np.concatenate([
            self.inertial_position_dot(),
            self.inertial_velocity_dot(),
            self.inertial_attitude_dot(),
            self.body_angular_velocity_dot()
        ])
    
    def step(self, dt):
        new_state = self.integrator(self.state, self.state_dot, dt)

        return new_state
    
    def update(self, dt):
        self.state = self.step(dt)



class Aircraft(SixDOF):
    def __init__(self, aerodynamic_offset=np.zeros(3)):
        super().__init__()
        self.aerodynamic_offset = aerodynamic_offset
        pass

    @property
    @abstractmethod
    def stability_forces(self):
        pass

    @property
    @abstractmethod
    def stability_moments(self):
        pass

    @property
    def body_forces(self):
        """
        Computes the inertial forces from the body forces.
        """
        return self.from_stability_to_body(self.stability_forces)
    
    @property
    def body_moments(self):
        """
        Computes the inertial moments from the body moments and adds the contribution
        due to the body forces.
        """
        body_forces = self.body_forces()
        body_moments = self.from_stability_to_body(self.stability_moments)

        body_moments = body_moments + np.cross(self.aerodynamic_offset, body_forces)

        return body_moments
    
    # Stability Frame Transformations
    def from_body_to_stability(self, vector):
        """
        Transforms a vector from the body frame to the stability frame.
        In the stability frame:
        - x-axis aligns with relative airflow.
        - z-axis is vertical with respect to airflow.
        """
        alpha = self.alpha
        rotation = R.from_euler('y', -alpha)
        return rotation.apply(vector)
    
    def from_stability_to_body(self, vector):
        """Transform a vector from the stability frame to the body frame."""
        alpha = self.alpha
        rotation = R.from_euler('y', alpha)
        return rotation.apply(vector)
    

    # Reference Datum Frame
    def from_body_to_rdf(self, vector):
        """
        Transforms a vector from the body frame to the reference datum frame.
        Example: Offsets for aerodynamic centers, control surfaces.
        """
        # Example: Offset by [dx, dy, dz] in the body frame.
        offset = np.array([0.1, 0.2, -0.1])  # Example offset
        return vector + offset
    

        # def from_body_to_surface(self, vector):
    #     """
    #     Transforms a vector from the body frame to the surface frame.
    #     """
    #     # Example: Rotate by 45 degrees around the x-axis.
    #     rotation = R.from_euler('x', 45, degrees=True)
    #     return rotation.apply(vector)
    
    # def _ecef_to_surface_matrix(self, lat, lon):
    #     """
    #     Compute the rotation matrix from ECEF to the surface (NED) frame.
    #     :param lat: Geodetic latitude (radians)
    #     :param lon: Longitude (radians)
    #     :return: 3x3 rotation matrix (ECEF -> Surface)
    #     """
    #     sin_lat = np.sin(lat)
    #     cos_lat = np.cos(lat)
    #     sin_lon = np.sin(lon)
    #     cos_lon = np.cos(lon)

    #     # Rows of the rotation matrix (ECEF -> NED)
    #     r_north = [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat]
    #     r_east = [-sin_lon, cos_lon, 0]
    #     r_down = [-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat]

    #     return np.array([r_north, r_east, r_down])
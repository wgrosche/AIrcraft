# State

## Coordinate Systems

We use 4 coordinate systems to describe our aircraft dynamics. Each coordinate
system is right-handed.

### Inertial Frame (NED)

A non-accelerating, non-rotating reference frame, it uses the axis convention
NED (North, East, Down) and is related to the earth frame by an angular velocity
vector $\vec{\omega} _{i\rightarrow e}$.

### Earth Frame (ECEF)

The rotation of the earth with respect to the inertial frame is mostly neglected
here though the functionality to include it exists. To introduce a rotating earth
simply change the value of $\vec{\omega} _{i\rightarrow e}$ within the control
variables.

This frame also considers NED its axis convention.

### Aircraft Frame (NED)

The aircraft frame considers the centre of mass of the aircraft as the origin,
the x-axis (north) runs from the tail of the aircraft to the tip (nose positive),
the z-axis (down) lies within the plane of symmetry of the aircraft and points
in opposition to the lift force, perpendicular to the x-axis. The y-axis (east)
form a right handed system with the x and z-axes and roughly aligns with the
right wing of the aircraft.

### Wind Frame (SEU)

The wind frame also has its origin at the centre of mass of the aircraft but is
aligned with the relative airspeed vector instead. It has the components Drag (x-axis),
Sideforce (y-axis) and Lift (z-axis)

## State Vector

The state of the aircraft is characterised by:

- Orientation: (conversion from inertial to body frame) (Quaternion(x, y, z, w))
- Position: in the inertial frame (Vector(x, y, z))
- Velocity: in the inertial frame (Vector(u, v, w))
- Angular Velocity (change in orientation) Vector($\omega _x$, $\omega _y$, $\omega _z$)

[return to main](../README.md)
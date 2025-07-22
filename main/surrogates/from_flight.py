# %%

import pandas as pd
import numpy as np
from aircraft.config import DATAPATH
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid as cumtrapz
from mpl_toolkits.mplot3d import Axes3D
# %%
datapath = Path(DATAPATH) / "raw"/ "flight_data" / 'flight.csv'#"example_flight_data.csv"
data = pd.DataFrame(pd.read_csv(datapath))

print(data.keys())
# # %%
# # ax, ay, az = data['X [g]'], data['Y [g]'], data['Z [g]']
# # %%
# # Parameters
# g = 9.81  # m/s²
# dt = 0.1  # time step in seconds (adjust to your sampling rate)

# # Convert accelerations from g to m/s²
# ax = data['X [g]'].to_numpy() * g
# ay = data['Y [g]'].to_numpy() * g
# az = data['Z [g]'].to_numpy() * g

# # Time array
# t = np.arange(len(ax)) * dt

# # Heading in degrees (0° = North), convert to radians
# # heading_deg = data['Flight Direction [°]'].to_numpy()
# heading_deg = data['Direction [°]'].to_numpy()
# heading_rad = np.deg2rad(heading_deg)

# ax_aligned =  np.cos(heading_rad) * ax + np.sin(heading_rad) * ay
# ay_aligned = -np.sin(heading_rad) * ax + np.cos(heading_rad) * ay
# az_aligned = az  # unchanged (assumes no roll/pitch)


# ax = ax_aligned
# ay = ay_aligned
# az = az_aligned


# # Integrate acceleration to get velocity
# vx = cumtrapz(ax, t, initial=0)
# vy = cumtrapz(ay, t, initial=0)
# vz = cumtrapz(az, t, initial=0)

# # Integrate velocity to get position
# x = cumtrapz(vx, t, initial=0)
# y = cumtrapz(vy, t, initial=0)
# z = cumtrapz(vz, t, initial=0)

# # Normalize acceleration
# a_total = np.sqrt(ax**2 + ay**2 + az**2)
# ax_norm = ax / a_total
# ay_norm = ay / a_total
# az_norm = az / a_total

# # Estimate pitch and roll (in radians)
# pitch_est = np.arcsin(-ax_norm)  # assuming aircraft pitches forward/backward in x
# roll_est  = np.arcsin(ay_norm)   # assuming lateral bank causes y-axis acceleration

# # Construct rotation matrices from pitch and roll (yaw from heading)
# # Here’s a simplified rotation for ZYX (roll, pitch, yaw)
# def rotation_matrix(yaw, pitch, roll):
#     cy, sy = np.cos(yaw), np.sin(yaw)
#     cp, sp = np.cos(pitch), np.sin(pitch)
#     cr, sr = np.cos(roll), np.sin(roll)

#     Rz = np.array([[cy, -sy, 0],
#                    [sy,  cy, 0],
#                    [0,    0, 1]])

#     Ry = np.array([[cp, 0, sp],
#                    [0,  1, 0],
#                    [-sp, 0, cp]])

#     Rx = np.array([[1, 0, 0],
#                    [0, cr, -sr],
#                    [0, sr,  cr]])

#     return Rz @ Ry @ Rx

# # === Compare Speed: GPS vs Integrated IMU ===

# # Speed from GPS (converted to m/s)
# speed_gps = data['Speed [km/h]'].to_numpy() / 3.6

# # Speed from integrated accelerations
# speed_imu = np.sqrt(vx**2 + vy**2 + vz**2)

# # Plot the speeds
# plt.figure(figsize=(10, 5))
# plt.plot(t, speed_gps, label='GPS Speed [m/s]', linewidth=2)
# plt.plot(t, speed_imu, label='IMU Speed [m/s]', linewidth=2, alpha=0.75)
# plt.xlabel("Time [s]")
# plt.ylabel("Speed [m/s]")
# plt.title("Speed: GPS vs IMU Integration")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # Optional: Compute Mean Absolute Error
# mae = np.mean(np.abs(speed_gps - speed_imu))
# print(f"Mean Absolute Speed Error: {mae:.2f} m/s")

# # Plot 3D trajectory
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(x, y, z, label='3D Trajectory')
# ax.set_xlabel('X [m]')
# ax.set_ylabel('Y [m]')
# ax.set_zlabel('Z [m]')
# ax.set_title('3D Trajectory from Acceleration Data')
# ax.legend()
# plt.tight_layout()
# plt.show()


# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import pandas as pd

# Define column names
cols = data.columns.tolist()
num_cols = len(cols)
LAT_RANGE = (0, 10)
# Container for fixed rows
fixed_rows = []

for i, row in data.iterrows():
    values = row.to_list()

    # Check if the value in the LAT column is within expected range
    lat_val = values[cols.index('LAT')]
    if not (LAT_RANGE[0] <= lat_val <= LAT_RANGE[1]):
        # Likely shift: insert NaN at LAT position and shift right
        values.insert(cols.index('LAT'), np.nan)
        values = values[:num_cols]  # truncate to original number of columns
        print(values)
    fixed_rows.append(values)

# Recreate cleaned DataFrame
data_fixed = pd.DataFrame(fixed_rows, columns=cols)
data = data_fixed
# Get data
lat = data['LON'].to_numpy()
lon = data['Flight Direction [°]'].to_numpy()
alt = data['Altitude NN [m]'].to_numpy()

# %%
print(lat)

# %%
print(lon)

# %%
# Convert to local coordinates (ENU)
R = 6371000  # Earth radius in meters
lat0 = np.deg2rad(lat[0])
lon0 = np.deg2rad(lon[0])

lat_rad = np.deg2rad(lat)
lon_rad = np.deg2rad(lon)

dx = R * (lon_rad - lon0) * np.cos(lat0)
dy = R * (lat_rad - lat0)
dz = alt

# Stack into trajectory
# pos = np.vstack((dx, dy, dz)).T

# # Compute deltas (displacement between points)
# dpos = np.diff(pos, axis=0)
# step_dist = np.linalg.norm(dpos, axis=1)

# # Compute threshold (e.g. 3 standard deviations)
# threshold = np.mean(step_dist) + 3 * np.std(step_dist)

# # Mark valid indices (keep first point)
# valid = np.ones(len(pos), dtype=bool)
# valid[1:] = step_dist < threshold

# # Apply mask
# pos_clean = pos[valid]
# dx_clean, dy_clean, dz_clean = pos_clean[:, 0], pos_clean[:, 1], pos_clean[:, 2]

# === Plot 3D trajectory ===
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(dx, dy, dz, label='Flight Path')
ax.set_xlabel('East [m]')
ax.set_ylabel('North [m]')
ax.set_zlabel('Altitude [m]')
ax.set_title('3D Flight Trajectory from GPS')
ax.legend()
plt.tight_layout()
plt.show()
# %%

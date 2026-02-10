import pandas as pd
import pyquaternion as pyq
# import math
import matplotlib.pyplot as plt
# import numpy as np
import data_sync as sync
import helper_plot as h_plt

# 1. Import a dataset
# 2. Extract some sub-sample
# A) Try to invert the Z axis readings
# ~~B) Low pass filter the accel readings~~
# 3. Initialize filter
# 4. Conver sensor readings to SI units
# 5. compute the orientation quaternions from IMU data
# 6. Align the frames of reference
# 7. Compute roll and pitch
# 8. Test plot roll and pitch

# read the sample dataset
# data = pd.read_csv("synced_data/synced_0roll_0pitch_tt_1.csv")
data = pd.read_csv("synced_data/synced_Freehand_sin_1.csv")
n = 10000
data = data.loc[:n][data.columns]

# extract acc/gyr data into n x 3 np array and filter acc data
acc, gyr, time = sync.get_IMU_data(data)
acc_filt = sync.get_filtered_acc(acc, n=12)
acc_z_inv = sync.get_inverted_Z(acc)

# compute IMU quaternions
Q_IMU = sync.get_IMU_quat(acc,gyr)
Q_IMU_inv_z = sync.get_IMU_quat(acc_z_inv,gyr)
Q_IMU_filt = sync.get_IMU_quat(acc_filt,gyr)
# Q_IMU_delta = Q_IMU - Q_IMU_filt # seems like filtering helps a bit

# extract SPA base quaternions from optitrack data
Q_OPT = data[['BR_W','BR_X','BR_Y','BR_Z']].to_numpy()

# Now rotate all optitrack quaternions to align with IMU frame
q0_imu = pyq.Quaternion(Q_IMU_inv_z[0])
q0_opt = pyq.Quaternion(Q_OPT[0])
qd = q0_opt.conjugate * q0_imu
Q_OPT_r = sync.rotate_quaternions(Q_OPT,qd)

# Extract roll and pich from IMU and optitrack
euler_opt = sync.get_euler_angles(Q_OPT)
euler_opt_r = sync.get_euler_angles(Q_OPT_r)
euler_imu = sync.get_euler_angles(Q_IMU)

d_euler = euler_imu - euler_opt_r


# h_plt.plot_2x_vertical(euler_imu,euler_opt_r,time)
h_plt.plot_3x_vertical(euler_imu,euler_opt,euler_opt_r,time)
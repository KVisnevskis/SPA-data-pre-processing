import pandas as pd
import ahrs.filters.complementary as estimator
import pyquaternion as pyq
import math
import matplotlib.pyplot as plt
import numpy as np
from skinematics import view
from skinematics import quat as sk_quat

def get_euler_from_quat(q):
    phi   = math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x**2 + q.y**2))
    theta = math.asin (2 * (q.w * q.y - q.z * q.x))
    psi   = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2))
    return [phi,theta,psi]

def rad_to_deg(euler_array):
    return euler_array * 180 / math.pi

# read the sample dataset
# data = pd.read_csv("synced_data/synced_0roll_0pitch_tt_1.csv")
data = pd.read_csv("synced_data/synced_Freehand_sin_1.csv")

# extract the first n samples
n = 10000
data = data.loc[:n][data.columns]

# extract gyroscope data into n x 3 np array
gyr = data[['gyr_x','gyr_y','gyr_z']].to_numpy()
acc = data[['acc_x','acc_y','acc_z']].to_numpy()
time = data['Time'].to_numpy()
# extract optitrack orientations
Q_OPT = data[['BR_W','BR_X','BR_Y','BR_Z']].to_numpy()

# convert to SI units (m/ss and rad/s)
acc = acc * 9.81
gyr = gyr * math.pi / 180

# create and initialize complementary filter object
filter = estimator.Complementary()
# filter.q0 = Quaternion(initial_q_np)
filter.acc = acc
filter.gyr = gyr
filter.frequency = 240
filter.gain = 0.03

# get orientations based on IMU data
Q_IMU = filter._compute_all()

# Initial orientations
opt_q0 = pyq.Quaternion(data.loc[1][['BR_W','BR_X','BR_Y','BR_Z']].to_numpy())
imu_q0 = pyq.Quaternion(Q_IMU[0])

# rotate all optitrack quaternions by initial IMU quaternion
Q_OPT_rotated = []  # rotated optitrack orientattions
Q_IMU_rotated = []
IMU_euler = []      # euler angles of global frame orientation for IMU
OPT_euler = []      # euler angles of global frame orientation for OPT

qd = opt_q0.conjugate * imu_q0
for row in Q_OPT:
    # convert vector to quaternion object
    opt_q = pyq.Quaternion(row)
    # rotate the optitrack quaternion by initial optitrack orientation
    opt_q = opt_q * qd
    # then rotate the opitrack quaternion by the initial arduino orientation
    # opt_q = imu_q0 * opt_q
    # additional rotation to align with imu frame
    # rot1 = pyq.Quaternion(axis = [0, 0, 1], degrees=90)
    # rot2 = pyq.Quaternion(axis = [1, 0, 0], degrees=0)
    # opt_q = rot1.conjugate * opt_q
    # opt_q = rot2.conjugate * opt_q

    # obtain euler angles for both
    OPT_euler.append(get_euler_from_quat(opt_q))
    Q_OPT_rotated.append([opt_q.w,opt_q.x,opt_q.y,opt_q.z])


for row in Q_IMU:
    imu_q = pyq.Quaternion(row)
    # imu_q = imu_q0.conjugate * imu_q
    # Q_IMU_rotated.append([imu_q.w,imu_q.x,imu_q.y,imu_q.z])
    IMU_euler.append(get_euler_from_quat(imu_q))

# plt.figure()
# view.orientation(sk_quat.unit_q(Q_OPT[0]))
# plt.figure()
# view.orientation(sk_quat.unit_q(Q_OPT_rotated[0]))

IMU_euler = np.array(IMU_euler)
OPT_euler = np.array(OPT_euler)

IMU_euler = rad_to_deg(IMU_euler)
OPT_euler = rad_to_deg(OPT_euler)

# # IMU_euler = unwrap_angles(IMU_euler)
# # OPT_euler = unwrap_angles(OPT_euler)

fig, (euler_imu_plot,euler_opt_plot) = plt.subplots(2)
euler_imu_plot.plot(time,IMU_euler)
euler_opt_plot.plot(time,OPT_euler)
plt.show()
# print acc and gyro data
# fig, (acc_plot, gyr_plot) = plt.subplots(2)
# acc_plot.plot(time, acc)
# gyr_plot.plot(time, gyr)
# plt.show()

# plt.figure()
# fig, (q_imu_plot,q_opt_plot) = plt.subplots(2)
# q_imu_plot.plot(time,Q_IMU)
# q_opt_plot.plot(time,Q_OPT_rotated)
# plt.show()
# q_plot.plot(time,Q_IMU)

# print("Info for Q_IMU: ")
# print(type(Q_IMU))
# print(Q_IMU.shape)

# print("Info on Q_OPT: ")
# print(type(Q_OPT))
# print(Q_OPT.shape)
# print(gyr.shape)
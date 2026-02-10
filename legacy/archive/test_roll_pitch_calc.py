import pandas as pd
import matplotlib.pyplot as plt
import math
import pyquaternion as pyq
import ahrs.filters.complementary as estimoator

# load in single dataset

data = pd.read_csv("synced_data/synced_Freehand_sin_1.csv")

# add columns for roll and pitch for arduino and optitrack
data.insert(len(data.columns),'ard_roll',None)
data.insert(len(data.columns),'ard_pitch',None)
data.insert(len(data.columns),'opt_roll',None)
data.insert(len(data.columns),'opt_pitch',None)
data.insert(len(data.columns),'opt_yaw',None)

def compute_roll_and_pitch_IMU(acc_x, acc_y, acc_z, convert_to_deg = True):
    # returns roll and pitch in radians
    roll = math.atan(acc_y/math.sqrt(acc_x*acc_x + acc_z*acc_z))
    pitch = math.atan(acc_x/math.sqrt(acc_y*acc_y + acc_z*acc_z))
    if convert_to_deg:
        roll = roll * 180 / math.pi
        pitch = pitch * 180 / math.pi
    return roll, pitch

estimoator.Complementary()
def compute_roll_and_pitch_OPT(q, convert_to_deg = True):
    phi   = math.atan2( 2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x**2 + q.y**2) )
    theta = math.asin ( 2 * (q.w * q.y - q.z * q.x) )
    psi   = math.atan2( 2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2) )

    if psi < 0:
        psi = psi + math.pi*2
    # roll = math.atan2(2.0*(q.x*q.y + q.w*q.z), q.w*q.w + q.x*q.x - q.y*q.y - q.z*q.z)
    # pitch = math.asin(-2.0*(q.x*q.z - q.w*q.y))
    # yaw = math.atan2(2.0*(q.y*q.z + q.w*q.x), q.w*q.w - q.x*q.x - q.y*q.y + q.z*q.z)

    if convert_to_deg:
        psi = psi * 180 / math.pi
        theta = theta * 180 / math.pi
        phi = phi * 180 / math.pi

    return phi, theta, psi

for index,row in data.iterrows():
    accX = row['acc_x']
    accY = row['acc_y']
    accZ = row['acc_z']

    q_base = pyq.Quaternion(row['BR_W'],row['BR_X'],row['BR_Y'],row['BR_Z'])

    IMU_roll, IMU_pitch = compute_roll_and_pitch_IMU(accX,accY,accZ)
    OPT_roll, OPT_pitch, OPT_yaw = compute_roll_and_pitch_OPT(q_base)

    # place results in dataframe
    data.at[index,'ard_roll'] = IMU_roll
    data.at[index,'ard_pitch'] = IMU_pitch
    data.at[index,'opt_roll'] = OPT_roll
    data.at[index,'opt_pitch'] = OPT_pitch
    data.at[index,'opt_yaw'] = OPT_yaw

fig, (ard, opt) = plt.subplots(2)
ard.plot(data['ard_pitch'],label='pitch')
ard.plot(data['ard_roll'],label='roll')

opt.plot(data['opt_pitch'],label='pitch')
opt.plot(data['opt_roll'],label='roll')
opt.plot(data['opt_yaw'],label='yaw')
# plt.plot(data['opt_roll'])
# plt.plot(data['opt_pitch'])
# plt.plot(data['opt_yaw'])
plt.show()
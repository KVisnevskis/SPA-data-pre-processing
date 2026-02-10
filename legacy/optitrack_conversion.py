import pandas as pd
import pyquaternion as pyq
import math

# rename rigid body variables to have meaningful names
header = ['Time','BR_X','BR_Y','BR_Z','BR_W','BP_X','BP_Y','BP_Z',
                 'TR_X','TR_Y','TR_Z','TR_W','TP_X','TP_Y','TP_Z']
# BR/BP = Base Rotation/Position
# TR/TP = Tip Rotation/Position

# columns for import
cols = [1,2,3,4,5,6,7,8,26,27,28,29,30,31,32]

def insert_cols(data):
    data.insert(len(data.columns),'phi',None)
    data.insert(len(data.columns),'theta',None)
    data.insert(len(data.columns),'psi',None)
    data.insert(len(data.columns),'dx',None)
    data.insert(len(data.columns),'dy',None)
    data.insert(len(data.columns),'dz',None)
    return data

def compute_angle(data,name):
    print("Processing file: " + name)
    frac = math.floor(len(data)/100)
    pct = 0
    # iterate through each sample, computing the difference in rotation and position
    for index,row in data.iterrows():
        # initialize orientation quaternions for base and tip
        q_base = pyq.Quaternion(row['BR_W'],row['BR_X'],row['BR_Y'],row['BR_Z'])
        q_tip = pyq.Quaternion(row['TR_W'],row['TR_X'],row['TR_Y'],row['TR_Z'])

        # compute difference of quaternions
        qd = q_base.conjugate * q_tip

        # convert to euler angles
        phi   = math.atan2( 2 * (qd.w * qd.x + qd.y * qd.z), 1 - 2 * (qd.x**2 + qd.y**2) )
        theta = math.asin ( 2 * (qd.w * qd.y - qd.z * qd.x) )
        psi   = math.atan2( 2 * (qd.w * qd.z + qd.x * qd.y), 1 - 2 * (qd.y**2 + qd.z**2) )

        # record angle difference
        data.at[index,'phi'] = phi
        data.at[index,'theta'] = theta
        data.at[index,'psi'] = psi

        # record cartesian coordinate difference
        data.at[index,'dx'] = row['BP_X'] - row['TP_X']
        data.at[index,'dy'] = row['BP_Y'] - row['TP_Y']
        data.at[index,'dz'] = row['BP_Z'] - row['TP_Z']

        if index % frac == 0:
            pct = pct + 1
            print("Processing: " + str(pct) + " pct done")
    
    return data
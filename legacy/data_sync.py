from scipy.signal import correlate
import numpy as np
import pandas as pd
import math
import ahrs.filters.complementary as estimator
import pyquaternion as pyq

enable_debug = 0

def get_simulated_acc(q_opt):
    G_VECTOR = np.array([0,0,-9.81])
    measured_g = []
    for row in q_opt:
        base_orienation = pyq.Quaternion(row)
        accel_out = base_orienation.rotate(G_VECTOR)
        measured_g.append(accel_out)

    measured_g = np.array(measured_g)
    return measured_g

def get_IMU_acc_data(data, for_syncing = False):
    acc_x = data['acc_x'].to_numpy() * 9.81
    acc_y = data['acc_y'].to_numpy() * 9.81
    acc_z = data['acc_z'].to_numpy() * 9.81
    acc = [acc_x,acc_y,acc_z]
    if for_syncing:
        acc = [acc_y,acc_x,acc_z]

    acc = np.array(acc).transpose()
    return acc

def get_opt_quat(data):
    return data[['BR_W','BR_X','BR_Y','BR_Z']].to_numpy()

def get_euler_from_quat(q):
    phi   = math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x**2 + q.y**2))
    theta = math.asin (2 * (q.w * q.y - q.z * q.x))
    psi   = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2))
    return [phi,theta,psi]

def get_euler_angles(q_in):
    angles = []
    for row in q_in:
        q = pyq.Quaternion(row)
        angles.append(get_euler_from_quat(q))
    return np.array(angles)

def get_inverted_Z(acc_in):
    acc_out = []
    for row in acc_in:
        acc_out.append([row[0],row[1],-row[2]])
    return np.array(acc_out)

def rotate_quaternions(in_quat,rot):
    out_quat = []
    for i,row in enumerate(in_quat):
        in_q = pyq.Quaternion(row)
        in_q = rot.conjugate * in_q * rot
        out_quat.append([in_q.w,in_q.x,in_q.y,in_q.z])
    return np.array(out_quat)

def get_IMU_quat(acc,gyr,gain=0.03):
    # create and initialize complementary filter object
    filter = estimator.Complementary()
    # filter.q0 = Quaternion(initial_q_np)
    filter.acc = np.array(acc)
    filter.gyr = np.array(gyr)
    filter.frequency = 240
    filter.gain = gain
    # get orientations based on IMU data
    Q_IMU = filter._compute_all()
    return Q_IMU

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ret = ret[n - 1:] / n
    # add zeros to the beginning to make it match in size
    ret = np.append(np.ones([n-1,1])*ret[0],ret)
    # np.append(np.zeros(n,1))
    return ret

def get_filtered_acc(acc_in, n = 3):
    acc_out = np.ones_like(acc_in)
    acc_out[:,0] = moving_average(acc_in[:,0],n)
    acc_out[:,1] = moving_average(acc_in[:,1],n)
    acc_out[:,2] = moving_average(acc_in[:,2],n)
    return acc_out

def get_IMU_data(data, for_syncing = False):
    gyr_x = data['gyr_x'].to_numpy() * math.pi / 180
    gyr_y = data['gyr_y'].to_numpy() * math.pi / 180
    gyr_z = data['gyr_z'].to_numpy() * math.pi / 180
    acc_x = data['acc_x'].to_numpy() * 9.81
    acc_y = data['acc_y'].to_numpy() * 9.81
    acc_z = data['acc_z'].to_numpy() * 9.81
    time = data['Time'].to_numpy()
    acc = [acc_x,acc_y,acc_z]
    gyr = [gyr_x,gyr_y,gyr_z]
    if for_syncing:
        acc = [acc_y,acc_x,acc_z]

    acc = np.array(acc).transpose()
    gyr = np.array(gyr).transpose()
    time = np.array(time)
    # acc = np.concatenate([acc_x,acc_y,acc_z],axis=0)
    # gyr = np.concatenate([gyr_x,gyr_y,gyr_z],axis=0)
    return acc, gyr, time

def get_xcorr(pressure, optitrack_variable):
    xcorr = correlate(pressure,optitrack_variable)
    max_xcorr = xcorr.max()
    sample_shift = len(optitrack_variable) - np.argmax(xcorr)
    return sample_shift, max_xcorr

def get_normalized(data,invert = False):
    if invert:
        data = -1*data

    # normalize within range [-1,1] for best xcorr
    normalized = (data - data.min())/(data.max()-data.min())
    normalized = (normalized - 0.5)*2
    return normalized

def get_normalized_optitrack_vars(optitrack_df):
    d = {'psi'      :get_normalized(optitrack_df['psi'],    False),
         'psi_inv'  :get_normalized(optitrack_df['psi'],    True ),
         'theta'    :get_normalized(optitrack_df['theta'],  False),
         'theta_inv':get_normalized(optitrack_df['theta'],  True ),
         'phi'      :get_normalized(optitrack_df['phi'],    False),
         'phi_inv'  :get_normalized(optitrack_df['phi'],    True ),
         'dx'       :get_normalized(optitrack_df['dx'],     False),
         'dx_inv'   :get_normalized(optitrack_df['dx'],     True ),
         'dy'       :get_normalized(optitrack_df['dy'],     False),
         'dy_inv'   :get_normalized(optitrack_df['dy'],     True ),
         'dz'       :get_normalized(optitrack_df['dz'],     False),
         'dz_inv'   :get_normalized(optitrack_df['dz'],     True )}
    norm_optitrack_df = pd.DataFrame(data=d)
    return norm_optitrack_df

def compute_xcorr(pressure_n,optitrack_n):
    var_names = optitrack_n.columns
    best_var = 'blank'
    max_corr = 0
    sample_shift = 0
    for var in var_names:
        if enable_debug:
            print("Testing cross correlation of pressure and " + var)
        var_sample_shift, var_xcorr = get_xcorr(pressure_n,optitrack_n[var])
        if var_xcorr > max_corr:
            best_var = var
            max_corr = var_xcorr
            sample_shift = var_sample_shift
    if enable_debug:
        print("Best correlated variable: " + best_var)
        print("Corresponding max correlation val: " + str(max_corr))
        print("Corresponding sample shift for arduino data: " + str(sample_shift))
    # sample_shift, max_xcorr = get_xcorr(pressure_n,optitrack_n['psi'])
    # print("Sample shift: " + str(sample_shift))
    # print("Max cross correlation value: " + str(max_xcorr))

    return sample_shift, best_var, max_corr

def compute_xcorr_acc(acc_m,acc_s):
    var_names = ['acc_x','acc_y','acc_z']
    best_var = 'blank'
    max_corr = 0
    sample_shift = 0
    for i,var_name in enumerate(var_names):
        var_sample_shift, var_xcorr = get_xcorr(acc_m[:,i],acc_s[:,i])
        if var_xcorr > max_corr:
            best_var = var_name
            max_corr = var_xcorr
            sample_shift = var_sample_shift
    return sample_shift, best_var, max_corr

def get_synchronised_combined_df(ard_df, opt_df):
        # fill any NaN values
    ard_df = ard_df.fillna(method = 'ffill')
    opt_df = opt_df.fillna(method = 'ffill')
        # get normalized pressure and opitrack df
    pressure_n = get_normalized(ard_df['pressure'])
    opt_vars_n = get_normalized_optitrack_vars(opt_df)
        # compute best variable for cross correlation and sample shift
    sample_shift, best_var, max_xcorr = compute_xcorr(pressure_n,opt_vars_n)
        # shift the arduino samples by the computed sample shift
    ard_df = ard_df.shift(sample_shift)
        # combine the datasets
    out = pd.concat([ard_df,opt_df],axis=1)
    # remove all NaN containing rows
    out = out.dropna(how = 'any')
    return out, best_var, sample_shift, max_xcorr

def get_synchronised_combined_freehand_df(ard_df, opt_df):
        # fill any NaN values
    ard_df = ard_df.fillna(method = 'ffill')
    opt_df = opt_df.fillna(method = 'ffill')

        # extract IMU accelerations and compute simulated
    acc_measured = get_IMU_acc_data(ard_df,True)
    opt_q = get_opt_quat(opt_df)
    acc_simulate = get_simulated_acc(opt_q)

    # compute the cross correlateions
    sample_shift, best_var, max_xcorr = compute_xcorr_acc(acc_measured,acc_simulate)
        # compute best variable for cross correlation and sample shift
    # sample_shift, best_var, max_xcorr = compute_xcorr(pressure_n,opt_vars_n)
        # shift the arduino samples by the computed sample shift
    ard_df = ard_df.shift(sample_shift)
    #     # combine the datasets
    out = pd.concat([ard_df,opt_df],axis=1)
    # # remove all NaN containing rows
    out = out.dropna(how = 'any')
    return out, best_var, sample_shift, max_xcorr

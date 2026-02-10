import pandas as pd
import math
from scipy.signal import correlate
import matplotlib.pyplot as plt
import numpy as np

def get_sample_shift(arduino,optitrack,opt_var,is_inverted):
    # use entire signal for xcorr (arduino pressure and optitrack angle)
    press_norm = get_normalized_pressure(arduino)
    var_norm = get_normalized_opt_var(optitrack,opt_var,is_inverted)

    # # normalize the sine wave part of the data
    # press_norm = (press_sin - press_sin.min())/(press_sin.max() - press_sin.min())
    # angle_norm = (angle_sin - angle_sin.min())/(angle_sin.max() - angle_sin.min())
    # # invert the angle
    # angle_norm = -angle_norm + 1

    # perform cross correlation
    xcorr = correlate(press_norm,var_norm)
    sample_shift = len(var_norm) - np.argmax(xcorr)
    print("Samples need shifted by: "+str(sample_shift))
    return sample_shift

def get_normalized_pressure(arduino):
    press = arduino['pressure']
    press_norm = (press - press.min())/(press.max() - press.min())
    return press_norm

def get_normalized_angle(optitrack, angle):
    angle = optitrack[angle]

    # normalize the sine wave part of the data
    angle_norm = (angle - angle.min())/(angle.max() - angle.min())
    # invert the angle
    angle_norm = -angle_norm + 1
    return angle_norm

def get_normalized_opt_var(optitrack, varname, is_inverted):
    var = optitrack[varname]
    var = (var - var.min())/(var.max() - var.min())
    if is_inverted:
        var = -var + 1

    return var

def find_best_corr(optitrack,arduino):
    xcorr_vars = ['phi','theta','psi','dx','dy','dz','phi','theta','psi','dx','dy','dz']
    xcorr_max = [0,0,0,0,0,0,0,0,0,0,0,0]
    is_inverted = [False,False,False,False,False,False,True,True,True,True,True,True]
    pressure_norm = get_normalized_pressure(arduino)
    for i,test_var in enumerate(xcorr_vars):
        var = get_normalized_opt_var(optitrack, test_var, is_inverted[i])
        xcorr_max[i] = correlate(pressure_norm, var).max()
    
    return is_inverted[np.argmin(xcorr_max)], xcorr_vars[np.argmax(xcorr_max)]
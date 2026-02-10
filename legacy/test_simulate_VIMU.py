import pandas as pd
import pyquaternion as pyq
import numpy as np
import data_sync as sync
import helper_plot as h_plt

data = pd.read_csv("synced_data/new_synced_converted_mFreehand_sin_1.csv")
# data = pd.read_csv("arduino_data/converted_mFreehand_sin_1.csv")
# opt_data = pd.read_csv("optitrack_data/converted_oFreehand_sin_1.csv")
# acc = sync.get_IMU_acc_data(data,True)
# q_opt = sync.get_opt_quat(opt_data)
# acc_sim = sync.get_simulated_acc(q_opt)

# q_opt = sync.get_opt_quat(data)

# h_plt.plot_2x_vertical(acc,sync.get_simulated_acc(data,q_opt),time)


h_plt.plot_2x_vertical_no_time(data['pressure'],data['phi'])
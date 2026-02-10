import os
import pandas as pd
import data_sync as sync
import matplotlib.pyplot as plt

# create list of all converted data filenames
ard_filelist = os.listdir("arduino_data/")[:-1]
opt_filelist = os.listdir("optitrack_data/")[:-1]

# extract single dataset
ard_dataset = pd.read_csv("arduino_data/" + ard_filelist[0])
opt_dataset = pd.read_csv("optitrack_data/" + opt_filelist[0])

out = sync.get_synchronised_combined_df(ard_dataset, opt_dataset)

# # forward fill any missing values in dataframe
# ard_dataset = ard_dataset.fillna(method = 'ffill')
# opt_dataset = opt_dataset.fillna(method = 'ffill')

# # get normalized pressure
# pressure_n = sync.get_normalized(ard_dataset['pressure'])

# # get new dataframe with normalized euler angles and coord deltas
# opt_data_n = sync.get_normalized_optitrack_vars(opt_dataset)

# # compute best variable for cross correlation and sample shift
# sample_shift, best_var, max_xcorr = sync.compute_xcorr(pressure_n,opt_data_n)

# # shift the arduino samples by the computed sample shift
# ard_dataset = ard_dataset.shift(sample_shift)

# # combine the datasets
# out = pd.concat([ard_dataset,opt_dataset],axis=1)

# # remove all NaN containing rows
# out = out.dropna(how = 'any')

# following code only for demonstration purposes
# recalculate the normalized value for pressure and best var
# invert_best_var = False
# if '_inv' in best_var:
#     best_var = best_var[:-4]
#     invert_best_var = True

best_var_n = sync.get_normalized(out['dz'],True)
pressure_n = sync.get_normalized(out['pressure'])

# test plot
plt.plot(pressure_n)
plt.plot(best_var_n)
plt.show()
# now compute the cross-correlation between pressure and 

# # print the columns to test if the dataframe was read correctly
# print(ard_dataset.columns)
# print(opt_dataset.columns)
# opt_dataset.head()
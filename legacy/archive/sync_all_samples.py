import os
import pandas as pd
import data_sync as sync

# create list of all converted data filenames
ard_filelist = os.listdir("arduino_data/")[:-1]
opt_filelist = os.listdir("optitrack_data/")[:-1]

run_name_list = []
best_var_list = []
sample_shift_list = []
max_xcorr_list = []

for i,file in enumerate(ard_filelist):
    
    run_name = file[11:]
    print("Processing datasets for run: " + run_name)

    # read i-th dataset
    ard_dataset = pd.read_csv("arduino_data/" + ard_filelist[i])
    opt_dataset = pd.read_csv("optitrack_data/" + opt_filelist[i])

    out, best_var, sample_shift, max_xcorr = sync.get_synchronised_combined_df(ard_dataset, opt_dataset)
    
    # save the new dataframe as csv
    out.to_csv("synced_data/" + "synced_" + run_name, index=False)

    run_name_list.append(run_name)
    best_var_list.append(best_var)
    sample_shift_list.append(sample_shift)
    max_xcorr_list.append(max_xcorr)

# record dataframe containing info of how stuff was shifted
d = {'run name'                 : run_name_list,
     'best variable'            : best_var_list,
     'sample shift'             : sample_shift_list,
     'maximum cross correlation': max_xcorr_list}
log = pd.DataFrame(data = d)
log.to_csv("sync_log.csv")
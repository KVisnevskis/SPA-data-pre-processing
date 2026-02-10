# import os
import pandas as pd
import data_sync as sync
import matplotlib.pyplot as plt

sync_log = pd.read_csv("sync_log.csv")

run_names = sync_log['run name']

for i,run_name in enumerate(run_names):
    var = sync_log.at[i,'best variable']
    run = sync_log.at[i,'run name']

    invert_flag = False
    if '_inv' in var:
        invert_flag = True
        var = var[:-4]

    # load the data
    data = pd.read_csv("synced_data/synced_"+run)

    # normalize and plot
    n_pressure = sync.get_normalized(data['pressure'])
    n_var = sync.get_normalized(data[var],invert_flag)

    plt.figure()
    plt.title(run_name)
    plt.plot(n_pressure,label='pressure')
    plt.plot(n_var,label=var)
    plt.legend(loc='upper left')
    plt.show()
    

    # if i == 0:
    #     break

    # sync_log.at
# for row in sync_log.iterrows():
#     print(row[])
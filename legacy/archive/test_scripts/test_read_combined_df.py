import pandas as pd
import matplotlib.pyplot as plt
import sync_functions as sync

data = pd.read_csv('synced_data/synced_comb_0roll_135pitch_tt_1.csv')

data = data.dropna(how = 'any')

if data.isnull().any().any():
    print("Some null values exist in df")

plt.plot(sync.get_normalized_pressure(data))
plt.plot(sync.get_normalized_angle(data,'psi'))
plt.show()

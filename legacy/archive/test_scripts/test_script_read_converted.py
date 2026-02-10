import pandas as pd
import matplotlib.pyplot as plt

# data = pd.read_csv("./arduino_data/converted_m0roll_0pitch_tt_2.csv")
# # acc_x = data
# plt.plot(data['acc_x'])
# plt.show()

data = pd.read_csv('test_processed_optitrack.csv')
plt.plot(data['dz'])
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import sync_functions as sync

show_plot = 1

# first load the data from arduino and optitrack
# arduino = pd.read_csv('test_processed_arduino_0r_0p.csv')
# optitrack = pd.read_csv('test_processed_optitrack_0r_0p.csv')
arduino = pd.read_csv('test_data/test_processed_arduino_135r_135p.csv')
optitrack = pd.read_csv('test_data/test_processed_optitrack_135r_135p.csv')

# fill all NaN values in dataframes with previous value
arduino = arduino.fillna(method='ffill')
optitrack = optitrack.fillna(method='ffill')

# find best correlatted variable
is_inverted, best_corr_var = sync.find_best_corr(optitrack,arduino)

print("Best correlating variable found: " + best_corr_var)
print("Does variable need inverted? - " + str(is_inverted))

# compute sample shift

sample_shift = sync.get_sample_shift(arduino,optitrack,best_corr_var,is_inverted)
print("Computed sample shift is: " + str(sample_shift))

arduino = arduino.shift(sample_shift)

if show_plot == 1:
    plt.plot(sync.get_normalized_pressure(arduino))
    plt.plot(sync.get_normalized_opt_var(optitrack,best_corr_var,is_inverted))
    plt.show()

# # combine dataframes
# out = pd.concat([arduino,optitrack],axis=1)
# # drop any rows containing NaN
# out = out.dropna(how = 'any')
# out.to_csv('test_combined_df.csv')

# # extract only the part where sin wave is present
# press_sin = arduino['pressure'][0:12000]
# angle_sin = optitrack['phi'][0:12000]


# # normalize the data to range [0,1]
# pressure_norm = (arduino['pressure']-arduino['pressure'].min())/(arduino['pressure'].max() - arduino['pressure'].min())
# phi_norm = -(optitrack['phi']-optitrack['phi'].min())/(optitrack['phi'].max()-optitrack['phi'].min()) + 1

# # subtract the mean
# pressure_norm = pressure_norm - np.mean(pressure_norm)
# phi_norm = phi_norm - np.mean(phi_norm)


# print("Sample shift: %d") sample_shift

# print(np.argwhere(np.isnan(phi)))
# see if any values are NaN
# for value in pressure:
#     if math.isnan(value):
#         print("NaN value in pressure detected")

# for value in phi:
#     if math.isnan(value):
#         print("NaN value in phi detected at %d") %index

# normalize extracted values
# pressure_max = np.nanmax(pressure)
# pressure_min = np.nanmin(pressure)
# phi_max = np.nanmax(phi)
# phi_min = np.nanmin(phi)

# pressure_n = (pressure - pressure_min)/(pressure_max - pressure_min)
# phi_n = -(phi - phi_min)/(phi_max-phi_min)+1

# # recalculate normalized pressure after shifting
# press_norm = (arduino['pressure']-arduino['pressure'].min())/(arduino['pressure'].max() - arduino['pressure'].min())

# shift all samples by the determined sample shift
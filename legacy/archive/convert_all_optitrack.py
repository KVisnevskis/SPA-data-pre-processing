import optitrack_conversion as o_convert
import pandas as pd
import os

from optitrack_conversion import cols, header

raw_dir = "optitrack_data/raw"
out_dir = "optitrack_data/"
opt_filelist = os.listdir(raw_dir)

for file in opt_filelist:
    path = raw_dir + "/" + file

    # read file
    data = pd.read_csv(path, skiprows = 6, usecols = cols)
    data = data.set_axis(header, axis=1)

    # insert new columns to hold bending angle and coord difference info
    data = o_convert.insert_cols(data)
    data = o_convert.compute_angle(data,file)

    # save the dataframe as new test output
    data.to_csv(out_dir + "converted_" + file, index=False)
    # break
    # print(path)
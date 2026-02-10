import optitrack_conversion as o_convert
import pandas as pd

# rename rigid body variables to have meaningful names
header = ['Time','BR_X','BR_Y','BR_Z','BR_W','BP_X','BP_Y','BP_Z',
                 'TR_X','TR_Y','TR_Z','TR_W','TP_X','TP_Y','TP_Z']
# BR/BP = Base Rotation/Position
# TR/TP = Tip Rotation/Position

# read raw optitrack data
cols = [1,2,3,4,5,6,7,8,26,27,28,29,30,31,32] # select columns to be used
data = pd.read_csv('o135roll_135pitch_tt_2.csv',skiprows = 6, usecols = cols)
data = data.set_axis(header, axis=1)

# insert new columns to hold bending angle and coord difference info in dataframe
data = o_convert.insert_cols(data)
data = o_convert.compute_angle(data,'test_dataset')

# save the dataframe as new test output
data.to_csv('test_processed_optitrack_135r_135p.csv',index=False)

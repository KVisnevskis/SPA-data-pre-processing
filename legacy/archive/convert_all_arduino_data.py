import os
import arduino_conversion

# list all files in directory
dir = './arduino_data/'
file_list = os.listdir(dir)
# print(file_list)

# remove all files that do not need to be converted ( do not start with 'm')
for file in file_list:
    if file.find('m') != 0:
        file_list.remove(file)

for file in file_list:
    converted = dir +'converted_' + file
    arduino_conversion.convert_arduino_dataset(dir + file,converted)

# test function
# for file in file_list:
#     print("File: " + file)
    # new_name = 'converted_' + file
    # print("Will be saved as: " + new_name)
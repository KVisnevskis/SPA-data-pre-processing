import csv

header = ["acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z","flex","pressure"]

def toSigned16(n):
    n = n & 0xffff
    return n | (-(n & 0x8000))

def convert_sensor_data(input_row):
    sample = []
    for i in input_row:
        sample.append(int(i))
    # sample = bytearray(sample)
    # ax = int.from_bytes(sample[0:2],'little')/16384
    ax = toSigned16(sample[0] | sample[1] << 8) / 16384
    ay = toSigned16(sample[2] | sample[3] << 8)/16384
    az = toSigned16(sample[4] | sample[5] << 8)/16384
    gx = toSigned16(sample[6] | sample[7] << 8)/131
    gy = toSigned16(sample[8] | sample[9] << 8)/131
    gz = toSigned16(sample[10] | sample[11] << 8)/131
    p = (sample[12] | sample[13] << 8) >> 4
    f = (sample[14] | sample[15] << 8) >> 4
    out = [ax,ay,az,gx,gy,gz,f,p]
    return out

def convert_arduino_dataset(input_file, output_file):
    # First open the matlab data and add header to csv
    with open(output_file, 'w', newline='') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow(header)
        # read every line of the input   
        with open(input_file, 'r', newline='') as input:
            reader = csv.reader(input)
            for row in reader:
                converted = convert_sensor_data(row)
                writer.writerow(converted)

# test functionality
# convert_arduino_dataset('m0roll_0pitch_tt_1.csv','test_output.csv')
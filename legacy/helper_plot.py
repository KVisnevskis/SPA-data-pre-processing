import matplotlib.pyplot as plt

def plot_2x_vertical(data1,data2,time):
    fig, (data1_plot,data2_plot) = plt.subplots(2)
    # acc_plot.plot(time,acc[0],label='Acc X')
    # acc_plot.plot(time,acc[1],label='Acc Y')
    # acc_plot.plot(time,acc[2],label='Acc Z')
    # gyr_plot.plot(time,gyr[0],label='Gyr X')
    # gyr_plot.plot(time,gyr[1],label='Gyr Y')
    # gyr_plot.plot(time,gyr[2],label='Gyr Z')
    data1_plot.plot(time,data1)
    data2_plot.plot(time,data2)
    plt.show()
    pass

def plot_2x_vertical_no_time(data1,data2):
    fig, (data1_plot,data2_plot) = plt.subplots(2)
    # acc_plot.plot(time,acc[0],label='Acc X')
    # acc_plot.plot(time,acc[1],label='Acc Y')
    # acc_plot.plot(time,acc[2],label='Acc Z')
    # gyr_plot.plot(time,gyr[0],label='Gyr X')
    # gyr_plot.plot(time,gyr[1],label='Gyr Y')
    # gyr_plot.plot(time,gyr[2],label='Gyr Z')
    data1_plot.plot(data1)
    data2_plot.plot(data2)
    plt.show()
    pass

def plot_3x_vertical(data1,data2,data3,time):
    fig, (d1_plot,d2_plot,d3_plot) = plt.subplots(3)
    d1_plot.plot(time,data1)
    d2_plot.plot(time,data2)
    d3_plot.plot(time,data3)
    plt.show()
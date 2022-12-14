import matplotlib.pyplot as plt


def plot_data_preprocessing():
    n_cpus = [1, 2, 4, 6, 8, 10, 12]
    processing_times = [111.29, 111.32, 108.89, 112.18, 114.45, 111.77, 111.74]

    plt.title("Data Preprocessing time vs Num of Processes", fontsize=20)
    plt.plot(n_cpus, processing_times)

    plt.ylabel("Processing Time(s)")
    plt.xlabel("Number of Processes")

    plt.ylim((50, 150))

    plt.show()
    plt.pause(0)


def plot_feature_enhancement():
    processing_time = [330.73, 165.34]

    plt.xticks([1, 2], labels=["serial_executing", "parallel_executing"], fontsize=13)
    plt.bar([1, 2], processing_time, width=0.4)

    plt.title("Serial vs Parallel Processing Time", fontsize=15)
    plt.ylabel("Processing Time(s)", fontsize=13)

    plt.show()
    plt.pause(0)


def plot_ML_fitting():
    # fitting RF
    n_cpus = [1, 2, 4, 6, 8, 10, 12]
    processing_times = [1140, 611, 355, 275, 246, 205, 191]

    plt.subplot(1, 2, 1)
    plt.title("RandomForest Fitting time vs Threads Num", fontsize=15)
    plt.plot(n_cpus, processing_times)

    plt.ylabel("Processing Time(s)", fontsize=12)
    plt.xlabel("Number of Threads", fontsize=12)

    plt.ylim((40, 1200))

    # fitting with XGBR
    n_cpus = [1, 2, 4, 6, 8, 10, 12]
    processing_times = [249, 132, 78, 60, 53, 50, 45]

    plt.subplot(1, 2, 2)
    plt.title("XG Boost Fitting time vs Threads Num", fontsize=15)
    plt.plot(n_cpus, processing_times)

    plt.ylabel("Processing Time(s)", fontsize=12)
    plt.xlabel("Number of Threads", fontsize=12)

    plt.ylim((40, 1200))

    plt.show()
    plt.pause(0)


def plot_DL_fitting():
    # Single GPU
    batch_size = [64, 128, 256, 512, 1024]
    epoch_time = [0.134, 0.086, 0.069, 0.048, 0.041]

    plt.subplot(1, 2, 1)
    plt.title("Epoch time vs Batch Size(Single GPU)", fontsize=15)
    plt.plot(batch_size, epoch_time)
    for x, y in zip(batch_size, epoch_time):
        plt.text(x, y, str(y))

    plt.ylabel("Epoch Time(s)", fontsize=12)
    plt.xlabel("Batch Size", fontsize=12)

    plt.ylim((0, 0.8))

    # multi GPUs
    batch_size = [64, 128, 256, 512, 1024]
    epoch_time = [0.704, 0.365, 0.191, 0.103, 0.066]

    plt.subplot(1, 2, 2)
    plt.title("Epoch time vs Batch Size(4 GPUs)", fontsize=15)
    plt.plot(batch_size, epoch_time)
    for x, y in zip(batch_size, epoch_time):
        plt.text(x, y, str(y))

    plt.ylabel("Epoch Time(s)", fontsize=12)
    plt.xlabel("Batch Size", fontsize=12)

    plt.ylim((0, 0.8))

    plt.show()
    plt.pause(0)


if __name__ == '__main__':
    plot_data_preprocessing()
    plot_feature_enhancement()
    plot_ML_fitting()
    plot_DL_fitting()

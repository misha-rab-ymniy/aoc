import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

files = ["CPU/MultiNilakanthaMethod",
        "CPU_NO_HT/MultiNilakanthaMethod",
        "GPU/MultiNilakanthaMethod",
        "CPU/NilakanthaMethod",
        "CPU_NO_HT/NilakanthaMethod",
        "GPU/NilakanthaMethod",
        "CPU/LeibnizMethod",
        "CPU_NO_HT/LeibnizMethod",
        "GPU/LeibnizMethod"]

plt.xlabel('Accuracy')
plt.ylabel('Time')

i = 0
while i < len(files):
    try:
        data0 = pd.read_csv(files[i] + ".csv")
        data1 = pd.read_csv(files[i+1] + ".csv")
        data2 = pd.read_csv(files[i+2] + ".csv")
    except FileNotFoundError:
        i += 3
        continue
    plt.title(files[i][4:])
    plt.plot(data0['accuracy'], data0['time'], label=files[i][:3])
    plt.plot(data1['accuracy'], data1['time'], label=files[i+1][:9])
    plt.plot(data2['accuracy'], data2['time'], label=files[i+2][:3])
    plt.legend()
    plt.xticks(np.arange(min(data0['accuracy']), max(data0["accuracy"])+1, 1.0))
    plt.show()
    i += 3
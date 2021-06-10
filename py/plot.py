#!/usr/bin/env python3

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# from statistics import mean, stdev
# from math import sqrt


if __name__ == "__main__":
  file_path = "/home/birromer/ros/file_eq.csv"

  data = pd.read_csv(file_path)

  fig, axes = plt.subplots(nrows=2, ncols=1)

  print(data)

  data[["sim1_eq1", "sim1_eq2", "sim1_eq3"]].plot(ax=axes[0], label="Sim1")
  data[["sim2_eq1", "sim2_eq2", "sim2_eq3"]].plot(ax=axes[1], label="Sim2")
  axes[0].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
  axes[1].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

  plt.show()

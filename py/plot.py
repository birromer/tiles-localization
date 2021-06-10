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

  figure, ax1 = plt.subplots()

  #ax1.plot(data.iloc[:,0], data.iloc[:,1], data.iloc[:,2], label="Sim1")
  ax1.plot(data,label="Sim1")
#  ax1.plot(data.iloc[:,3], data.iloc[:,4], data.iloc[:,5], label="Sim2")
  ax1.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
  plt.show()

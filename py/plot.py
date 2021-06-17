#!/usr/bin/env python3

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
  """
    file_yx = "/home/birromer/ros/data_tiles/eq_yx.csv"
    file_yp = "/home/birromer/ros/data_tiles/eq_yp.csv"
    data_yx = pd.read_csv(file_yx)
    data_yp = pd.read_csv(file_yp)
    fig1, axes1 = plt.subplots(nrows=2, ncols=1)
    fig2, axes2 = plt.subplots(nrows=2, ncols=1)
    print(data_yx)
    print(data_yp)
    data_yx[["sim1_eq1", "sim1_eq2", "sim1_eq3"]].plot(ax=axes1[0], label="Sim1")
    data_yx[["sim2_eq1", "sim2_eq2", "sim2_eq3"]].plot(ax=axes1[1], label="Sim2")
    axes1[0].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    axes1[1].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    data_yp[["sim1_eq1", "sim1_eq2", "sim1_eq3"]].plot(ax=axes2[0], label="Sim1")
    data_yp[["sim2_eq1", "sim2_eq2", "sim2_eq3"]].plot(ax=axes2[1], label="Sim2")
    axes2[0].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    axes2[1].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
  """

  file_test = "/home/birromer/ros/data_tiles/test_sim.csv"
  test_files = [
    "~/test_tiles_85.csv",
    "~/test_tiles_90.csv",
    "~/test_tiles_95.csv",
    "~/test_tiles_100.csv",
    "~/test_tiles_105.csv",
    "~/test_tiles_110.csv",
    "~/test_tiles_115.csv",
  ]

  for tf in test_files:
    data = pd.read_csv(tf)

    fig1, axes1 = plt.subplots(nrows=2, ncols=1)

    print(data)

    data[["sim1_eq1", "sim1_eq2", "sim1_eq3"]].plot(ax=axes1[0], label="Sim1")
    data[["sim2_eq1", "sim2_eq2", "sim2_eq3"]].plot(ax=axes1[1], label="Sim2")

    axes1[0].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    axes1[1].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

    plt.show()
    a = input("Press any key to continue")

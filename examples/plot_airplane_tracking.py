import matplotlib.pyplot as plt
import numpy as np
import sys
import csv

time = []
true_pos = []
est_pos = []
est_pos_std_dev = []
true_vel = []
est_vel = []
est_vel_std_dev = []
true_alt = []
est_alt = []
est_alt_std_dev = []
true_climb = []
est_climb = []
est_climb_std_dev = []

with open(sys.argv[1],'r') as csv_file:
    plots = csv.reader(csv_file, delimiter=',')
    headers = next(plots, None)
    for row in plots:
        time.append(float(row[0]))
        true_pos.append(float(row[1]))
        est_pos.append(float(row[2]))
        est_pos_std_dev.append(float(row[3]))
        true_vel.append(float(row[4]))
        est_vel.append(float(row[5]))
        est_vel_std_dev.append(float(row[6]))
        true_alt.append(float(row[7]))
        est_alt.append(float(row[8]))
        est_alt_std_dev.append(float(row[9]))
        true_climb.append(float(row[10]))
        est_climb.append(float(row[11]))
        est_climb_std_dev.append(float(row[12]))

fig, axes = plt.subplots(4, 1)
fig.suptitle("Airplane Tracking Estimation")

axes[0].plot(time, true_pos, "k-", time, est_pos, "b-", \
             time, np.array(est_pos) + np.array(est_pos_std_dev), "g--", \
             time, np.array(est_pos) - np.array(est_pos_std_dev), "g--")
axes[0].set(xlabel="Time [s]", ylabel="Position [m]")

axes[1].plot(time, true_vel, "k-", time, est_vel, "b-", \
             time, np.array(est_vel) + np.array(est_vel_std_dev), "g--", \
             time, np.array(est_vel) - np.array(est_vel_std_dev), "g--")
axes[1].set(xlabel="Time [s]", ylabel="Velocity [m/s]")

axes[2].plot(time, true_alt, "k-", time, est_alt, "b-", \
             time, np.array(est_alt) + np.array(est_alt_std_dev), "g--", \
             time, np.array(est_alt) - np.array(est_alt_std_dev), "g--")
axes[2].set(xlabel="Time [s]", ylabel="Altitude [m]")

axes[3].plot(time, true_climb, "k-", time, est_climb, "b-", \
             time, np.array(est_climb) + np.array(est_climb_std_dev), "g--", \
             time, np.array(est_climb) - np.array(est_climb_std_dev), "g--")
axes[3].set(xlabel="Time [s]", ylabel="Climb Rate [m/s]")

# Hide x labels and tick labels for top plots and y ticks for right plots.
for axis in axes.flat:
    axis.label_outer()

plt.show()

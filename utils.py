# Defining Plots Attributes
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.axisartist.axislines import Subplot 
import math
import numpy as np

import seaborn as sns

sns.set_style('whitegrid') # darkgrid, white grid, dark, white and ticks
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
plt.rc('legend', fontsize=18)    # legend fontsize
plt.rc('font', size=13)          # controls default text sizes
titlefont = {'fontname':'Times New Roman', 'fontsize':20}
axisfont = {'fontname':'Times New Roman', 'fontsize':20}

# Custom Helper Functions
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def delete_multiple_element(list_object, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            # list_object = np.delete(list_object, idx)
            list_object.pop(idx)
            
            
def move_step(location, speed, acceleration, RL_acceleration, list_of_RL_indices):
    # IDM & RL Vehicles Movement
    t_step = 0.1

    # 1] Acceleration Update of IDM Vehicles According to the Intelligent Driver Model (IDM)
    # Define IDM Variables
    s0 = 2.0
    delta = 4
    T = 1.5
    a = 2.0
    b = 4.0
    v0 = 60.0

    # Let Loop length L = 1000 m
    roadlength = 333

    # Make First Vehicle Follow Last Vehicle
    location_tmp = location.copy()
    location_tmp.append(location_tmp[-1] + (roadlength - location_tmp[-1] + location_tmp[0])) 

    spacing = []

    for i in range(len(location_tmp)-1):
        spacing.append(location_tmp[i+1] - location_tmp[i] - 4.0)

    speed_tmp = speed.copy()
    speed_tmp.append(speed[0])
    deltav = []
    for i in range(len(speed)):
        deltav.append(speed[i] - speed_tmp[i+1])

    # Acceleration Update Formula Split into 'temp' parts
    temp_1 = [c*d for c,d in zip(speed,deltav)]
    temp_2 = [i / (2 * np.sqrt(8)) for i in temp_1]
    temp_3 = [i * T for i in speed]
    s_star = []
    for i in range(len(temp_3)):
        s_star.append(s0 + max(0,temp_3[i] + temp_2[i]))

    temp_4 = []
    for i in range(len(speed)):
        temp_4.append(math.pow(speed[i]/v0, delta))

    temp_5 = [math.pow(e/f, 2) for e,f in zip(s_star, spacing)]
    

    for i in range(len(acceleration)):
    #             self.acceleration[i] = a * (1 - temp_4[i] - temp_5[i])
        if i == 0:
            acceleration[i] = RL_acceleration[0]
        elif i == 4:
            acceleration[i] = RL_acceleration[1]
        elif i == 8:
            acceleration[i] = RL_acceleration[2]
        elif i == 12:
            acceleration[i] = RL_acceleration[3]
            
        else:
            acceleration[i] = a * (1 - temp_4[i] - temp_5[i])

    # 2] Speed Update of IDM Vehicles According to the Intelligent Driver Model (IDM)
    for i in range(len(speed)):
        speed[i] += acceleration[i] * t_step

    for i in range(len(speed)):
        if speed[i] < 0:
            speed[i] = 0

    # 3] Update location
    for i in range(len(location)):
        location[i] += speed[i] * t_step + 0.5 * acceleration[i] * (t_step**2)

    del location_tmp, speed_tmp
    
    return location, speed, acceleration

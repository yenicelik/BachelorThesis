"""
Demo of a simple plot with a custom dashed line.

A Line object's ``set_dashes`` method allows you to specify dashes with
a series of on/off lengths (in points).
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib

min, max = -10, 10

x = np.linspace(min, max)
y = np.linspace(min, max, num=20)
line, = plt.plot(x, 1.0/(1+np.exp(-x)), color='cornflowerblue', linestyle='solid', linewidth=5)

ax = plt.gca()
#plt.setp( ax.get_xticklabels(), visible=False)
plt.setp( ax.get_yticklabels(), visible=False)
#dashes = [10, 5, 100, 5] # 10 points on, 5 off, 100 on, 5 off
#line.set_dashes(dashes)

group_labels = ['-MAX_SIG','','','','MAX_SIG']

ax.set_xticklabels(group_labels)

plt.plot((min, max), (0.5, 0.5), color='black', linestyle='solid', linewidth=1)
plt.plot((0,0), (0,1), color='black', linestyle='solid', linewidth=1)
for point in y:
    plt.plot((point,point), (0, 1), 'k--', linewidth=0.3)

plt.text(-1.5, 0.48, '0.5', fontsize=20)
plt.text(-0.7, 0.98, '1', fontsize=20)
plt.text(-0.7, -0.02, '0', fontsize=20)

matplotlib.rcParams['font.size'] = 20

plt.show()


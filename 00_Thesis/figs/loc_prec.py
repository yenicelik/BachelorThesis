#!/usr/bin/env python
# a bar plot with errorbars
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pylab

N = 7
Means1 = (0.81666787439613531, 0.77199999999999991, 0.75217391304347825, 0.86333333333333341, 0.81000000000000006, 0.9225, 0.78000000000000003) 
Std1 = (0.02456475632558491, 0.011105780420791205, 0.029569016550359005, 0.026982196411811769, 0.031384611472474025, 0.013718578641954445, 0.034628354456119004) 

ind = np.arange(N)  # the x locations for the groups
width = 0.25       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind+width*2, Means1, width, color='palegreen', yerr=Std1)

Means2 = (0.76559299516908208, 0.74199999999999993, 0.71739130434782616, 0.79333333333333324, 0.7466666666666667, 0.86749999999999996, 0.72666666666666657) 
Std2 = (0.024005339905063785, 0.020777012673671384, 0.033059165517209541, 0.027764451051978011, 0.027764451051978018, 0.01735278190748624, 0.017314177228059516) 
rects2 = ax.bar(ind, Means2, width, color='wheat', yerr=Std2)

Means3 = (0.80194263285024148, 0.76800000000000002, 0.73573913043478266, 0.84999999999999995, 0.79000000000000003, 0.90124999999999998, 0.76666666666666672) 
Std3 = (0.025462704610936474, 0.025165519010507087, 0.020018318836869411, 0.027593244438317422, 0.029628354456119004, 0.00867639095374312, 0.020694399970062803) 
rects3 = ax.bar(ind+width, Means3, width, color='lightskyblue', yerr=Std3)

# add some text for labels, title and axes ticks
ax.set_ylabel('Precision at 10')
#ax.set_title('Scores by group and gender')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('overall', 'state', 'city', 'baketball', 'baseball', 'football', 'hockey'), rotation=-45 )

ax.legend( (rects2[0], rects3[0], rects1[0]), ('SGLM', 'CDSG', 'CDSGD'), loc=4 )

def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%.3f'%float(height),
                ha='center', va='bottom')

plt.gcf().subplots_adjust(bottom=0.01)
matplotlib.rcParams['font.size'] = 20
#autolabel(rects1)
#autolabel(rects2)
#autolabel(rects3)
pylab.xlim([-0.25,7])
pylab.ylim([0.6,1])
plt.tight_layout()
plt.show()

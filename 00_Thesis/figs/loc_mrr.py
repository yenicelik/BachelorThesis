#!/usr/bin/env python
# a bar plot with errorbars
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pylab

N = 7
Means1 = (0.49319922866917404, 0.59235084073750655, 0.52525975689251935, 0.32013634989021978, 0.51038783904583472, 0.63148323919068484, 0.37957734625827928) 
Std1 = (0.028303854100397735, 0.031667732446634702, 0.028474417579935777, 0.023810636102394492, 0.021368068417602387, 0.016060177747914215, 0.018442092307904834) 

ind = np.arange(N)  # the x locations for the groups
width = 0.25       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind+width*2, Means1, width, color='palegreen', yerr=Std1)

Means2 = (0.44986579433003432, 0.55684179149401063, 0.48381749131858843, 0.29197680341101395, 0.45658892927391378, 0.56682030809765517, 0.34314944238502428) 
Std2 =  (0.020769949618236722, 0.01362301028905114, 0.019107630011534191, 0.029220603765985904, 0.021312343015917345, 0.02769644686922508, 0.013659663757706676) 
rects2 = ax.bar(ind, Means2, width, color='wheat', yerr=Std2)

Means3 = (0.4810152177595734, 0.57319620001486627, 0.5095321619846511, 0.30391714425924951, 0.49431969121499347, 0.62518691849939144, 0.37993919058428861) 
Std3 = (0.02491247243285714, 0.030697281512148469, 0.042119162302136998, 0.022612085338507261, 0.015096985476705078, 0.022143445329799817, 0.01680587463784524) 
rects3 = ax.bar(ind+width, Means3, width, color='lightskyblue', yerr=Std3)

# add some text for labels, title and axes ticks
ax.set_ylabel('Mean Reciprocal Rank')
#ax.set_title('Scores by group and gender')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('overall', 'state', 'city', 'baketball', 'baseball', 'football', 'hockey'), rotation=-45 )

ax.legend( (rects2[0], rects3[0], rects1[0]), ('SGLM', 'CDSG', 'CDSGD'), loc = 4 )

def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),
                ha='center', va='bottom')

plt.gcf().subplots_adjust(bottom=0.15)
matplotlib.rcParams['font.size'] = 20
#autolabel(rects1)
#autolabel(rects2)
#autolabel(rects3)
pylab.xlim([-0.25,7])
pylab.ylim([0.2,0.7])
plt.tight_layout()
plt.show()

"""
Simple demo of a horizontal bar chart.
"""
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from contextnum import contextnumcomplete
from context import contextsid
from reverse_context import reverse_list

dic = contextnumcomplete.copy()

clist = [dic[i] for i in range(195,259)]

ilist = [i[0] for i in sorted(enumerate(clist), key=lambda x:x[1], reverse=True)]

ilist = ilist[2:22]

# Example data
#people = ('Tom', 'Dick', 'Harry', 'Slim', 'Jim')
people = [reverse_list[i+194] for i in ilist]

y_pos = np.arange(len(people))
#performance = 3 + 10 * np.random.rand(len(people))
pop = [100*dic[i+195]/float(dic[0]) for i in ilist]

matplotlib.rcParams['font.size'] = 17
plt.barh(y_pos, pop[::-1], height = 0.8, align='center', alpha=0.4)
plt.yticks(y_pos, people[::-1])
plt.ylim(-1, len(people))
plt.xlabel('Frequency (%)')
plt.tight_layout()
plt.gcf().subplots_adjust(left=0.24)
plt.show()

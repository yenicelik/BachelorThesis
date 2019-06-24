import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
from contextnum import contextnumcomplete
from context import contextsid
from reverse_context import reverse_list

dic = contextnumcomplete.copy()

clist = [dic[i] for i in range(1,195)]
cilist = [i[0] for i in sorted(enumerate(clist), key=lambda x:x[1], reverse=True)]
cilist = cilist[:20]

cpeople = [reverse_list[i] for i in cilist]

cy_pos = np.arange(len(cpeople))
cpop = [100*dic[i+1]/float(dic[0]) for i in cilist]

plt.subplot(121)

plt.barh(cy_pos, cpop[::-1], height = 0.8, align='center', alpha=0.4)
plt.yticks(cy_pos, cpeople[::-1])
plt.ylim(-1, len(cpeople))
plt.xlabel('Frequency (%)')
plt.tight_layout()

slist = [dic[i] for i in range(195,259)]
silist = [i[0] for i in sorted(enumerate(slist), key=lambda x:x[1], reverse=True)]
silist = silist[:20]

speople = [reverse_list[i+194] for i in silist]
sy_pos = np.arange(len(speople))
spop = [100*dic[i+195]/float(dic[0]) for i in silist]

plt.subplot(122)
plt.barh(sy_pos, spop[::-1], height = 0.8, align='center', alpha=0.4)
plt.yticks(sy_pos, speople[::-1])
plt.ylim(-1, len(speople))
plt.xlabel('Frequency (%)')
plt.tight_layout()

plt.show()

import matplotlib
import matplotlib.pyplot as plt
from contextnum import contextnumcomplete 
from matplotlib import font_manager as fm

legend_size = 15
perc_size = 15
text_size = 20

def my_autopct(pct):
    real_val = (pct/float(100)) * (dict[259]+dict[260])
    #return '{p:0.2f}'.format(p=pct)
    return '{p:0.2f}%'.format(p=100*(real_val/float(dict[0])))

if __name__ == "__main__":
    dict = contextnumcomplete.copy()

    labels = 'Male', 'Female'
    sizes = [dict[259], dict[260]]
    #colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
    colors = ['lightgreen', 'palegoldenrod']
    #explode = (0, 0.1, 0, 0) # only "explode" the 2nd slice (i.e. 'Hogs')
    explode = (0, 0) # only "explode" the 2nd slice (i.e. 'Hogs')

    #plt.pie(sizes, explode=explode, labels=labels, colors=colors,
    #        autopct='%1.1f%%', shadow=True, startangle=90)
    patches, texts, autotexts = plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct=my_autopct, shadow=True, startangle=90)
    proptease = fm.FontProperties()
    proptease.set_size(20)
    plt.setp(autotexts, fontproperties=proptease)
    plt.setp(texts, fontproperties=proptease)

    plt.axis('equal')
    matplotlib.rcParams['font.size'] = 20
    leg_labels = ['{0:.2f} m'.format(dict[259]/float(1000000)), '{0:.2f} m'.format(dict[260]/float(1000000))]
    plt.gcf().subplots_adjust(top=1.2)
    plt.legend(patches, leg_labels, loc="best")
    plt.tight_layout()

    plt.show()
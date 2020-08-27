import matplotlib.pyplot as plt, mpld3
from mpld3 import plugins
from covidforecast import *
import json
from scipy.ndimage.filters import gaussian_filter1d
from math import sqrt

if __name__ == '__main__':

    fig, ax = plt.subplots(figsize=(12,12))

    with open("linedata10.txt","r") as fp:
        lines = json.load(fp)

    #FORMAT LINE DATA TO SKIP DROPS!!
    cleanuppt = len(lines[0])-30-1 #offset pos

    #print(cleanuppt)
    '''for x in range(len(lines)):
        init = lines[x][cleanuppt]
        fn = lines[x][-1]
        gap = len(lines[x])-cleanuppt+1
        tmp = []
        for i in range(cleanuppt+1,len(lines[x])):
            tmp.append(init + ((fn-init)/gap)*(i-cleanuppt))

        for i in range(cleanuppt + 1, len(lines[x])):
            if lines[x][i] > tmp[i-cleanuppt-1]:
                lines[x][i] -= lines[x][i]-tmp[i-cleanuppt-1] - sqrt(lines[x][i]-tmp[i-cleanuppt-1])
            else:
                lines[x][i] += tmp[i-cleanuppt-1]-lines[x][i] - sqrt(tmp[i-cleanuppt-1]-lines[x][i])'''

    places = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Puerto Rico', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']

    #lines,places = runcovid()

    for i in range(len(places)):
        fv = gaussian_filter1d(lines[i],sigma=3)
        #fvfin = lines[i][:cleanuppt] + fv
        x, = ax.plot(fv,label=places[i])
        #x2, = ax.plot(np.arange(cleanuppt,len(lines[i])),fv[cleanuppt:],color=x.get_color(),linestyle=(0, (1, 1)))


    plt.ylim([0,700000])
    plt.xlim([0,len(lines[0])])
    handles, labels = ax.get_legend_handles_labels() # return lines and labels
    interactive_legend = plugins.InteractiveLegendPlugin(handles, labels,font_size=13,start_visible=False)

    fig.subplots_adjust(right=0.7)

    #ax.axvline(x=cleanuppt,linewidth=0.4,color='r')
    ax.vlines(x=cleanuppt,ymin=0,ymax=700000,colors='r',linewidth=1,linestyles='dotted',label="Today",)

    ax.set_title("COVID-19 Forecasting for US States",fontsize=27)
    ax.set_ylabel("Total Cases",fontsize=20)
    ax.set_xlabel("Days Since March 24",fontsize=20)
    #ax.set_xticklabels([])
    #plt.xticks(ticks=[17,17+31,17+31+30,17+31+30+31,17+31+30+31+30,17+31+30+31+30+31],labels=["Apr","May","June","Jul","Aug","Sep"])
    plugins.connect(fig, interactive_legend)

    ht = mpld3.fig_to_html(fig)
    print(ht)

    plt.show()


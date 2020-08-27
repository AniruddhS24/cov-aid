from src.covidforecast import *

fig, ax = plt.subplots()
ax.set_title('Click on legend line to toggle line on/off')
plns = []
lines,places = runcovid()
for i in range(len(lines)):
    x, = ax.plot(lines[i],label=places[i])
    plns.append(x)

leg = ax.legend(bbox_to_anchor=(1.04,1), loc='upper left', fancybox=True, shadow=True, fontsize=6)

leg.get_frame().set_alpha(0.4)


# we will set up a dict mapping legend line to orig line, and enable
# picking on the legend line

lined = dict()
for legline, origline in zip(leg.get_lines(), plns):
    legline.set_picker(5)  # 5 pts tolerance
    lined[legline] = origline
    origline.set_visible(False)


def onpick(event):
    # on the pick event, find the orig line corresponding to the
    # legend proxy line, and toggle the visibility
    legline = event.artist
    origline = lined[legline]
    vis = not origline.get_visible()
    origline.set_visible(vis)
    # Change the alpha on the line in the legend so we can see what lines
    # have been toggled
    if vis:
        legline.set_alpha(1.0)
    else:
        legline.set_alpha(0.2)
    fig.canvas.draw()

fig.canvas.mpl_connect('pick_event', onpick)

plt.show()
'''
Matplotlib
most popular plotting, similar to matlab plotting
works well pandas, matplotlib
seaborn built off it

conda install matplotlib

matplotlib/gallery - various plots capable of, see capabilities

http://www.matplotlib.org - The project web page for matplotlib.
https://github.com/matplotlib/matplotlib - The source code for matplotlib.
http://matplotlib.org/gallery.html - A large gallery showcaseing various types of plots matplotlib can create. Highly recommended!
http://www.loria.fr/~rougier/teaching/matplotlib - A good matplotlib tutorial.
http://scipy-lectures.github.io/matplotlib/matplotlib.html - Another good matplotlib reference.
'''

import matplotlib.pyplot as plt
# %matplotlib inline #important so display jupyter
#else in python just plt.show()

import numpy as np 
x=np.linspace(0,5,11)
y=x**2

# Functional matplotlib method (OOP usually better)

plt.plot(x,y, 'r-') #red, dashed
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Title')
plt.show() #not necc in jupyter

#multiplots same canvas
plt.subplot(1,2,1)#rows, cols, plot referring to
plt.plot(x,y,'r')
plt.subplot(1,2,2)
plt.plot(y,x, 'b')
plt.show()

''' 
OOP
'''

fig=plt.figure() #figureobject, blank canvas to add axes to
axes = fig.add_axes([0.1,0.1,0.8,0.8])#to left, right axes, width height axes
axes.plot(x,y)
axes.set_xlabel('X')
axes.set_ylabel('Y')
axes.set_title("Title")

fig= plt.figure() # can input figsize
axes1=fig.add_axes([0.1,0.1,0.8,0.8])#manually choose location
axes2= fig_add_axes([0.2,0.5,0.4,0.3])#percentage to left right, width height
#The above axes2 overlaps axes1

axes1.plot(x,y)
axes2.plot(y,x)
axes1.set_title('large plot')
axes2.set_title('small plot')

# Create subplots
fig.axes = plt.subplots()#can specify row, cols

axes.plot(x,y)

fig, axes=plt.subplots(nrows=1, ncols=2)#does add_axes auto off specified subplots
plt.tight_layout() # fixes subplots overlap
#axes is array of matplotlib axes
for current_ax in axes:
  current_ax.plot(x,y)

axes[0].plot(y,x)
axes[0].set_title("First plot")
axes[1].set_title("Second plot")

#Figure sizze, aspect ratio, dpi
fig = plt.figure(figsize=(3,2), dpi=100)# dots per pixel, inches width and height
ax= fig.add_axes([0,0,1,1])
ax.plot(x,y)

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8,2))
axes[0].plot(x,y)
axes[1].pot(y,x)
plt.tight_layout()

fig.savefig('pic.jpg', dpi=200)# can be png, can specify dpi

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(x,y)

#legends, use label text to clarify data
fig = plt.figure()
ax= fig.add_axes([0,0,1,1])
ax.plot(x,x**2, label='X squared')
ax.plot(x,x**3, label='X cubed')
ax.legend(loc='best')#some legend might overlay plot, so position using loc
# 'upper right' left, center, lower
ax.legend(loc=(0.1,0.1))#express as percentage from bottom left


''' Customising lines and appearance'''
fig = plt.figure()
axes = fig.add_axes([0,0,1,1])
axes.plot(x,y,color='blue', linewidth=3,markerfacecolor='yellow', markeredgewidth=3, markeredgecolor='green', alpha=0.5,marker='o', markersize=5, linestype='--')# rgb hex codes #FFFFFF, 
#linewidth 1 is default, 5 is 5* thick
#alpha controls opacity
# lw is shorthand for linewidth
# : dots, steps, -- dashed - solid
#mark where points occur on plot
# o dot, * star, + plus, 1 number code
#specify sizes of markers showing where datapoints are using markersize
# markeredgecolor, markeredgewidth, markerfacecolor

ax.plot(x,y, color='purple', lw=2, ls='--')
ax.set_xlim([0,1])
ax.set_ylim([0,2])

#Lots of different plottypes in matplotlib, scatter, boxplots, 
# but seaborn has ggreater capabilities for plotting so what will be using.


fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (10,4))
axes[0].plot(x,y, linewidth='5', color="yellow")
axes[1].plot(x,z, linewidth='8',ls='--', marker='o' , markerfacecolor='purple', markeredgecolor='black', markeredgewidth="1", markersize="4")
plt.show()

axes[1].set_yscale("log") #logarithmic scale


fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(x, x**2, x, x**3, lw=2) #plot two lines one for x**2, one for x**3

ax.set_xticks([1, 2, 3, 4, 5]) # where labels should be placed on axes
#ability to add custom tick labels
ax.set_xticklabels([r'$\alpha$', r'$\beta$', r'$\gamma$', r'$\delta$', r'$\epsilon$'], fontsize=18)

yticks = [0, 50, 100, 150]
ax.set_yticks(yticks)
ax.set_yticklabels(["$%.1f$" % y for y in yticks], fontsize=18); # use LaTeX formatted labels

# scientific notation is often better with larger numbers
# here will end up 10^2 scale
fig, ax = plt.subplots(1, 1)
      
ax.plot(x, x**2, x, np.exp(x))
ax.set_title("scientific notation")

ax.set_yticks([0, 50, 100, 150])

from matplotlib import ticker # used to create tick formatter, 1000 -> 10 on 10^2 scake
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 
ax.yaxis.set_major_formatter(formatter) 

# distance between  axis and the tick numbers on the axes
matplotlib.rcParams['xtick.major.pad'] = 5
matplotlib.rcParams['ytick.major.pad'] = 50

# padding between axis label and axis numbers, larger label moves away from the axis
ax.xaxis.labelpad = 50
ax.yaxis.labelpad = 5


# restore defaults, distance betwen axis and tick numbers
matplotlib.rcParams['xtick.major.pad'] = 3
matplotlib.rcParams['ytick.major.pad'] = 3

#adjust subplot so takes up less percentage of figure to prevent clipping of axes labels when outputting graph
fig.subplots_adjust(left=0.15, right=.9, bottom=0.1, top=0.9)

# Create background grid
axes[1].grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)

ax.spines['left'].set_color('red') #set customised axis spines, can be bottom, top, left, right
ax.spines['left'].set_linewidth(2)
ax.yaxis.tick_left() #only ticks on left hand side

''' Dual y/x axes in event overlapping x axis (can do same for y)'''
fig, ax1 = plt.subplots()

ax1.plot(x, x**2, lw=2, color="blue")
ax1.set_ylabel(r"area $(m^2)$", fontsize=18, color="blue")
for label in ax1.get_yticklabels(): #set all ax1 left tick labels to blue
    label.set_color("blue")
    
ax2 = ax1.twinx() #new axes matched on x
ax2.plot(x, x**3, lw=2, color="red")
ax2.set_ylabel(r"volume $(m^3)$", fontsize=18, color="red") #label right y axis
for label in ax2.get_yticklabels():
    label.set_color("red")


# Create custom axes spints in center of plot
fig, ax = plt.subplots()
ax.spines['right'].set_color('none') #rmove right and top spines
ax.spines['top'].set_color('none')

ax.xaxis.set_ticks_position('bottom') #ticks are below axes
ax.spines['bottom'].set_position(('data',0)) # set position of x spine to x=0

ax.yaxis.set_ticks_position('left') # ticks are to left of axes
ax.spines['left'].set_position(('data',0))   # set position of y spine to y=0

xx=np.linspace(-0.75, 1., 100)
ax.plot(xx, xx**3)


''' Gallery of graphs '''
fig, axes = plt.subplots(1, 4, figsize=(12,3))

axes[0].scatter(xx, xx + 0.25*np.random.randn(len(xx))) #scatter
axes[0].set_title("scatter") 

axes[1].step(n, n**2, lw=2) #steps inside of curved best fit line to points
axes[1].set_title("step")

axes[2].bar(n, n**2, align="center", width=0.5, alpha=0.5) # bar graph
axes[2].set_title("bar")

axes[3].fill_between(x, x**2, x**3, color="green", alpha=0.5) # fills space between x**2, x**3
axes[3].set_title("fill_between")

# Annotating custom, manual text in figures
ax.text(0.15, 0.2, r"$y=x^2$", fontsize=20, color="blue")
ax.text(0.65, 0.1, r"$y=x^3$", fontsize=20, color="green")#at pos 65%, 10%, create label y=x^3, uses r formatting

''' Multiple subplots 

fig.add_axes or using a sub-figure layout manager such as subplots, subplot2grid, or gridspec:
'''
fig, ax = plt.subplots(2, 3)
fig.tight_layout()

fig=plt.figure()
ax3 = plt.subplot2grid((3,3), (1,2), rowspan=2) #3*3 plot basis, plot from 1 row down, 2 cols across, point down two rows(colspan=2)
ax4 = plt.subplot2grid((3,3), (2,0)) # 2 rows down, 0 cols across, plot 1/9 plot

import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(2, 3, height_ratios=[2,1], width_ratios=[1,2,1])
# 
for g in gs:
    ax = fig.add_subplot(g)

# add_axes can be used to add inset zooom ins plots of larger axes]# inset
inset_ax = fig.add_axes([0.2, 0.55, 0.35, 0.35]) # X, Y, width, height

inset_ax.plot(xx, xx**2, xx, xx**3)
inset_ax.set_title('zoom near origin')

# set axis range
inset_ax.set_xlim(-.2, .2)
inset_ax.set_ylim(-.005, .01)

''' 
COLOR MAPS
COLORMAPS, CONTOUR FIGURES, useful plot functions two variables, map difference in 1d

matplotlib.cm. ... color maps presets
'''

#pcolor
# Create mesh grid from two array numbers
phi_m = np.linspace(0, 2*np.pi, 100)
phi_p = np.linspace(0, 2*np.pi, 100)
X,Y = np.meshgrid(phi_p, phi_m)
Z = flux_qubit_potential(X, Y).T

fig, ax = plt.subplots()
p = ax.pcolor(X/(2*np.pi), Y/(2*np.pi), Z, cmap=matplotlib.cm.RdBu, vmin=abs(Z).min(), vmax=abs(Z).max())
cb = fig.colorbar(p, ax=ax)


#imshow
fig, ax = plt.subplots()

im = ax.imshow(Z, cmap=matplotlib.cm.RdBu, vmin=abs(Z).min(), vmax=abs(Z).max(), extent=[0, 1, 0, 1])
im.set_interpolation('bilinear')

cb = fig.colorbar(im, ax=ax)


#draws contours instead of map coloes, contour lines
fig, ax = plt.subplots()
cnt = ax.contour(Z, cmap=matplotlib.cm.RdBu, vmin=abs(Z).min(), vmax=abs(Z).max(), extent=[0, 1, 0, 1])


'''
3D figures using matplotlib axes3d
project='3D' way to add 3d axes to canvas

'''

from mpl_toolkits.mplot3d.axes3d import Axes3D

fig = plt.figure(figsize=(14,6))
ax= fig.add_subplot(1,2,1,projection='3d')
p= ax.plot_surface(X,Y,Q, rstride=4, cstride=4, linewidth=0)

#surface plot with coloe grading and color bar
ax=fig.add_subplot(1,2,2,projection='3d')
p= ax.plot_surface(X,Y,Z, rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, linewidth=0, antialiased=False)
cb = fig.colorbar(p, shrink=0.5)

# wireframe plot
ax = fig.add_subplot(1, 1, 1, projection='3d')
p = ax.plot_wireframe(X, Y, Z, rstride=4, cstride=4)


#main surface plot with contour projections on x,y,z axes
fig = plt.figure(figsize=(8,6))

ax = fig.add_subplot(1,1,1, projection='3d')

ax.plot_surface(X, Y, Z, rstride=4, cstride=4, alpha=0.25)
cset = ax.contour(X, Y, Z, zdir='z', offset=-np.pi, cmap=matplotlib.cm.coolwarm)
cset = ax.contour(X, Y, Z, zdir='x', offset=-np.pi, cmap=matplotlib.cm.coolwarm)
cset = ax.contour(X, Y, Z, zdir='y', offset=3*np.pi, cmap=matplotlib.cm.coolwarm)

ax.set_xlim3d(-np.pi, 2*np.pi);
ax.set_ylim3d(0, 3*np.pi);
ax.set_zlim3d(-np.pi, 2*np.pi);

#offset contours so lie on axes by setting ylim3d, xlim3d, zlim3d to match offset


#matplotlib has stylesheets
import matplotlib as plt
plt.style.use('ggplot')
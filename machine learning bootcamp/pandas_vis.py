'''
Panda built in visualisation
Used for quick experimentation hist, plot.line are most useful
'''

import numpy as np 
import pandas as pd 
import seaborn as sns
#%matplotlib inline

df1=pd.read_csv('df1')
df2=pd.read_csv('df2')
df1.head()
df2.head()

df1['A'].hist(bins=30)#calls matplotlib under the hood
# unless import seaborn then defaults to its display style.

df1['A'].plot(kind='hist', bins=30)# can call plot, then specify kind
df1['A'].plot.hist()

df2.plot.area(alpha=0.4) #area plot, area of numerical values
df2.plot.bar() #takes index as categorical, then plot each numerical col as group for each category
# if large number of indexes wont work
df2.plot.bar(stacked=True)#plots numerical ontop of eachother
#only good quick analysis

df1['A'].plot.hist()
df1.plot.line(x=df1.index, y='B', figsize=(12,3), lw=1)
#lineplot,start passing in matplotlib arguments , linewidh etc

df1.plot.scatter(x='A', y='B', c='C', cmap='coolwarm')
#can set color based off another col, 3d plot
# show by size by passing in dataframe col
df1.plot.scatter(x='A', y='B', s=df1['C']*100)

df2.plot.box() #boxplot

df = pd.DataFrame(np.random.randn(1000,2), columns=['a','b'])
df.plot.hexbin(x='a', y='b', gridsize=25, cmap='coolwarm')
# like scatter plot except hexagonal bins
#gridsize controls hex size

df2['a'].plot.kde()
df2.plot.density() #plot density (kde) of each col in dataframe overlapping



d3.plot.scatter(x='a', y='b', figsize=(12,3)), s=50, c='red')
df3.plot.scatter(x='a', y='b', color='red', edgecolor='black', xlim=[-0.2,1.2], ylim=[-0.2,1.2], s=40, figsize=[10,4])

df3['a'].plot.hist(linewidth=1, edgecolor='black', color='blue', bins=5)

#matplotlib has stylesheets
import matplotlib as plt
plt.style.use('ggplot')
df3['a'].plot.hist(bins=20, alpha=0.6)

import seaborn as sns
sns.set_style('darkgrid')
df3['a'].plot.hist(bins=30, color='red', alpha=0.5, xlim=[0.0,1.0],ylim=[0,35]  )

df3[['a','b']].plot.box()

#kde same function
df3['d'].plot.density(color='red', ls='--', linewidth=3)# lw, ls


df3.ix[0:30].plot.area(alpha=0.5)# ix is first 30 rows

ax = df3.plot.area(xlim=[0,30],ylim=[0,3.0], cmap='coolwarm', legend=False)
ax.legend(loc=[1,0.5])

#or
df3.ix[0:30].plot.area(alpha=0.5)# ix is first 30 rows
plt.legend(loc='center left', bbox_to_anchor=(1.0,0.5))
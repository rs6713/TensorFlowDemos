'''
  statistical plotting library, built on top matplotlib, beautiful default styles
  conda install seaborn
  can read all code, seaborn open source
  example gallery on github.
  api reference various plot types.
'''
import seaborn as sns
#%matplotlib inline

###################################
#  Distribution plots

tips= sns.load_dataset('tips')#read in dataset pandsa
tips.head()
sns.distplot(tips['total_bill'], kde=False, bins=30)# get distribution kde, and historgram
#allow to chose show kde, num bins

sns.jointplot( x='total_bill', y='tip', data=tips, kind='hex')#col names , reg, hex, kde- 2d kde shows density
# plots two distributions with scatter plot x,y between
# kind lets impact what occurs within, default is scatter, can have hex, or regression line on top of scatter

sns.pairplot(tips, hue='sex', palette='cool')#plot pairwise relationships across dataframe, all vs eachother, when against self shows histogram,else scatter
#can pass in hue, can colour datapoints based off a categorical col otherwise excluded from visualisation
#coolwarm colour schemes

sns.rugplot(tips['total_bill'])#draws dash for every point along distribution line
#kde kernel density estimation plot, on every rugplot dash is a gaussian distribution, sum all gaussian distributions to give kde
#gives smooth curve

sns.kdeplot(tips['total_bill'])#just plot kde

#################################
# Categorical plot

import numpy as np
sns.barplot(x='sex',y='total_bill',data=tips, estimator=np.std)#aggregate categorical data based off numerical col and function, default mean
#can put in own custom functions

sns.countplot(x='sex', data=tips) #estimator explicitly counting number occurences
#same as barplot but y axis already chosen

sns.boxplot(x='day',y='total_bill',data=tips, hue='smoker')#shows quartiles, distr for each category along x.
# dots are outliers, with hue can split each boxplot by day and smoker, with 4 days, now have 8 plots

sns.violinplot(x='day', y='total_bill', data=tips, hue='sex', split=True)# allows plot all components correspond to datapoints, more information about individual data points, but less easy to read
# can also do hue, split puts colour distribution of each category on either side so still 4 plots but spilt

sns.stripplot(x='day', y='total_bill', data=tips, jitter=True, hue='sex')
#vertical scatter plots, can' tell how many points stacked on eachother, so add jitter to add random kitter left/right

sns.swarmplot(x='day', y='total_bill', data=tips)#combination stripplot and violin plot, violin form with points spread out in shape
# dont scale that well to large numbers, and large distributions long computation , not great to use in isolation

#plot both on top of eachother
sns.violinplot(x='day', y='total_bill', data=tips)
sns.swarmplot(x='day', y='total_bill', data=tips, color='black')

#swarmplots, violinplots more suited data scientist not corporate display

sns.factorplot(x='day', y='total_bill', data='tips', kind='bar')# call any type plot you want

##################################################
# Matrix plots

flights = sns.load_dataset('flights')

# Need data in matrix plot, so heatmap works, need variable on rows and cols
tc=tips.corr()#now col, index label relevant to data. could also use pivot table
sns.heatmap(tc, annot=True, cmap='coolwarm')

fp = flights.pivot_table(index='month', columns='year',values='passengers' )
sns.heatmap(fp, cmap='magma', linecolor='white', linewidths=3) #can see which month/year has most travel
#magma dark to light
#linewidths increase separation of grid elems

sns.clustermap(fp, standard_scale=1)#normalise so can see similarity of values in comparison cols/indexes
#same as heatmap but now tries to do clusters of entries based off similarities. 
#shows most similar cols, most similar rows to eachother, see heirarchies of most similar months for example
#cluster maps do rearrange cols/indexes so most similar next to eachother

##########################################################
# grids

iris = load_dataset('iris')

iris['species'].unique()
sns.pairplot(iris)
# Using grid mechanism can customise pairplot
g = sns.PairGrid(iris)# just creates empty subplots for us, where in pairplot would be filled
# now can map plottypes
g.map_diag(sns.distplot)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot) # lower is all plots below diagonal, more control over plots to map

g = sns.FacetGrid(data=tips, col='time', row='smoker')
# same subplots as before but on chosen cols,
g.map(sns.distplot, 'total_bill')
# chose plottype, separate based off col, row, distributions are distplot of total bill
g.map(plt.scatter, 'total_bill', 'tips')#scatter needs 2 data cols


#########################################################
# Regression plots
#linear model plot lmplot
sns.lmplot(x='total_bill', y='tip', data=tips, hue='sex', markers=['o', 'v'], scatter_kws={'s':100}   )
#different markers based on hue
#scatter with multiple regression lines for each hue
#to affect matplotlib under the hood, change size markers to 100
sns.lmplot(x='total_bill', y='tip', data=tips, col='sex', row='time')
#now have two subplots one for each sex, each with scatter and regression line
#split into rows with additional categorical cols
#can use hue, row and col, to give huge plot

sns.lmplot(x='total_bill', y='tip', data=tips, col='sex', hue='sex', aspect=0.6, size=8)
#change aspect ratio plot and size.


####################################################
# Style and color

sns.set_style('ticks')#ticks at side of grid
#darkgrid - dark grey grid background
#white - white grid background, whitegrid
sns.countplot(x='sex', data=tips)
sns.despine(left=True)#remove top, bottom spines auto,left/right can be specified

plt.figure(figsize=(12,3))#overrides seaborn, set size figures
sns.countplot(x='sex', data=tips)

sns.set_context('poster', font_scale=3)#paper, notebook, talk, poster
#size of font

sns.lmplot(x='total_bill', y='tip', data=tips, hue='sex', palette='coolwarm')
#go to matplotlib colormap doc page, see colormaps
#inferno, plasma, seismic, magma

fg = sns.FacetGrid(data=titanic, col='sex')
fg.map(plt.hist, 'age')
sns.distplot(titanic['fare'], color='red', bins=30, kde=False)
#palette= 'rainbow', Set2










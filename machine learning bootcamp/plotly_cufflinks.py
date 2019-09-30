'''
 plotly - interactice visualisation library,for jupyter 
 cufflinks connects plotly to pandas

pip install plotly
pip install cufflinks

plotly, library is free, 

'''
import numpy as np 
import pandas as pd 
from plotly import __version__
print(__version__)
import cufflinks as cf 
#%matplotlib_inline

from plotly.offline import download_plotlyjs, init_notebook_mode, iplot 
init_notebook_mode(connected=True)

cf.go_offline()

df=pd.DataFrame(np.random.randn(100,4), columns='A B C D'.split())
df2 = pd.DataFrame({'Category':['A', 'B','C'], 'Values':[32,43,50]})

df.plot()
df.iplot() #now interactive image, can zoom in, save plot, downlaod png


df.iplot(kind='scatter', x='A', y='B', mode='markers', s=20)

df2.iplot(kind='bar', x='Category', y='Values')

df.count().iplot(kind='bar')
df.sum().iplot(kind='bar')#a,b,c,d cols
df.iplot(kind='box')#get  quartile info

df3 = pd.DataFrame({'x':[1,2,3,4,5], 'y': [10,20,30,20,10], 'z':[500,400,300,200,100]})
df3.iplot(kind='surface', colorscale='rdylbu')#3d surface plot
# red yellow blue

df['A'].iplot(kind='hist', bins=50)
df.iplot(kind='hist', bins=50)#overlapping cols

df[['A','B']].iplot(kind='spread')#see spread between two vars

df.iplot(kind='bubble', x='A', y='B', z='C')
#scatter but with size determined by 3rd dimension

df.scatter_matrix() # hist on same cols, rest are col vs scatter in subplots


# Some technical analysis plots built in, ta.py in cufflinks, only relevant for financial data, running averages etc




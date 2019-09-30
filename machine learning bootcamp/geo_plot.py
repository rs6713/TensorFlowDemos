'''
Geographical plotting

Challenging due to various formats data, use plotly, but matplotlib also has basemap extension.

plot.ly/python/reference/#choropleth
see all plotly types
usually reference documentation for these sort of plots
'''
import plotly.plotly as py 
import plotly.graph_objs as go 
# tab to autocomplete
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

#can see everything in jupyter notebook
init_notebook_mode(connected=True)
import pandas as pd

# Choropleth maps- draw data local or global scale
#build date dic
data = dict(
  type='choropleth', #type geoplot
  locations=['AZ', 'CA', 'NY'],#list state abbrev codes
  locationmode = 'USA-states', #can go to county level
  colorscale='Greens',
  text=['text 1', 'text 2', 'text 3'],#labels for locations
  z= [1.0,2.0,3.0], # value want to represent as a color, e.g. population
  colorbar={
    'title':'Colorbar title here'
  }
)

layout = dict(geo={'scope':'usa'})# use states choropleth

choromap = go.figure(data = [data], layout=layout)
iplot(choromap)  #could also just do plot


df = pd.read_csv('2011_US_AGRI_Exports')
df.head()

data = dict(
  type='choropleth',
  colorscale= 'YlOrRd',
  locations= df['code'],
  locationmode = 'USA-states',
  z = df['total exports'],
  text = df['text'],
  marker = dict(line=dict(color='rgb(255,255,255)', width=2)),#spacing between states
  colorbar={'title': 'Millions USD'}
)

layout = dict(
  title = '2011 US Agriculture Exports by State',
  geo = dict(scope='usa', showlakes=True,lakecolor='rgb(85,173,240)' )
)

choromap2=go.Figure(data=[data], layout=layout)
iplot(choromap2)

#World choropleth plot
df= pd.read_csv('2014_World_GDP')
df.head()
data = dict(
  type= 'choropleth',
  locations = df['CODE'],
  z = df['GDP BILLIONS'],
  text = df['COUNTRY'],
  colorbar = {'title': 'GDP in Billions USD'}
)
layout = dict(
  title = '2014 Global GDP',
  geo = dict(showframe=False, projection= {'type':'Mercator'})
)

choromap3 = go.Figure(data=[data], layout=layout)
iplot(choromap3)

data = dict(
    z= df2['Voting-Age Population (VAP)'],
    locations= df2['State Abv'],
    locationmode='USA-states',#'country names'
    colorscale='Greens',#Viridis
    colorbar={'title':"VAP Per state"},
    type='choropleth',
    text= df2['State']
    #reversescale=True

)
layout= dict(
    title = "VAP Per State in US",
    geo= dict(scope = 'usa', showcoastlines=True, coastlinewidth=5, coastlinecolor='rgb(255,0,0)',
             showlakes=True, lakecolor='rgb(0,0,255)', showrivers=True, rivercolor='rgb(255,0,255)')
)
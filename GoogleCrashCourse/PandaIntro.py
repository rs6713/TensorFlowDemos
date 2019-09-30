import pandas as pd 
import numpy as np
pd.__version__
pd.Series(['San Francisco', 'San Jose'])
city_names=pd.Series(["New york", "Halloween","Christmas"])
population=pd.Series([123,543,888])
pd.DataFrame({"City": city_names, 'Size':population})

california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")
california_housing_dataframe.describe()
cities=pd.DataFrame({'Cities': city_names, 'Size':population})
type(cities)
cities['Cities']
cities['Cities'][1]
cities[0:2]#top two rows, non-inclusive
population/1000 #apply arith to Series

np.log(population)
population.apply(lambda val: val>1000000)
cities['Area']=pd.Series([43,87,22])
cities['Density']=cities.Size/cities.Area
cities['Is wide and has saint name'] = (cities['Area'] > 50) & cities['Cities'].apply(lambda name: name.startswith('San'))
cities
cities.index
cities.reindex([1,2,0])#reorder rows
cities.reindex(np.random.permutation(cities.index))#randomize
cities.reindex([1,5,6])#fills nan so when specifiying string indexes, dont have to sanitise input when load index vals from external list


'''
Open source library built on top of numpy
fast analyis, data cleaning, preparation,
built in visualisation features
work data variety sources

conda install pythons
Series, dataframes, missing data, groupbym nerging joining concatenatin, operations, data in/out
'''

#Series-datatype, built on top numpy array, idnexed by label
import numpy as np
import pandas as pd
from numpy.random import randn
labels = ['a', 'b', 'c']
my_data=[10,20,30]
arr=np.array(my_data)
d={'a':10, 'b':20,'c':30}
pd.Series(data = my_data)
pd.Series(data=my_data, index=labels)#indexes are labelled
pd.Series(my_data, labels)
pd.Series(d)
#can also hold any type data object, hold references to functions, cant do with numpy array
pd.Series([sum, print, len])

ser1 = pd.Series([1,2], ["us", "germany"])
ser2=pd.Series([3,4], ['italy', 'japan'])
ser1['us']#know index is string, when index is number pass in number
ser3=pd.Series(data=labels)
ser3[0]

ser1+ser2 # tries to add based on index, when no match between produces NaN
# pandas and numpy tends to convert things to floats

'''Dataframes'''
np.random.seed(101)# set seed so replicable
df = pd.DataFrame(randn(5,4), ['A', 'B', 'C'], ['X', 'Y', 'Z'])# data, inexes, cols
#dataframe is collection series that share indexes. 
df['Y']# w col, just a series
type(df['Y'])
df.W #also works but not as preferable, as may get overwritten by inbuilt function
df[['Y','Z']]#get dataframe

df['new']=df['Y']+df['Z']#create new
df.drop('new', axis=1)#axis=1 cols, default drop via row indexes
df.drop('new', axis=1, inplace=True)#to perm removed, safeguard to stop accidental removal
df.drop('A')
df.shape # rows are axis 0, cols are 1 as iherited from numpy shape structure

df.head(n=10)
df.info #number non-null vals, index, column dtypes

# sal[sal['EmployeeName']=='JOSEPH DRISCOLL']['JobTitle']
# job title of joesph driscoll

#select rows
df.loc['A']#returns series, rows are series as well
df.iloc[0]

#subset rows and cols
df.loc['A', 'Y'] #rows specified ffirst
df.loc[['A', 'B'], ['Z']]
df.loc['A','Y':'Z'] #Can use : for indexing

'''Conditional selection'''
df > 0 # get dataframe boolaen values
df[df>0] #returns NaNs where condition not met
df['Y']>0 #get bool series, can use to filter rows
df[df['Y']>0]#conditional based on series filters rows
df[df['Z']<0] # get dataframe result
#can just apply chain selections
df[df['Y']>0][['Y', 'Z']]#select cols of filtered dataframe by row


df[(df['Y']>0) and (df['Z']>0)]#Error python and op can only take into account two boolean values, not two series of bools
df[(df['Y']>0) & (df['Z']>0)]
df[(df['Y']>0) | (df['Z']>0)]

df.iloc[:,0]>0 # col 0 entries greater than 0
df.iloc[0,:]>0 # row 0 entries greater than 0

df.loc[:, df.iloc[2,:]>0] # filter cols where row2 entries are not greater than 0

df.loc[:, df.iloc[:,0]>0]
df.loc[:, df[0]>0] #same, filter cols same index as rows in col 0 that are less than 0


#resetting pandas index
df.reset_index()#also creates col called index, with orig labels
df.reset_index(inplace=True)

newind= 'CA NY WY OR CO'.split()
df['States']= newind
df.set_index('States')#still needs inplace to make long impact

''' MULTI INDEXING '''

outside = ['G1', 'G1', 'G1', 'G2', 'G2', 'G2']
inside = [1,2,3,1,2,3]
hier_index = list(zip(outside, inside))
hier_index = pd.MultiIndex.from_tuples(hier_index)
#takes list creates multiindex with multiple labels
#index heirarchy

df = pd.DataFrame(randn(6,2), hier_index, ['A', 'B'])
df.loc['G1']
df.loc['G1'].loc[1]
df.index.names ## indexes currently dont have names
df.index.names=['Groups', 'Num']
df.loc['G1'].loc[1]['A']

df.xs('G1') # returns cross section, ability to skip go inside multiindex
df.xs(1, level='Num')# collect all where level Num =1

''' Missing data '''
#Pandas auto fills in with null/NaN
d= {'A':[1,2,np.nan],'B':[5, np.nan, np.nan], 'C': [1,2,3]}
df = pd.DataFrame(d)

df.dropna() #drops any row with missing values
df.dropna(axis=1)#drop any cols with null vals
df.dropna(thresh=2) # require that many nan vals
df.fillna(value='FILL')
df['A'].fillna(value=df['A'].mean())

sal.groupby('Year')['BasePay'].mean()#basepay per year
sal['JobTitle'].value_counts().iloc[:5] #5 most common jobs
#auto sorts descending

df.fillna(df.mean())#replace with respective col mean
#axis not currently implemented for fillna so need to iterate through
#fillna according to row avg
m = df.mean(axis=1)#average across cols, therefore for entire row
for i, col in enumerate(df):
             # using i allows for duplicate columns
             # inplace *may* not always work here, so IMO the next line is preferred
             # df.iloc[:, i].fillna(m, inplace=True)
             df.iloc[:, i] = df.iloc[:, i].fillna(m)
df.mean().reset_index(drop=True)#get mean across cols, drop col index and reset

''' Group by 
Allow group together rows based off column, 
Aggregate functions - any takes in multiple values, outputs single value
'''

data = {'Company':['GOOG','GOOG','MSFT','MSFT','FB','FB'],
       'Person':['Sam','Charlie','Amy','Vanessa','Carl','Sarah'],
       'Sales':[200,120,340,124,243,350]}

df = pd.DataFrame(data)
byComp=df.groupby('Company')#returns group by object, input col want to groupby, works mostly with numerical cols
byComp.mean() # ignores string cols as none for mean
byComp.sum()
byComp.sum().loc['FB']
df.groupby('Comapny').min() #.max() .count() - works with strings as alphabetical order
byComp.describe()
byComp.describe().transpose()['FB']# returns sales horizontally describe info

''' Merging, concatenating, joining '''

df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                        'B': ['B0', 'B1', 'B2', 'B3'],
                        'C': ['C0', 'C1', 'C2', 'C3'],
                        'D': ['D0', 'D1', 'D2', 'D3']},
                        index=[0, 1, 2, 3])

df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                        'B': ['B4', 'B5', 'B6', 'B7'],
                        'C': ['C4', 'C5', 'C6', 'C7'],
                        'D': ['D4', 'D5', 'D6', 'D7']},
                         index=[4, 5, 6, 7]) 

df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
                        'B': ['B8', 'B9', 'B10', 'B11'],
                        'C': ['C8', 'C9', 'C10', 'C11'],
                        'D': ['D8', 'D9', 'D10', 'D11']},
                        index=[8, 9, 10, 11])


'''
Concatenation glues together dataframes, dimensions should match along axis concatenating on, 

'''
pd.concat([df1, df2, df3]) # default axis 0, join along rows
pd.concat([df1, df2, df3], axis=1)#bunch missing vals as missing for row indexes, 3 lots A B C D cols

left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})
   
right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                          'C': ['C0', 'C1', 'C2', 'C3'],
                          'D': ['D0', 'D1', 'D2', 'D3']})  

# Merge logic, similar to logic merging SQL tables
pd.merge(left, right, how='inner', on='key')#inner is default, on- key column, can pass in more than one, joined on key col they share, instead of just gluing


left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                     'key2': ['K0', 'K1', 'K0', 'K1'],
                        'A': ['A0', 'A1', 'A2', 'A3'],
                        'B': ['B0', 'B1', 'B2', 'B3']})
    
right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                               'key2': ['K0', 'K0', 'K0', 'K0'],
                                  'C': ['C0', 'C1', 'C2', 'C3'],
                                  'D': ['D0', 'D1', 'D2', 'D3']})

# Can merge on multiple cols
pd.merge(left, right, how='outer', on=['key1', 'key2'])
#keeps nan values where no match for inserted vals
pd.merge(left, right, how='right', on=['key1', 'key2'])# left as well
#keeps all of right entries, and left fills in nan where missing

'''
 Joining, combining cols of two potentially differently indexed dataframes
 Join on indexes instead of cols(merge), very similar
'''

left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                     'B': ['B0', 'B1', 'B2']},
                      index=['K0', 'K1', 'K2']) 

right = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
                    'D': ['D0', 'D2', 'D3']},
                      index=['K0', 'K2', 'K3'])

left.join(right) #inner join between left and right
left.join(right, how='outer')

''' OPERATIONS '''

df = pd.DataFrame(
  {'col1':[1,2,3,4],
  'col2':[444,555,666,444],
  'col3':['abc','def','ghi','xyz']})
df.head()

df['col2'].unique()
df['col2'].nunique() #number unique vals len(df['col2'].unique())
df['col2'].value_counts() # table how often each val occurred

#Conditional selection
df[(df['col1']>2) & (df['col2']==444)]

def times2(x):
  return x*2
#How to apply custom function
df['col1'].apply(times2)#broadcast function to each elem in col
df['col1'].apply(len)
df['col1'].apply(lambda x: x*2)#apply own custom lambda expression

df.drop('col1', axis=1, inplace=True)
df.columns # list col names
df.index # list indexes
df.sort_values(by='col2')

df.isnull() #where any elems are null
df.dropna()

df = pd.DataFrame({'col1':[1,2,3,np.nan],
                   'col2':[np.nan,555,666,444],
                   'col3':['abc','def','ghi','xyz']})
df.head()
df.fillna('Fill')

data = {'A':['foo','foo','foo','bar','bar','bar'],
     'B':['one','one','two','two','one','one'],
       'C':['x','y','x','y','x','y'],
       'D':[1,3,2,5,4,1]}

df = pd.DataFrame(data)
df.pivot_table(values='D', index=['A', 'B'], columns='C') #takes in values, indexes, columns
#created multiindex from a,b  , columns become c so x,y, values are d, NaN where no vals exist

''' INPUT AND OUTPUT 
CSV, EXCEL, HTML, SQL 
LOTS OF OPTIONS

conda install sqlalchemy
conda install lxml
conda install html5lib
conda install BeautifulSoup4
conda install xlrd

'''

pwd #check location jupyter notebook currently in
df = pd.read_csv('example.csv')#click tab to auto complete
df.to_csv('My_output', index=False)# save to output, dont save index, so doesn't become an articficial col
#can only import data, not images etc

pd.read_excel('Excel_Sample.xslx', sheetname='Sheet1')

df.to_excel('Excel.xlsx', sheet_name='Newsheet')#when writing to excel, it is sheet_name not sheetname
data=pd.read_html('htmllink')
#usually doesnt relate to dataframe
type(data)#pandas tried to look for all tables on html, convert each table found to dataframe, stored in a list
data[0]#first table on html

# Working ith SQL, not best way to read SQL database like postgres etc
# going to create quick sql engine, search for specific driver, to work with your flavour sql
#this is an example
from sqlalchemy import create_engine
engine = create_engine('sqlite:///:memory:')
df.to_sql('my_table', engine)#engine is usually a connection, 
#pandas by iteself not best way to read sql into pandas, look for specialised python libraries
sqldf = pd.read_sql('my_table', con=engine)


#Count number of jobs that occur only once in 2013
ser=sal[sal['Year']==2013]['JobTitle'].value_counts()
ser[ser==1].count()
#OR!!!
sum(sal[sal['Year']==2013]['JobTitle'].value_counts() ==1)

#correlation between jobtitle length and totalpay
df= pd.concat([sal['JobTitle'].apply(lambda x: len(x)), sal['TotalPayBenefits']], axis=1)
df.corr()

#or create new col
sal['title_len']= sal['JobTitle'].apply(len)
sal[['title_len', 'TotalPayBenefits']].corr()

sal.loc[sal['TotalPayBenefits'].idxmax()] #get highest paid person
sal.iloc[sal['TotalPayBenefits'].argmin()] #get lowest paid person

sal['JobTitle'].value_counts().iloc[:5]
sal['JobTitle'].value_counts().head(n=5)

def chief_string(title):
    if 'chief' in title.lower().split():
        return True
    else:
        return False

sum(sal['JobTitle'].apply(lambda x: chief_string(x)))

len(sal[(sal['JobTitle'].str.contains("Chief"))])
sum(sal['JobTitle'].str.contains("Chief"))

#top 5 email hosts
df['Email'].apply(lambda x: x.split("@")[1]).value_counts().head()
#cards that expire 2025
def expire25(dt):
    if dt.split("/")[1]=="25":
        return True
    else:
        return False
    
sum(df['CC Exp Date'].apply(lambda x: expire25(x)))

sum(df['CC Exp Date'].apply(lambda x: e[3:]=='25'))
df[df['CC Exp Date'].apply(lambda x: e[3:]=='25')].count()

len(df.columns)
len(df.index)

#count people language = 'en'
sum(df['Language']=='en')
df[df['Language']=='en'].count()['Language']
len(df[df['Language']=='en'].index)

len(df[(df["CC Provider"]=='American Express') & (df['Purchase Price']>95)])
df[(df["CC Provider"]=='American Express') & (df['Purchase Price']>95)].count()






















 
















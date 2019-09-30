'''
  Data analysis library
  linear algebra library, all other libraries rely on umpy,
  conda install numpy
  numpy arrays: vectors 1-d arrays, matrics 2-d arrays
'''

#Numpy arrays
import numpy as np
import random as rd

lst=[1,2,3]
arr = np.array(lst) #casted
my_mat=[[1,2],[3,4]]
np.array(my_mat) # numpy matrix

np.arange(0,10)#start, stop, step
np.zeros(3)#shape
np.zeros((2,3))
np.ones((3,4))
np.linspace(0,5,10)#start, stop, number points
np.eye(4)#identity matrix, diagonal of ones

np.random.rand(5)#creates rand away from uniform distr 0->1
np.random.rand(5,5)
np.random.randn(2)#standard normal distribution centered 0
# array([-0.12, 0.27])
np.random.randint(1,100)#random range, 1 inclusize, 100 exclusive
np.random.poisson(lam=rd.randrange(10,1000), size=5)

arr = np.arange(25)
ranarr= np.random.randint(0,50,10)

arr.reshape(5,5)# reshape 5 by 5, error if cant fill array completely
ranarr.max() #.min()
ranarr.shape
ranarr.argmax()#index of max value argmin
arr.dtype # type array

from numpy.random import randint
randint(2,10)

''' Array indexing '''
arr = np.arange(0,11)#0-10
arr[8]
arr[1:5]#up to not including index 5, slice notation
arr[:-1]
arr[0:6]
arr[5:]# 5 to end
arr[0:5]=100 # broadcasts value to all elements
arr=np.arange(0,11)
slice_of_arr=arr[0:6]#creates just a view of arr
slice_of_arr[:]=99 # this changes original arr as well, broadcasting 99

#done to avoid huge memory use, to create new, not just referential copy, 
arr_copy=arr.copy()
arr_2d = np.array([[5,10], [35,40]])
#double bracket and single bracket format with comma to select items
arr_2d[0][0]
arr_2d[0,0]#recommended
arr_2d[:2, 1:]

#conditional selection
arr=np.arange(1,11)
bool_arr = arr>5 #get array boolean values
arr[bool_arr] # return array items greater than 5, conditional selection
arr[arr>5]
arr_2d=np.arange(50).reshape(5,10)

'''
Numpy operations
'''
arr = np.arange(0,11)
arr+ arr #adds each element,same multiplication
arr + 1 #broadcasts 1 to each element array to add
1/0 #get python error
arr/arr #even if 0/0 only get warning, returns nan in element, still get output

1/arr # inf when element nan, 
arr**2 #array to power 2

# Universal array functions
#LOOK UP ALL UNIVERSAL FUNCTIONS
np.sqrt(arr) # sqrt everthing in arr
np.exp(arr)
arr.max()
np.sin(arr)
np.log(arr)

np.zeros(10)+5
np.ones(10)*5
np.repeat(5,10) # array of 10 5's
np.arange(10,51,2)
np.arange(9)# 0-> 8

mat = np.arange(1,26).reshape(5,5)

mat[3,-1:] # last elem in row
mat[3, :-1] #all but last elem in row
mat.reshape(5)#converts to row
np.sum(mat)
np.sum(mat, axis=0) # sum of all the cols, 0 across columns down, 1, along columns across
np.std(mat)#standard deviation

np.random.rand(1)#random 0-> 1

np.arange(1,101).reshape(10,10)/100
np.arange(0.01,1.01, 0.01).reshape(10,10)
np.linspace(0.01,1,100).reshape(10,10)
mat[3,4]#single element

mat[:3,1:2]#keeps in 2d array 
mat[:3,1] # reeturns row
mat[-1,:]#last row
np.sum(mat)
mat.sum()
mat.std()
np.std(mat)

np.nan #nan val
np.pi

# Mesh grid - way to generate grid from two arrays (useful for matplots)
phi_m = np.linspace(0, 2*np.pi, 100)
phi_p = np.linspace(0, 2*np.pi, 100)
X,Y = np.meshgrid(phi_p, phi_m)








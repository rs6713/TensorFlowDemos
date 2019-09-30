import random as rd
seq=[1,2,3,4,5]
a=2
b=5
population=seq
k=2
x=seq
start=0
stop=10
step=2

rd.choice(seq)#picks random from seq
rd.randint(a,b) # rand int a<=x<=b
rd.shuffle(x)#shuffles in place
rd.sample(population, k)#returns k length list unique elements chosen from pop seq,no replacement
rd.random() # next rand floating point 0--> 1
rd.uniform(a,b) #return random float number a<=x<=b
rd.randrange(start, stop,step)#return random selected element from range
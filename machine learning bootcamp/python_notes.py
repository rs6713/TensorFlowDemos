# python 2.7
num, name=3,4
print('My number is: {one}, and my name is: {two}'.format(one=num,two=name))
#tuples are immutable
{1,2,3}
{1,2,3,4,4,4}# sets will become {1,2,3}
2**2 # square

list(map(lambda var: var*2,[1,2]))
"HI".lower() #.upper()
'x' in [1,2,3] #false

#count occurences dog in string
"hello dog".count("dog")
len("hello dog".split("dog"))-1 # ["hello",""]

def caught_speeding(speed, is_birthday):
    if speed<=60+is_birthday*5:
        return "No Ticket"
    elif (61+is_birthday*5)<=speed<=(80+is_birthday*5):
        return "Small Ticket"
    else:
        return "Big Ticket"
    pass

'''
documentation string
'''
lst=[1,2,3]
lst.pop() #.pop(0)

x=[(1,2),(3,4)]
#tuple unpacking inside for loop
for (a,b) in x:
  print a
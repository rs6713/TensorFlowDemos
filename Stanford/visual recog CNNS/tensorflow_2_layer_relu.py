import numpy as np 
import tensorflow as tf 

# Code defines numerical graph
N, D, H = 64, 1000, 100 # 64 examples, 1000 features, 
# create placeholders, entry points to graph input data
x = tf.placeholder(tf.float32, shape=(N,D))#first is number rows, then cols
y = tf.placeholder(tf.float32, shape=(N,D))
w1 = tf.placeholder(tf.float32, shape=(D,H))
w2 = tf.placeholder(tf.float32, shape=(H,D))

# perform tensorflow functions forward pass compute y prediction, loss

h = tf.maximum(tf.matmul(x,w1),0)# produces N,H
y_pred= tf.matmul(h, w2) # produc  es N, D
diff = y_pred -y
#l2 euclidean loss
loss = tf.reduce_mean(tf.reduce_sum( diff ** 2, axis =1)) # y, y_pred are series examples, calculate mean loss across batch

# Ask tensorflow to compute the gradients in respect to loss for inputs
grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])

#run the graph many times
#eenter tensorflow session
with tf.Session() as sess:
  #construct numpy arrays to feed in data
  values = {
    x: np.random.randn(N,D),
    w1: np.random.randn(D,H),
    w2: np.random.randn(H,D),
    y: np.random.randn(N,D),
  }
  #run the graph, tells what we want as output, then pass in inputs
  out = sess.run([loss, grad_w1, grad_w2], feed_dict=values)
  loss_val, grad_w1_val, grad_w2_val = out
  # this was only one forward pass could do instead

  learning_rate = 1e-5
  #train network run over and over to update weights
  for t in range(50):
    out = sess.run([loss, grad_w1, grad_w2],
                feed_dict = values)
    loss_val, grad_w1_val, grad_w2_val = out
    # SGD
    values[w1] -= learning_rate * grad_w1_val
    values[w2] -= learning_rate * grad_w2_val

    #can plot loss see goes down.
    # Problem every time run graph copying huge numpy rows into tensorflow and backout
    # GPU bottleneck to copy data

# Want to change w1, w2 from placeholders to variables (persist in graph between calls)
# now live inside graph, tells how to be initialised.
w1 = tf.Variable(tf.random_normal((D,H)))
w2 = tf.Variable(tf.random_normal((H,D)))

#want to mutate within graph, assign function, mutated persists over multiple runs 
learning_rate = 1e-5
new_w1= w1.assign(w1 - learning_rate * grad_w1)
new_w2 = w2.assign(w2 - learning_rate * grad_w2)
#solve bug
updates = tf.group(new_w1, new_w2) #returns tensorflow node operation val, when tensorflow told wants to compute concrete val during running returns none

with tf.Session() as sess:
  #set up variables
  sess.run(tf.global_variables_initializer())
  values= {x:np.random.randn(N,D), y: np.random.randn(N,D),}
  for t in range(50):
    #compute loss for us
    # need to tell tensorflow we want to calculate new w1, w2 variables
    loss_val, = sess.run([loss], feed_dict=values)# bug no loss improvement
    loss_val,_ = sess.run([loss, updates], feed_dict=values)# bug no loss improvement

    #dependencies to compute loss dont need to update w1,w2, 
    # could include w1,w2 as outputs to force calculation but then would copy, bottleneck
    # so create dummy node, that depends on w1,w2 updates, then compute dummy node
    # dummy node returns none so no copying. 
    # as x,y the same could have kept in graph as variables
    #but realistically want to feed in using mini batches


# Has optimizer to simplify process to compute gradients
optimizer = tf.train.GradientDescentOptimiser(1e-5)#can specify different gradient methods
updates = optimizer.minimize(loss)#compute gradients and update weights, 

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  values = {
    x: np.random.randn(N,D),
    y: np.random.randn(N,D),
  }
  losses = []
  for t in range(50):
    loss_val, _ = sess.run([loss, updates], feed_dict = values)

# tensorflow provides convenience functions e.g for loss
loss = tf.losses.mean_squared_error(y_pred, y)
# bunch higher level libraries wrap around tensorflow to provide convolution abilities, matrix dot multiplication+bias

#xavier intiializer
init= tf.contrib.layers.xavier.initializer()# reasonable strategy for initialisation
# tf.layers auto sets up weight adn bias 
# sets up weights of right shape, xavier tells init function, does relu inside
h = init.layers.dense(inputs=x, units=H, acivation=tf.nn.relu, kernel_initializer=init)
y_pred = tf.layers.dense(inputs=h, units=D, kernel_initializer=init)

loss= tf.losses.mean_squared_error(y_pred, y)
optimizer = tf.train.GradientDescentOptimizer(1e0)
updates = optimizer.minimize(loss)
#then run session as before



#########
#KERAS layer on top of tensorflow, more common east things toldo 
#########

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD

#define model sequence layers
N,D,H = 64,1000,100
model= Sequential()
model.add(Dense(input_dim=D, output_dim=H))
model.add(Activation('relu'))
model.add(Dense(input_dim=H, output_dim=D))

optimizer=SGD(lr=1e0)
#compile model
model.compile(loss='mean_squared_error', optimizer=optimizer)

x = np.random.randn(N,D)
y = np.random.randn(N,D)
#train model single line
history = model.fit(x,y, nb_epoch=50, batch_size=N, verbose=0)

#Huge number of tensorflow higher level wrappers, tf.layers, sonnet deepmind, tf.contrib.learn
# tensorboard can plot losses, see strucuture computational graph
# lets run distributed
# tensorflow derived from tensorflow

# tensorflow build graph, then run iteratively, static computational graph












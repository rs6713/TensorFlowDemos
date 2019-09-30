import tensorflow as tf 
import numpy as np 

# y is w1*x if z>0 else w2*x

N,D,H =3,4,5
x = tf.placeholder(tf.float32, shape=(N,D))
z = tf.placeholder(tf.float32, shape=None)
w1 = tf.placeholder(tf.float32, shape=(D,H))
w2 = tf.placeholder(tf.float32, shape=(D,H))

def f1(): return tf.matmul(x,w1)
def f2(): return tf.matmul(x,w2)
y = tf.cond(tf.less(z,0),f1,f2)

with tf.Session() as sess:
  values = {
    x: np.random.randn(N,D),
    z: 10,
    w1: np.random.randn(D,H),
    w2: np.random.randn(D,H),
  }
  y_val = sess.run(y, feed_dict=values)


T,N,D = 3,4,5
x = tf.placeholder(tf.float32, shape=(T,D))
y0 = tf.placeholder(tf.float32, shape=(D,))
w = tf.placeholder(tf.float32, shape=(D,))

def f(prev_y, cur_x):
  return (prev_y + cur_x) * w

# yt = (y(t-1) + xt)*w
y = tf.foldl(f,x,y0)
#tensorflow fold makes dynamic graphs easier in tensorflow via dynamic matches

with tf.Session() as sess:
  values = {
    x: np.random.randn(T,D),
    y0: np.random.randn(D),
    w: np.random.randn(D),
  }
  y_val = sess.run(u, feed_dict = values)
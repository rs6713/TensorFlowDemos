f = lambda x: 1.0/(1.0 + np.exp(-x))
x = np.random.randn(3,1)
h1= f(np.dot(W1,x)+b1)
h2= f(np.dot(W2, h1)+b2)
out=np.dot(W3,h2)+b3

# Preprocess data
X -=np.mean(X,axis=0)
X /= np.std(X, axis=0)

# Weight initialisation
W=0.01*np.random.randn(D,H)
#xavier init
W= np.random.randn(fan_in, fan_out)/ np.sqrt(fan_in)

# RATIO OF WEIGHT UPDATES, weight magnitudes
#assume param vector W and gradient vector dW

param_scale= np.linalg.norm(W.ravel())
update=-learning)rate*dW 
update_scale = np.linalg.norm(update.ravel())
W+= update
print update_scale/param_scale

#rho gives friction
vx = 0
while True:
  dx = compute_gradient(x)
  vx = rhho * vx + dx
  x += learning_rate * vx

while True:
  dx = compute_gradient(x)
  x+= learning_rate*dx

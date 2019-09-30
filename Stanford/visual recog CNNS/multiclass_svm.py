import numpy as np 

def L_I_vectorized(x,y,W):
  scores= W.dot(x)
  margins = np.maximum(0, scores[y]+1)
  margins[y]=0
  loss_i=np.sum(margins)
  return loss_i
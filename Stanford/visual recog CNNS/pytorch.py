'''
create own modules for models
then create training loops that call model.forward() passes, compute loss, optimize internals
pytorch switches between gpu and cpu by changing data types 
dataloader- wrap dataset provides minibatching, shuffling , threading
provides pretrained models
package visdom allows visualize loss statistics
evolution of torch

python, autograd, fast, but less existing code, changing rapidly

Each forward pass defines new graph (dynamic)
'''
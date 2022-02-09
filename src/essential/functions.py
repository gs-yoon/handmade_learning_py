import numpy as np

def step(x):
    y = x > 0
    return y.astype(np.int)

def sigmoid(x):
    return 1 / ( 1 + np.exp(-x))

def relu(x):
    return np.maximum(0,x)

def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x - c)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y

def sumSquareError(y, t):
    return 0.5 * np.sum((y-t)**2)
    
def crossEntropy(y, t, one_hot = True):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    if one_hot == True:
        return -np.sum(t * np.log(y + 1e-7)) / batch_size
    else:
        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


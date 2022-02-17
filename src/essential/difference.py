import numpy as np

def numericalDiff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2*h)

def numericalGrad(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 値を元に戻す
        it.iternext()   
        
    return grad

def gradientDescent(f, init_x, lr=0.01, step_num = 100):
    x = init_x

    for i in range(step_num):
        grad = numericalGrad(f,x)
        x -= lr * grad
    return x

def function_2(x):
    return x[0]**2 + x[1]**2

def function_tmp2(x1):
    return 3.0**2.0 + x1*x1

import numpy as np

def numericalDiff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2*h)

def numericalGrad(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        x_tmp = x[idx]
        x[idx] = x_tmp + h
        fx_add_h = f(x)

        x[idx] = x_tmp - h
        fx_sub_h = f(x)

        grad[idx] = (fx_add_h - fx_sub_h) / (2*h)
        x[idx] = x_tmp 

    return grad
s
def function_2(x):
    return x[0]**2 + x[1]**2

def function_tmp2(x1):
    return 3.0**2.0 + x1*x1

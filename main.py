import numpy as np

x_inicial = np.array([2, 4])
theta = 0.25
alpha = 0.005

def f(x):
    return 1 + x[0] ** 2 + x[1] ** 2 + 2 * x[0]

def dfx(x):
    return 2 * x[0] + 2

def dfy(x):
    return 2 * x[1]

def grad(x):
    return np.array([dfx(x), dfy(x)])

def g(x):
    print(x - theta * grad(x))
    return f(x - theta * grad(x))

def armijo(x):
    gradx = grad(x)
    x_k = x - theta * gradx
    n = 1
    while f(x_k) >= f(x) + alpha * np.dot(gradx.T, x_k * theta):
        x_k = x - theta / n * gradx
        n = n + 1
    return x_k

def main():
    x = x_inicial

    n = 0
    while n < 10:
        x = armijo(x)
        n = n + 1
    return x

print(main())
import numpy as np 
import matplotlib.pyplot as plt 
lr = .1

def ham(x):
    return x**2 - 5 * x + 2

X = np.arange(-100, 100, .2)
Y = ham(X)

plt.plot(X, Y)
# plt.show()

def dao_ham(x):
    return 2 * x - 5

def GD(x_init, dao_ham, lr):
    x = [x_init]
    for it in range(100):
        x_new = x[-1] - lr*dao_ham(x[-1])
        if np.abs(dao_ham(x_new)) < 1e-6:
            break
        x.append(x_new)
        if it%10 == 0:
            print("Sau %d vong lap tim duoc x = %f" %(it, x[-1]))
    return it, x

x_init = 0
it, x_op = GD(x_init, dao_ham, lr)
print("Optimize x = %f sau %d vong lap!" %(x_op[-1], it))

y_op = [ham(_) for _ in x_op]
plt.plot(x_op, y_op, 'r.')
plt.show()
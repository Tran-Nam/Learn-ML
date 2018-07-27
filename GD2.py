import numpy as np 
import matplotlib.pyplot as plt 
lr = .01

def ham(x1, x2):
    return (x2 - 2*x1 + 3)**2 + (3*x2 + x1 - 4)**2


def gradx1(x1, x2):
    return 2*(x2 - 2*x1 + 3)*(-2) + 2*(3*x2 + x1 -4)

def gradx2(x1, x2):
    return 2*(x2 - 2*x1 + 3) + 2*(3*x2 + x1 - 4)*3

def GD2(x1_init, x2_init, gradx1, gradx2, lr):
    x1 = [x1_init]
    x2 = [x2_init]
    for it in range(100):
        x1_new = x1[-1] - lr*gradx1(x1[-1], x2[-1])
        x2_new = x2[-1] - lr*gradx2(x1[-1], x2[-1])
        if np.abs(gradx1(x1_new, x2_new)) < 1e-6 \
            and np.abs(gradx2(x1_new, x2_new)) < 1e-6:
            break
        x1.append(x1_new)
        x2.append(x2_new)
        if it%10 == 0:
            print("Sau %d vong lap tim duoc x1, x2, y = %f, %f, %f " %(it, x1[-1], x2[-1], ham(x1[-1], x2[-1])))
    return it, x1, x2
 
X1 = np.arange(-100, 100, .2)
X2 = np.arange(-100, 100, .2)
Y = ham(X1, X2)
# print(Y)

from mpl_toolkits import mplot3d

fig = plt.figure()
ax = plt.axes(projection='3d')
# ax.scatter3D(X1, X2, Y, c=Y, cmap="Greens")
# plt.show()

x1_init = 0
x2_init = 0
it, x1, x2 = GD2(x1_init, x2_init, gradx1, gradx2, lr)

print("Optimize x1, x2, y = %f, %f, %f sau %d vong lap" %(x1[-1], x2[-1], ham(x1[-1], x2[-1]), it))

y_op = [ham(x1[i], x2[i]) for i in range(len(x1))]
print(y_op)
ax.scatter3D(x1, x2, y_op, c=y_op, cmap="Reds")

print(ham(1.85, 0.7))
plt.show()



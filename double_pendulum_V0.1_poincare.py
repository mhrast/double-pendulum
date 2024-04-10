import matplotlib.pyplot as plt
import numpy as np

epsilon = 0.01                      # Epsilon: discrete time interval
steps = 10000000
delta = 0.0001                      # delta for angle query abs(theta1)<delta, then theta2=0

y = np.array([0.0, 0.0, 2.0, 2.0])  # initial values for theta1, theta2, p1, p2

m = 1                               # mass of mass points
L = 1                               # distance between mass points
g = 9.81                            # gravitational constant

k1 = np.zeros(4)                    # 4 k vectors for RK4
k2 = np.zeros(4)
k3 = np.zeros(4)
k4 = np.zeros(4)

t = np.empty(steps)                 # time
E = np.empty(steps)                 # energy
theta1 = np.empty(steps)            # angles
theta2 = np.empty(steps)
p1 = np.empty(steps)                # momenta
p2 = np.empty(steps)


def f(y):                           # equations of motion for double pendulum: 4 coordinates per time step
    y1 = y[0]
    y2 = y[1]
    y3 = y[2]
    y4 = y[3]
    
    c1 = 1 / (m * L**2)
    c2 = 1 + np.sin(y1-y2)**2
    c3 = y3**2 + 2 * y4**2 - 2 * y3 * y4 * np.cos(y1-y2)
    
    f0 = c1 * (y3 -y4 * np.cos(y1-y2)) / c2
    f1 = c1 * (2 * y4 - y3 * np.cos(y1-y2)) / c2
    f2 = -c1 * (y3 * y4 * np.sin(y1-y2)) / c2 + c1 * c3 / c2**2 * np.sin(y1-y2) * np.cos(y1-y2) - 2 * m * g * L *np.sin(y1)
    f3 = c1 * (y3 * y4 * np.sin(y1-y2)) / c2 - c1 * c3 / c2**2 * np.sin(y1-y2) * np.cos(y1-y2) - m * g * L * np.sin(y2)
    
    return np.array([f0, f1, f2, f3])


t[0] = 0                           # initial values
theta1[0] = y[0]
theta2[0] = y[1]
p1[0] = y[2]
p2[0] = y[3]

print("------------------------------------------------")
print("Double pendulum with RK4 V0.1 - Poincaré conjectures")
print("------------------------------------------------\n")
print("Initial values:", y)


fig = plt.figure()                 # plot initialization
ax = fig.add_subplot(1,1,1)
ax.set_title(f'Poincaré-Schnitt Doppelpendel [{theta1[0]}, {theta2[0]}, {p1[0]}, {p2[0]}]')
ax.set_xlabel(r'$\theta_{1}$')
ax.set_ylabel(r'$p_{1}$')


for i in range(1,steps):           # RK4 algorithm
    it = i * epsilon
    
    k1 = f(y)
    k2 = f(y + 0.5 * epsilon * k1)
    k3 = f(y + 0.5 * epsilon * k2)
    k4 = f(y + epsilon * k3)
    
    y = y + epsilon * (k1 + 2.0 * k2 + 2.0 * k3 + k4)/6.0
    
    angle = (np.abs(y[1]) / (2*np.pi)) % 1    # Angle theta2 at first decimal point
    
    if angle < delta and y[3] > 0:            # if theta2 less than delta and p2 more than 0:
        ax.plot(y[0],y[2],'.',color='blue')   # then plot theta1 and p1

plt.show()                                    # 1D curve is called regular motion, 2D area is called chaotic behavior

print('Done')
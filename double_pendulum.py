import matplotlib.pyplot as plt
import numpy as np

epsilon = 0.01                      # discrete time interval
steps = 10000

y = np.array([0.0, 0.0, 2.0, 2.0]) # initial values for theta_1, theta_2, p_1 and p_2

m = 1                               # mass of mass points
L = 1                               # distance between mass points
g = 9.81                            # gravitational constant

k1 = np.zeros(4)                    # the 4 k vectors for Runge-Kutta 4
k2 = np.zeros(4)
k3 = np.zeros(4)
k4 = np.zeros(4)

t = np.empty(steps)                 # time
E = np.empty(steps)                 # energy
theta1 = np.empty(steps)            # angles
theta2 = np.empty(steps)
p1 = np.empty(steps)                # momentum
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

def energy(y):                    # total energy T + U
    y1 = y[0]
    y2 = y[1]
    y3 = y[2]
    y4 = y[3]
    
    U = m * g * L * (3 - 2 * np.cos(y1) - np.cos(y2))
    c = (2 * m * L**2 * (1 + np.sin(y1-y2)**2))
    e = (y3**2 + 2 * y4**2 - 2 * y3 * y4 * np.cos(y1 - y2)) / c + U  # H = T + U
   
    return e


t[0] = 0                           # initial values
E[0] = energy(y)
theta1[0] = y[0]
theta2[0] = y[1]
p1[0] = y[2]
p2[0] = y[3]

print("------------------------------------------------")
print("Double pendulum with RK4 algorithm V0.1")
print("------------------------------------------------\n")
print("initial values:", y)


for i in range(1,steps):           # RK4 algorithm
    it = i * epsilon
    
    k1 = f(y)
    k2 = f(y + 0.5 * epsilon * k1)
    k3 = f(y + 0.5 * epsilon * k2)
    k4 = f(y + epsilon * k3)
    
    y += epsilon * (k1 + 2.0 * k2 + 2.0 * k3 + k4)/6.0
    
    t[i] = it
    theta1[i] = y[0]
    theta2[i] = y[1]
    p1[i] = y[2]
    p2[i] = y[3]
    E[i] = energy(y)

    
fig1 = plt.figure()                # plots of trajectories (theta1,theta2), (p1,p2) and energy
fig2 = plt.figure()
fig3 = plt.figure()

axtheta = fig1.add_subplot(1,1,1)
axtheta.set_title("Double pendulum - angles")
axtheta.plot(theta1,theta2)
axtheta.set_xlabel(r'$\theta_{1}$')
axtheta.set_ylabel(r'$\theta_{2}$')

axP = fig2.add_subplot(1,1,1)
axP.set_title("Double pendulum - momentum")
axP.plot(p1,p2)
axP.set_xlabel(r'$p_{1}$')
axP.set_ylabel(r'$p_{2}$')

axE = fig3.add_subplot(1,1,1)
axE.set_title("Double pendulum - energy")
axE.plot(t,E)
axE.set_xlabel('t')
axE.set_ylabel('E')

plt.show()

print('Done')
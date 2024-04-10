import matplotlib.pyplot as plt
import numpy as np

epsilon = 0.01                          # Epsilon: discrete time interval
steps = 10000

y1 = np.array([0.0, 0.0, 2.0, 2.0])     # initial values for theta1, theta2, p1, p2
y2 = np.array([0.0, 0.0, 2.0, 2.001])   # slightly changed initial values

m = 1                                   # mass of mass points
L = 1                                   # distance between mass points
g = 9.81                                # gravitational constant
    
k1_1 = np.zeros(4)            # k values for Runge-Kutta for first trajectory
k2_1 = np.zeros(4)
k3_1 = np.zeros(4)
k4_1 = np.zeros(4)

k1_2 = np.zeros(4)            # k values for Runge-Kutta of second trajectory
k2_2 = np.zeros(4)
k3_2 = np.zeros(4)
k4_2 = np.zeros(4)

t = np.empty(steps)           # time
E = np.empty(steps)           # energy

theta1_1 = np.empty(steps)    # coordinates of first trajectory (angles, momenta)
theta2_1 = np.empty(steps)
p1_1 = np.empty(steps)
p2_1 = np.empty(steps)

theta1_2 = np.empty(steps)    # coordinates of second trajectory (angles, momenta)
theta2_2 = np.empty(steps)
p1_2 = np.empty(steps)
p2_2 = np.empty(steps)

def f(y):
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


t[0] = 0                      # initial values

theta1_1[0] = y1[0]
theta2_1[0] = y1[1]
p1_1[0] = y1[2]
p2_1[0] = y1[3]

theta1_2[0] = y2[0]
theta2_2[0] = y2[1]
p1_2[0] = y2[2]
p2_2[0] = y2[3]


print("------------------------------------------------")
print("Double pendulum with RK4 V0.1 - stability analysis")
print("------------------------------------------------\n")
print("Initial values:", y1, y2)


for i in range(1,steps):      # RK4 algorithm for two trajectories
    it = i * epsilon
    
    k1_1 = f(y1)
    k2_1 = f(y1 + 0.5 * epsilon * k1_1)
    k3_1 = f(y1 + 0.5 * epsilon * k2_1)
    k4_1 = f(y1 + epsilon * k3_1)
    
    k1_2 = f(y2)
    k2_2 = f(y2 + 0.5 * epsilon * k1_2)
    k3_2 = f(y2 + 0.5 * epsilon * k2_2)
    k4_2 = f(y2 + epsilon * k3_2)
    
    y1 += epsilon * (k1_1 + 2.0 * k2_1 + 2.0 * k3_1 + k4_1)/6.0
    y2 += epsilon * (k1_2 + 2.0 * k2_2 + 2.0 * k3_2 + k4_2)/6.0

    t[i] = it
    
    theta1_1[i] = y1[0]
    theta2_1[i] = y1[1]
    p1_1[i] = y1[2]
    p2_1[i] = y1[3]
    
    theta1_2[i] = y2[0]
    theta2_2[i] = y2[1]
    p1_2[i] = y2[2]
    p2_2[i] = y2[3]

delta = ((theta1_1-theta1_2)**2 + (theta2_1-theta2_2)**2 + (p1_1 - p1_2)**2 + (p2_1 - p2_2)**2)**0.5 # distance delta in phase space
fig = plt.figure()                                                                                   # plot of distance against time
ax = fig.add_subplot(1,1,1)
ax.set_title(f'Stabilitätsanalyse: [{theta1_1[0]},{theta2_1[0]},{p1_1[0]},{p2_1[0]}] - [{theta1_2[0]},{theta2_2[0]},{p1_2[0]},{p2_2[0]}]')
ax.set_xlabel('t')
ax.set_ylabel(r'$\delta$(t)')
ax.plot(t,delta)

plt.show()

print('Done')
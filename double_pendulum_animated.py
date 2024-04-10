import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


epsilon = 0.01                      # epsilon: discrete timeinterval in s
steps = 1000                        # maximum steps

m1= 1                               # mass [kg]
m2 = 1

l1 = 1                               # length [m]
l2 = 1

g = 9.81                            # gravity [m/s^2]

y = np.array([0., 0., 10., 0.])    # initial values theta1 [rad], theta2 [rad], p1 [m/s], p2 [m/s]

k1 = np.zeros(4)                    # four k vectors for RK4
k2 = np.zeros(4)
k3 = np.zeros(4)
k4 = np.zeros(4)

t = np.empty(steps)                 # time
E = np.empty(steps)                 # energy
theta1 = np.empty(steps)            # angles
theta2 = np.empty(steps)
p1 = np.empty(steps)                # momenta
p2 = np.empty(steps)


def calculate_df(y):                           # motion equations of double pendulum: four coordinates per time step
    """Calculate the DP's coordinate changes (theta1, theta2, p1, p2) per time step.
    
    Parameters:
    -----------
    y : array-like
        array of the values [theta1, theta2, p1, p2]
    
    Returns:
    --------
    q : ndarray
        the corresponding changes [dtheta1, dtheta2, dp1, dp2]
    """
    a1 = y[0]  # theta1
    a2 = y[1]  # theta2
    p1 = y[2]
    p2 = y[3]

    C1 = (p1 * p2 * np.sin(a1 - a2)) \
        / (l1 * l2 * (m1 + m2 * np.sin(a1 - a2)**2))
    C2 = (p1**2 * m2 * l2**2 \
          - 2 * p1 * p2 * m2 * l1 * l2 * np.cos(a1 - a2) \
          + p2**2 * (m1 + m2) * l1**2) * np.sin(2 * (a1 - a2)) \
        / (2 * l1**2 * l2**2 * (m1 + m2 * np.sin(a1 - a2)**2))


    da1 = (p1 * l2 - p2 * l1 * np.cos(a1 - a2)) \
        / (l1**2 * l2 * (m1 + m2 * np.sin(a1 - a2)**2))
    da2 = (p2 * (m1 + m2) * l1 - p1 * m2 * l2 * np.cos(a1 - a2)) \
        / (m2 * l1 * l2**2 * (m1 + m2 * np.sin(a1 - a2)**2))

    dp1 = - (m1 + m2) * g * l1 * np.sin(a1) - C1 + C2

    dp2 = - m2 * g * l2 * np.sin(a2) + C1 - C2

    return np.array([da1, da2, dp1, dp2])


def calculate_energy(y):  # Hamiltonian H = T + U
    """Calculate the DP's total energy E = E_kin + E_pot.
    """
    a1 = y[0]
    a2 = y[1]
    p1 = y[2]
    p2 = y[3]
    
    U = - (m1 + m2) * g * l1 * np.cos(a1) - m2 * g * l2 * np.cos(a2)

    t1 = m2 * l2**2 * p1**2 + (m1 + m2) * l1**2 * p2**2 \
      - 2 * m2 * l1 * l2 * p1 * p2 * np.cos(a1 - a2)
    
    t2 = 2 * m2 * l1**2 * l2**2 * (m1 + m2 * np.sin(a1 - a2)**2)

    T = t1 / t2

    E = T + U

    return E


t[0] = 0                           # initial values
E[0] = calculate_energy(y)
theta1[0] = y[0]
theta2[0] = y[1]
p1[0] = y[2]
p2[0] = y[3]

print("------------------------------------------------")
print("double pendulum using RK4 method V1.0")
print("------------------------------------------------\n")
print("initial values:", y)


for i in range(1,steps):           # Runge-Kutta 4 method
    it = i * epsilon
    
    k1 = calculate_df(y)
    k2 = calculate_df(y + 0.5 * epsilon * k1)
    k3 = calculate_df(y + 0.5 * epsilon * k2)
    k4 = calculate_df(y + epsilon * k3)
    
    y += epsilon * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    
    t[i] = it

    theta1[i] = y[0]
    theta2[i] = y[1]
    p1[i] = y[2]
    p2[i] = y[3]

    E[i] = calculate_energy(y)
   

# animation set-up


fig = plt.figure(dpi = 150)                  

ax = fig.add_subplot(1, 1, 1)
ax.set_title('double pendulum (RK4)')
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_xlim(-(l1 + l2) - 1, (l1 + l2) + 1)
ax.set_ylim(-(l1 + l2) - 1, (l1 + l2) + 1)
ax.set_aspect('equal')
ax.grid()

plot, = ax.plot([], [])

x1 = l1 * np.sin(theta1[0])
z1 = -l1 * np.cos(theta1[0])
x2 = x1 + l2 * np.sin(theta2[0])
z2 = z1 - l2 * np.cos(theta2[0])

text_t = ax.text(0.01, 0.01, '', transform=ax.transAxes)
text_E0 = ax.text(0.7, 0.01, f'$E_{0}$ = {E[0]:7.4f} J', transform=ax.transAxes)
text_E = ax.text(0.718, 0.05, '', transform=ax.transAxes)
text_copyright = ax.text(1.01, 0.0, '\u00a9'+'mhrast', transform=ax.transAxes)

text_a1 = ax.text(0.0, 0.96, r'$\theta_{1}$'+f'= {theta1[0]:3.2f} rad', size=9, transform=ax.transAxes)
text_a2 = ax.text(0.24, 0.96, r'$\theta_{2}$'+f'= {theta2[0]:3.2f} rad', size=9, transform=ax.transAxes)
text_p1 = ax.text(0.48, 0.96, f'$p_{1}$ = {p1[0]:3.2f} '+r'$kg\frac{m}{s}$', size=9, transform=ax.transAxes)
text_p2 = ax.text(0.75, 0.96, f'$p_{2}$ = {p2[0]:3.2f} '+r'$kg\frac{m}{s}$', size=9, transform=ax.transAxes)

# style = mpl.patches.ArrowStyle.Simple(head_length=6, head_width=0)
# arrow1 = mpl.patches.FancyArrowPatch((0, (l1 + l2)), (x1, z1), color = 'red', arrowstyle=style)
# arrow2 = mpl.patches.FancyArrowPatch((x1, z1), (x2, z2), color = 'red', arrowstyle=style)
# ax.add_artist(arrow1)
# ax.add_artist(arrow2)

line, = ax.plot([], [], "-", lw=1, color="red")
trace, = ax.plot([], [], ".-", lw=1, ms=2, color="orange")

dot1, = ax.plot(x1, z1, 'o', color='blue', markersize=6 + m1)  # first mass
dot2, = ax.plot(x2, z2, 'o', color='blue', markersize=6 + m2)  # second mass
dot3, = ax.plot(0, 0, 'o', color='red', markersize=3)  # zero dot

ani_running = True

# This section sets up the plot to use arrow keys to select frames and
# the space bar to pause the animation


def update_time():
    """Yield the frame number based on the maximum output of time steps.

    Returns
    -------
    t : iterator object
    """
    t = 0
    t_max = steps
    while t < t_max - 1:
        t += ani.direction
        yield t


def on_press(event):
    """Allow the animation to be paused and fast-forwarded or rewinded.

    Use the space key to pause the animation and the left/right arrow keys
    to fast-forward/rewind the animation.

    Returns
    -------
    None
    """
    if event.key.isspace():
        if ani.running:
            ani.pause()
        else:
            ani.resume()
        ani.running ^= True

    elif event.key == 'left':
        ani.direction = -1

    elif event.key == 'right':
        ani.direction = 1

    # Manually update the plot
    if event.key in ['left', 'right']:
        t = ani.frame_seq.__next__()
        update(t)
        plt.draw()


x1 = l1 * np.sin(theta1)
z1 = - l1 * np.cos(theta1)
x2 = x1 + l2 * np.sin(theta2)
z2 = z1 - l2 * np.cos(theta2)

history = 0

def update(i):
    """Update the figure for matplotlib's FuncAnimation.

    Parameters
    ----------
    i : integer
        number of frame

    Returns
    -------
    arrow1, arrow2, text_t, text_E : matplotlib objects
        The objects required to update the plot
    """
    
    dot1.set_data([x1[i]],[z1[i]])
    dot2.set_data([x2[i]],[z2[i]])
    text_t.set_text(f"t = {t[i]:5.2f} s")
    text_E.set_text(f"E = {E[i]:7.4f} J")

    line.set_data([0, x1[i], x2[i]], [0, z1[i], z2[i]])  # update the pendulum's lines

    global history

    if i == 1:
        history = 0

    if i - history >= steps / 2:
        history += 1

    trace.set_data(x2[history:i], z2[history:i])

    # print(theta1[history:i].size)
    # arrow1.set_positions((0,(l1 + l2)),(x1,z1))
    # arrow2.set_positions((x1,z1),(x2,z2))
    


    return dot1, dot2, text_t, text_E, line, trace


fig.canvas.mpl_connect("key_press_event", on_press)

ani = animation.FuncAnimation(fig, update, interval = 1, frames = update_time, save_count = steps - 1)
ani.running = True
ani.direction = 1


# plt.rcParams['animation.ffmpeg_path'] = "C:\\Users\\manol\\miniforge3\\envs\\climodel\\Library\\bin\\ffmpeg"
# FFwriter = animation.FFMpegWriter(fps=60, extra_args=['-vcodec', 'libx264'])
# ani.save('dp_animation.mp4', writer = FFwriter)


plt.show()
print("\nSuccessfully finished\n")
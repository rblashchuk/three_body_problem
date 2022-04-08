import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
import scipy as sci
import scipy.integrate as sciint
import tkinter as tk
from random import randint
from tkinter import colorchooser, messagebox, simpledialog

G = 6.67408e-11  # universal gravitation constant, N-m2/kg2
m_nd = 1.989e+30  # mass of the sun, kg
r_nd = 5.326e+12  # distance between stars in Alpha Centauri, m
v_nd = 30000  # relative velocity of earth around the sun, m/s
t_nd = 79.91 * 365 * 24 * 3600 * 0.51  # orbital period of Alpha Centauri, s

#  koefs
K1 = G * t_nd * m_nd / (r_nd ** 2 * v_nd)
K2 = v_nd * t_nd / r_nd

parent = tk.Tk()
parent.overrideredirect(1)
parent.withdraw()

specify_masses = messagebox.askyesno('Массы', 'Вы хотите сами задать значения масс?', parent=parent)
if specify_masses:
    masses = simpledialog.askstring('Массы', 'Введите массы тел через пробел (например, "1 2.3 4.56"', parent=parent)
    masses = list(map(float, masses.split()))
    m1 = masses[0]
    m2 = masses[1]
    m3 = masses[2]
else:
    m1 = 5
    m2 = 1
    m3 = 9

# initialising pos vectors
specify_init_r = messagebox.askyesno('Положение', 'Вы хотите сами задать начальное положение тел?', parent=parent)
if specify_init_r:
    r1 = simpledialog.askstring('Положение первого тела',
                                'Введите координаты первого тела через пробел (например, "1 2.3 4.56"', parent=parent)

    r2 = simpledialog.askstring('Положение второго тела',
                                'Введите координаты второго тела через пробел (например, "1 2.3 4.56"', parent=parent)

    r3 = simpledialog.askstring('Положение третьего тела',
                                'Введите координаты третьего тела через пробел (например, "1 2.3 4.56"', parent=parent)

    r1 = list(map(float, r1.split()))
    r2 = list(map(float, r2.split()))
    r3 = list(map(float, r3.split()))
else:
    r1 = [5, 0, 0]
    r2 = [0, 1, 4]
    r3 = [0, 1, 5]

# convert pos vectors to arrays
r1 = np.array(r1, dtype="float64")
r2 = np.array(r2, dtype="float64")
r3 = np.array(r3, dtype="float64")

# initialising velocities
specify_init_v = messagebox.askyesno('Скорости', 'Вы хотите сами задать начальные скорости тел?', parent=parent)
if specify_init_v:
    v1 = simpledialog.askstring('Скорость первого тела',
                                'Введите координаты начальной скорости первого тела через пробел (например, '
                                '"1 2.3 4.56"', parent=parent)

    v2 = simpledialog.askstring('Скорость второго тела',
                                'Введите координаты начальной скорости второго тела через пробел (например, '
                                '"1 2.3 4.56"', parent=parent)

    v3 = simpledialog.askstring('Скорость третьего тела',
                                'Введите координаты начальной скорости третьего тела через пробел (например, '
                                '"1 2.3 4.56"', parent=parent)

    v1 = list(map(float, v1.split()))
    v2 = list(map(float, v2.split()))
    v3 = list(map(float, v3.split()))
else:
    v1 = [0.1, 0.1, 0]
    v2 = [-0.5, 0, -0.1]
    v3 = [0.1, -0.1, 0]

# convert velocity vectors to arrays
v1 = np.array(v1, dtype="float64")
v2 = np.array(v2, dtype="float64")
v3 = np.array(v3, dtype="float64")

# initialising colors
specify_color = messagebox.askyesno('Цвета', 'Вы хотите сами задать цвета отрисовки траекторий?', parent=parent)
if specify_color:
    color = [0, 0, 0]
    color[0] = colorchooser.askcolor(initialcolor=(randint(0, 255), randint(0, 255), randint(0, 255)), parent=parent)
    color[1] = colorchooser.askcolor(initialcolor=(randint(0, 255), randint(0, 255), randint(0, 255)), parent=parent)
    color[2] = colorchooser.askcolor(initialcolor=(randint(0, 255), randint(0, 255), randint(0, 255)), parent=parent)

else:
    color = ["#FF93C0", "#9DAEFF", "#FFB896"]


# func defining the equations of motion
def three_body_equations(w, t, G, m1, m2, m3):
    r1 = w[:3]
    r2 = w[3:6]
    r3 = w[6:9]
    v1 = w[9:12]
    v2 = w[12:15]
    v3 = w[15:18]
    r12 = sci.linalg.norm(r2 - r1)
    r13 = sci.linalg.norm(r3 - r1)
    r23 = sci.linalg.norm(r3 - r2)

    dv1bydt = K1 * m2 * (r2 - r1) / r12 ** 3 + K1 * m3 * (r3 - r1) / r13 ** 3
    dv2bydt = K1 * m1 * (r1 - r2) / r12 ** 3 + K1 * m3 * (r3 - r2) / r23 ** 3
    dv3bydt = K1 * m1 * (r1 - r3) / r13 ** 3 + K1 * m2 * (r2 - r3) / r23 ** 3
    dr1bydt = K2 * v1
    dr2bydt = K2 * v2
    dr3bydt = K2 * v3
    r12_derivs = np.concatenate((dr1bydt, dr2bydt))
    r_derivs = np.concatenate((r12_derivs, dr3bydt))
    v12_derivs = np.concatenate((dv1bydt, dv2bydt))
    v_derivs = np.concatenate((v12_derivs, dv3bydt))
    derivs = np.concatenate((r_derivs, v_derivs))
    return derivs


# updating trajectories and scatters
def update_lines(num, data_lines, lines):
    global scat
    for i, (line, data) in enumerate(zip(lines, data_lines)):
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2, :num])
        scat[i].remove()
        scat[i] = ax.scatter(data[0, num], data[1, num], data[2, num], color=color[i], marker='x')
    # update axes limits
    ax.autoscale_view(tight=True)

    return lines


init_params = np.array([r1, r2, r3, v1, v2, v3])  # initial parameters array
init_params = init_params.flatten()
time_span = np.linspace(0, 200, 5000)

three_body_sol = sciint.odeint(three_body_equations, init_params, time_span, args=(G, m1, m2, m3))

r1_sol = three_body_sol[:, :3]
r2_sol = three_body_sol[:, 3:6]
r3_sol = three_body_sol[:, 6:9]

# reconstruct data to proper format
r1_data = [[r[0] for r in r1_sol], [r[1] for r in r1_sol], [r[2] for r in r1_sol]]
r2_data = [[r[0] for r in r2_sol], [r[1] for r in r2_sol], [r[2] for r in r2_sol]]
r3_data = [[r[0] for r in r3_sol], [r[1] for r in r3_sol], [r[2] for r in r3_sol]]
r1_data = np.array(r1_data)
r2_data = np.array(r2_data)
r3_data = np.array(r3_data)

data = [r1_data, r2_data, r3_data]

# attach 3D axis to the figure
fig = plt.figure()
ax = p3.Axes3D(fig, auto_add_to_figure=False)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
fig.add_axes(ax)

scat = [ax.scatter(0, 0, 0, color=color[i]) for i in range(3)]

lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]

lines[0].set_color(color[0])
lines[1].set_color(color[1])
lines[2].set_color(color[2])

# creating the Animation object
line_ani = animation.FuncAnimation(fig, update_lines, 5000, fargs=(data, lines), interval=1, blit=False)

ax.legend(scat, ['Тело 1', 'Тело 2', 'Тело 3'], loc='lower left')
plt.show()

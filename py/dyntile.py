from pyibex import *
from tubex_lib import *
import math
import random
import time

dt = 0.05 #0.02
iteration_dt = 3.0 #3.5
tdomain = Interval(0,5)  # [t0,tf=15]
x0 = (0, 0, 0)
u = Trajectory(tdomain, TFunction("3*(sin(t)^2)+t/100"), dt)
i_n = Interval(-0.02, 0.02)
n_u = RandTrajectory(tdomain, dt, i_n)
n_theta = RandTrajectory(tdomain, dt, i_n)
v_truth = TrajectoryVector(3)
x_truth = TrajectoryVector(3)
v_truth[2] = u + n_u
x_truth[2] = v_truth[2].primitive() + x0[2]
v_truth[0] = 10 * cos(x_truth[2])
v_truth[1] = 10 * sin(x_truth[2])
x_truth[0] = v_truth[0].primitive() + x0[0]
x_truth[1] = v_truth[1].primitive() + x0[1]

# Bounded trajectories (dead reckoning)
v = TubeVector(tdomain, dt, 3)
x = TubeVector(tdomain, dt, 3)
v[2] = Tube(u, dt).inflate(i_n.rad())  # command u with bounded uncertainties
x[2] = Tube(x_truth[2] + n_theta, dt).inflate(i_n.rad())  # heading measurement with bounded uncertainties
v[0] = 10 * cos(x[2])
v[1] = 10 * sin(x[2])
x = v.primitive() + IntervalVector(x0)  # dead reckoning


beginDrawing()
fig_map = VIBesFigMap("tiles")
fig_map.set_properties(50, 50, 1200, 600)
fig_map.add_tube(x, "x", 0, 1)
fig_map.add_trajectory(x_truth, "truth", 0, 1, "white")
fig_map.smooth_tube_drawing(True)
fig_map.show(1.)
y = TubeVector(tdomain, dt, 3)
ctc_g1 = CtcFunction(Function("x[3]", "y[3]","(sin(pi*(x[0]-y[0])) ; sin(pi*(x[1]-y[1])) ; sin(x[2]-y[2]))"))
ctc_g2 = CtcFunction(Function("x[3]", "y[3]","(sin(pi*(x[0]-y[1])) ; sin(pi*(x[1]-y[0])) ; cos(x[2]-y[2]))"))
ctc_g = ctc_g1|ctc_g2
ctc_f = CtcFunction(Function("x[3]", "v[3]","(v[0] - 10 * cos(x[2]) ; v[1] - 10 * sin(x[2]))"))
cn = ContractorNetwork()
cn.add(ctc.deriv, [x, v])
cn.add(ctc_g, [x,y])
cn.add(ctc_f, [x,v])
t = tdomain.lb()

while t < tdomain.ub():
    yt = x_truth(t)[0:3]
    ti = cn.create_dom(Interval(t))
    yi = cn.create_dom(IntervalVector(yt))
    xi = cn.create_dom(IntervalVector(3))
    yi=yi.inflate(0.01)

    '''
    y1, y2, y3 = 0.1, 0.2, 0.3
    f = Function("a1", "a2","sin(pi*(%f+a1*cos(%f)-a2*sin(%f)))*sin(pi*(%f+a1*sin(%f)+a2*cos(%f)))" % (y1, y3, y3, y2, y3, y3))
    S = SepFwdBwd(f, Interval(0, 0.001))
    pySIVIA(IntervalVector([[-2, 2], [-2, 2]]), S, 0.1)
    vibes.axisEqual()
    '''
    cn.add(ctc.eval, [ti, xi, x, v])
    cn.add(ctc_g, [xi, yi])
    contraction_dt = cn.contract_during(iteration_dt)
    if iteration_dt > contraction_dt: time.sleep(iteration_dt - contraction_dt)
    print('t=',t)
    print(x(t))
    fig_map.draw_box(x(t).subvector(0, 1))
    t += dt
cn.contract(True)
fig_map.show()
endDrawing()
print('fini')

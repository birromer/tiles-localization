from pyibex import *
from vibes import *
import numpy as np
import random
import math

def rand(I):
    x=I.lb()+random.random()*I.diam()
    return x

class ctc_evol():
    def __init__(C):        
        f1 = Function("xp1", "x1", "x3", "xp1-x1-%f*10*cos(x3)" % dt)
        f2 = Function("xp2", "x2", "x3", "xp2-x2-%f*10*sin(x3)" % dt)
        f3 = Function("xp3", "x3", 'u', "xp3-x3-%f*u" % dt)
        C.C1 = CtcFwdBwd(f1)
        C.C2 = CtcFwdBwd(f2)
        C.C3 = CtcFwdBwd(f3)

    def contract(C, xp1, xp2, xp3, x1, x2, x3, u):
        X = IntervalVector(3)
        X[0], X[1], X[2] = xp1, x1, x3
        C.C1.contract(X)
        xp1,x1,x3=X[0],X[1],X[2]
        X=IntervalVector(3)
        X[0],X[1],X[2]=xp2,x2,x3
        C.C2.contract(X)
        xp2,x2,x3=X[0],X[1],X[2]
        X=IntervalVector(3)
        X[0],X[1],X[2]=xp3,x3,u
        C.C3.contract(X)
        xp3,x3,u=X[0],X[1],X[2]
        return xp1,xp2,xp3,x1,x2,x3
        
class ctc_tile():
    def __init__(C):        
        g1a=Function("x1","x2","x3","y1","y2","y3","sin(pi*(x1-y1))")
        g2a=Function("x1","x2","x3","y1","y2","y3","sin(pi*(x2-y2))")
        g1b=Function("x1","x2","x3","y1","y2","y3","sin(pi*(x1-y2))")
        g2b=Function("x1","x2","x3","y1","y2","y3","sin(pi*(x2-y1))")
        g3a=Function("x1","x2","x3","y1","y2","y3","sin(x3-y3)")
        g3b=Function("x1","x2","x3","y1","y2","y3","cos(x3-y3)")
        C.Cm=(CtcFwdBwd(g1a)&CtcFwdBwd(g2a)&CtcFwdBwd(g3a))|(CtcFwdBwd(g1b)&CtcFwdBwd(g2b)&CtcFwdBwd(g3b))
    def contract(C,x1,x2,x3,y1,y2,y3):
        X=IntervalVector(6)
        X[0],X[1],X[2],X[3],X[4],X[5]=x1,x2,x3,y1,y2,y3
        C.Cm.contract(X)
        x1,x2,x3,y1,y2,y3 =X[0],X[1],X[2],X[3],X[4],X[5]
        return x1,x2,x3


    
vibes.beginDrawing()
vibes.newFigure('Tile')
vibes.setFigureSize(1000,1000)
    
_x1,_x2,_x3,dt=0,0,0,0.02
_X1,_X2,_X3=[],[],[]
U,Y1,Y2,Y3=[],[],[],[]
noise=0.1*Interval(-1,1)
kmax=1000
for k in range(0,kmax):   #generation of the data
    _u=(3*np.sin(k*dt)**2+k*dt/100+rand(noise))
    Y1.append(_x1+random.randint(-10,10)+rand(noise)+noise)
    Y2.append(_x2+random.randint(-10,10)+rand(noise)+noise)
    Y3.append(_x3+math.pi*random.randint(-10,10)+rand(noise)+noise)
    U.append(_u+noise)
    _X1.append(_x1)
    _X2.append(_x2)
    _X3.append(_x3)
    _x1=_x1+dt*10*np.cos(_x3)
    _x2=_x2+dt*10*np.sin(_x3)
    _x3=_x3+dt*_u

X1=[Interval(-30,30)]*(kmax+1)
X2=[Interval(-30,30)]*(kmax+1)
X3=[Interval(-10000,10000)]*(kmax+1)
X1[0],X2[0],X3[0]=Interval(0),Interval(0),Interval(0)
c_evol=ctc_evol()
c_tile=ctc_tile()

for n in range(0,5):
    for k in range(1,kmax):
        X1[k],X2[k],X3[k],X1[k-1],X2[k-1],X3[k-1]=c_evol.contract(X1[k],X2[k],X3[k],X1[k-1],X2[k-1],X3[k-1],U[k-1])
    for k in range(kmax,1,-1):
        X1[k],X2[k],X3[k],X1[k-1],X2[k-1],X3[k-1]=c_evol.contract(X1[k],X2[k],X3[k],X1[k-1],X2[k-1],X3[k-1],U[k-1])

    for k in range(0,kmax):
        X1[k],X2[k],X3[k]=c_tile.contract(X1[k],X2[k],X3[k],Y1[k],Y2[k],Y3[k])

for j in range(-20,20):
    vibes.drawLine([[-20, j], [20, j]], 'green')
    vibes.drawLine([[j, -20],[j, 20]], 'green')


for X1k,X2k in zip(X1,X2):
    vibes.drawBox(X1k[0],X1k[1],X2k[0],X2k[1],'blue[black]')

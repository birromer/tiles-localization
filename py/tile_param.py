from pyibex import *
from vibes import *

y1,y2,y3=0.1,0.2,0.3

f = Function("a1", "a2",
             "sin(pi * (%f + a1*cos(%f) - a2*sin(%f))) * sin(pi * (%f + a1*sin(%f) + a2*cos(%f)))"  % (y1, y3, y3, y2 ,y3 ,y3))

S = SepFwdBwd(f, Interval(0, 0.001))

vibes.beginDrawing()
vibes.newFigure('tile')
vibes.setFigureSize(1000,1000)
pySIVIA(IntervalVector([[-2,2], [-2,2]]), S, 0.1)
vibes.axisEqual()

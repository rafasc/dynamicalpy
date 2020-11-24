from dynamicalpy.model import SuperBlock
from dynamicalpy.blocks import Square
from dynamicalpy.blocks import ContinuousTimeLinearSystem
import numpy as np

m, k, β = 1, 0.5, 0.10

A = np.array([[0,1],
              [-k/m, -β/m]])
B = np.array([[0],[1/m]])

spring = ContinuousTimeLinearSystem(name='spring',
                                    state_space=(A,B))
spring.set_initial_state([0.5, 0])

f_externa = Square(name='force', frequency_hz=3/120,
                   dutycycle=0.05, phase=-5, high=1.5)

system = SuperBlock(name='system',
                    outputs=('force', 'pos'),
                    subsystems=[spring,f_externa])

system.connect('force:out0', 'spring:in0')
system.connect('spring:out0', 'system:pos')
system.connect('force:out0', 'system:force')
system.draw_diagram()

record=[':pos',':force']
system.run(120, record=record)
system.plot(plot=record)
from dynamicalpy.model import SuperBlock
from dynamicalpy.blocks import ContinuousTimeLinearSystem
import numpy as np

k, m = 0.5, 1

A = np.array([[0,    1],
              [-k/m, 0]])
B = np.array([[]])

spring = ContinuousTimeLinearSystem(name='spring',
                                  state_space=(A,B))
spring.set_initial_state([1,0])

system = SuperBlock(name='system', subsystems=[spring])

record=['spring:out0', 'spring:out1']
system.run(30, record=record)
system.plot(record)



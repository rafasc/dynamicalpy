from dynamicalpy.model import SuperBlock
from dynamicalpy.blocks import Step, Constant, Product
from dynamicalpy.blocks import ContinuousTimeLinearSystem
import numpy as np

amplitude, start, end = 1.5, 5, 7
step1 = Step('s1', start, 0, 1)
step2 = Step('s2', end, 1, 0)
c = Constant('constant', value=amplitude)
p = Product('*', inputs=3)
f_externa = SuperBlock('force', outputs=1,
                     subsystems=[step1,step2,p,c])

f_externa.connect('s1:out0', '*:in0')
f_externa.connect('s2:out0', '*:in1')
f_externa.connect('constant:out0', '*:in2')
f_externa.connect('*:out0', ':out0')

f_externa.draw_diagram()
record=[':out0','s1:out0', 's2:out0']
f_externa.run(10, record=record)
f_externa.plot(['s1:out0','s2:out0'])
f_externa.plot(':out0')

# Simulação conforme o exemplo anterior
m, k, β = 1, 0.5, 0.10
A = np.array([[0,1],
              [-k/m, -β/m]])
B = np.array([[0],[1/m]])
spring = ContinuousTimeLinearSystem(name='spring',
                                    state_space=(A,B))
spring.set_initial_state([0.5, 0])

# Força externa definida anteriormente
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









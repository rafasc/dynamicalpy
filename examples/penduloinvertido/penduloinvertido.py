from dynamicalpy.model import SuperBlock
from dynamicalpy.blocks import *

import numpy as np
from scipy import constants

m = .23 # massa pendulo
M = 2.4 # massa do carro
l = .35 # tamanho do pendulo
g = constants.g # constante gravitacional
f = .1  # coeficiente de fricção entre o pendulo e o carro
F = 0   # Forca aplicada no carro

def dpendulum(t,x,u):
    x1, x2, x3,x4 = x
    F = u[0]
    dx1 = x2
    dx2 = ((-m*g*np.sin(x3)*np.cos(x3) +
            m*l*x4**2*np.sin(x3) +
            f * m * x4 * np.cos(x3) + F ) /
           (M + (1-np.cos(x3)**2)*m))

    dx3 = x4
    dx4 = (((M+m)*(g*np.sin(x3)-f*x4) -
            (l*m*x4**2*np.sin(x3) + F) * np.cos(x3)) /
            (l*(M+(1-np.cos(x3)**2)*m)))
    return np.array([dx1, dx2, dx3, dx4])

outputs= ('car-position', 'car-velocity',
          'pendulum-angle', 'pendulum-angular-velocity')
pend = ContinuousTimeSystem(name='pendulum',
                            outputs=outputs,
                            inputs=1,
                            state_eq=dpendulum,
                            order=4)
pend.set_initial_state([0,0,np.radians(10),0])


# Proportional Controler
def PController(name, P):
    c = Constant(value=P)
    p = Product(inputs=2)
    controller = SuperBlock(name=name, inputs=1, outputs=1,
                 subsystems=[c,p])
    controller.connect(':in0', f"{p.name}:in0")
    controller.connect(f"{c.name}:out0", f"{p.name}:in1")
    controller.connect(f"{p.name}:out0", ":out0")
    return controller

P_angle = PController('angle_control', 30)
P_position = PController('pos_control', 0.05)
P_velocity = PController('velocity_control',0.05)
s = Sum(signs='+++')
system = SuperBlock(name='system', outputs=outputs,
                    subsystems=[pend, s,
                                P_angle,
                                P_position,
                                P_velocity])

system.connect('pendulum:pendulum-angle',
               f'{P_angle.name}:in0')
system.connect('pendulum:car-position',
               f'{P_position.name}:in0')
system.connect('pendulum:car-velocity',
               f'{P_velocity.name}:in0')

system.connect(f'{P_angle.name}:out0',
               f'{s.name}:in0')
system.connect(f'{P_position.name}:out0',
               f'{s.name}:in1')
system.connect(f'{P_velocity.name}:out0',
               f'{s.name}:in2')

system.connect(f'{s.name}:out0', 'pendulum:in0')

for port in outputs:
    system.connect(f'pendulum:{port}', f'system:{port}')

record=[f':{port}' for port in outputs]
system.run(80, record=record)
for port in record:
    system.plot(port)
system.draw_diagram()
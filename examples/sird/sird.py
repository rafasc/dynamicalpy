from dynamicalpy.model import SuperBlock
from dynamicalpy.blocks import ContinuousTimeSystem
import numpy as np

β = 0.5
𝛾 = 0.07
µ = 0.005
initial_state = [997, 3, 0, 0]
N = sum(initial_state)

print(f"Modelo SIRD com R0 de {β / 𝛾}")

def dsird(t,x,u):
    S, I, R, D = x
    dS = - (β * I * S) / N
    dI = (β * I * S) / N - I * 𝛾 - I * µ
    dR = 𝛾 * I
    dD = µ * I
    return np.array([dS, dI, dR, dD])

ports=('S','I','R','D')
sird = ContinuousTimeSystem(name='sird', outputs=ports,
                            state_eq=dsird, order=4, )
sird.set_initial_state(initial_state)

system = SuperBlock(name='system', outputs=ports,
                    subsystems=[sird])

record=[f"sird:{p}" for p in ports]
for port in ports:
    system.connect(f"sird:{port}", f"system:{port}")

system.run(70, record=record)
system.plot(record)
system.draw_diagram()

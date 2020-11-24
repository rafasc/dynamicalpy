from dynamicalpy.model import *
from dynamicalpy.blocks import *

#1 definição dos blocos
wave1 = Sinusoid('w1')
wave2 = Sinusoid('w2', frequency_hz=.7, amplitude=1)
wave3 = Sum('w3', '++')

#2 aggregação dos blocos em SuperBloco
signal = SuperBlock('signal', outputs=1,
                    subsystems=[wave1, wave2, wave3])

#3 ligações entre blocos
signal.connect('w1:out0', 'w3:in0')
signal.connect('w2:out0', 'w3:in1')
signal.connect('w3:out0', ':out0')

#4 visualização do sistema em estudo
signal.draw_diagram()
signal.print_links()

#5 execução da simulação por 10 segundos
signal.run(10, record=['w3:out0', 'w1:out0',
                       'w2:out0', ':out0'])

#6 visualização gráfica dos valores
signal.plot('w1:out0')
signal.plot(['w2:out0'])
signal.plot([':out0'])
signal.plot(['w3:out0', 'w1:out0', 'w2:out0'])

#7 requisição dos valores
print(signal.values(record=['w3:out0', 'w2:out0']))
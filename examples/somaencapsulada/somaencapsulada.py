from dynamicalpy.model import *
from dynamicalpy.blocks import *

#1 definição dos blocos
wave1 = Sinusoid('w1')
wave2 = Sinusoid('w2', frequency_hz=.7, amplitude=1)

#2 configuração do superbloco dos sinasi e ligações
signals = SuperBlock('signals', outputs=2,
                     subsystems=[wave1, wave2])
signals.connect('w1:out0', ':out0')
signals.connect('w2:out0', ':out1')

#3 configuração do superbloco para simulação
result = Sum('sum', '++')
system = SuperBlock('system', outputs=1,
                    subsystems=[signals,result])
system.connect('signals:out0', 'sum:in0')
system.connect('signals:out1', 'sum:in1')
system.connect('sum:out0', 'system:out0')

# #4 visualização do sistema em estudo
system.draw_diagram()
signals.draw_diagram()

#5 execução da simulação por 10 segundos
system.run(10, record=[':out0',
                       'signals:out0',
                       'signals:out1'])

#6 visualização gráfica dos valores
system.plot('signals:out0')
system.plot(['signals:out1'])
system.plot([':out0'])
system.plot([':out0', 'signals:out0', 'signals:out1'])
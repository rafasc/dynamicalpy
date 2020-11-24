import warnings
import logging
import sys

from contextlib import suppress
from functools import singledispatch, partial

from abc import ABCMeta, abstractmethod

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import integrate

from itertools import accumulate
from collections import OrderedDict

#draw
import pygraphviz as pgv
from io import BytesIO
from IPython.display import Image, display

class BaseBlock(object, metaclass=ABCMeta):
    _Counter = 0

    @abstractmethod
    def __init__(self, name=None, inputs=0, outputs=0):
        self.name = self._generate_name(name)
        self.input_names = self._parse_portnames(inputs, prefix='in')
        self.output_names = self._parse_portnames(outputs, prefix='out')
        self.input_number = len(self.input_names)
        self.output_number = len(self.output_names)
        self.order = 0

    def state_deriv(self, t, x, u):
        pass

    def output_eq(self, t, x, u):
        raise NotImplementedError

    def set_initial_state(self, state):
        self.initial_state = np.array(state)

    def __getitem__(self, key):
        if key in self.input_names or key in self.output_names:
            return (self.name, key)
        else:
            raise IndexError('port name does not exist')

    def __str__(self):
        template = 'Block name: {block.name}\n'\
                   'inputs({block.input_number}): {ins}\n'\
                   'outputs({block.output_number}): {outs}'
        string = template.format(block = self,
                                 ins=', '.join(self.input_names),
                                 outs=', '.join(self.output_names))
        return string


    def __repr__(self):
        return '{}: {}'.format(self.__class__.__qualname__ , self.name)


    @staticmethod
    def _names_from_portnames(ports, prefix):

        @singledispatch
        def f(ports, prefix):
            raise TypeError('Port type not supported')

        @f.register(int)
        def _(ports, prefix):
            return [prefix+str(i) for i in range(ports)]

        @f.register(str)
        def _(ports, prefix):
            return [n.strip() for n in ports.split(',')]

        @f.register(list)
        @f.register(tuple)
        def _(ports, prefix):
            return [n.strip() for n in ports]

        return f(ports, prefix)


    def _parse_portnames(self, ports, prefix='port'):
        ports = self._names_from_portnames(ports, prefix=prefix)

        port_names = list(ports)
        with suppress(AttributeError):
            port_names += self.input_names
        with suppress(AttributeError):
            port_names += self.output_names

        dup_names = [ n for n in set(port_names) if port_names.count(n) > 1]
        if dup_names:
            raise NameError('Duplicate port names {}'.format(dup_names))
        return ports

    @classmethod
    def _generate_name(cls, name):
        if name is None:
            cls._Counter += 1
            new_name = '{}_{}'.format(cls.__name__, cls._Counter)
        else:
            new_name = str(name)

        return new_name


class SuperBlock(BaseBlock):

    def __init__(self, name=None, inputs=0, outputs=0, subsystems=None):
        super().__init__(name=name, inputs=inputs, outputs=outputs)
        self.subsystems = self._error_on_duplicates_names(subsystems)
        self.links = {}
        self.feedforward = False
        self._record_data = None

    def get_subsystem_index_by_name(self, name):
        return [s.name for s in self.subsystems].index(name)

    @staticmethod
    def _error_on_duplicates_names(subsystems):
        if subsystems:
            names  = [s.name for s in subsystems]
            dups = [name for name in set(names) if names.count(name) > 1]
            if dups:
                msg = 'Duplicate names in Supperblock subsystems: {}'.format(dups)
                raise NameError(msg)
            return subsystems
        else:
            return []

    def state_deriv(self, t, x, u):
        self._recalculate_inputs_outputs(t, x, u)
        deriv = np.empty(self.order)
        for i, s in enumerate(self.slices):
            deriv[s] = self.subsystems[i].state_deriv(t, x[s], self.uu[i])
        return deriv

    def _recalculate_inputs_outputs(self, t, x, u):
        for blk_index, blk in enumerate(self.subsystems):
            for input_index in range(blk.input_number):
                other_blk, outport = self.links[(blk_index, input_index)]
                self.uu[blk_index][input_index] = u[outport] if other_blk is None else self.yy[other_blk][outport]
            self.yy[blk_index] = blk.output_eq(t, x[self.slices[blk_index]], self.uu[blk_index])
                # some uu[i][j] may be invalid, but they'll be corrected
                # on the following step.

        for blk_index, blk in enumerate(self.subsystems):
            if not blk.feedforward:
                for input_index in range(blk.input_number):
                    other_blk, outport = self.links[(blk_index, input_index)]
                    self.uu[blk_index][input_index] = u[outport] if other_blk is None else self.yy[other_blk][outport]


    def output_eq(self, t, x, u):
        self._recalculate_inputs_outputs(t, x, u)
        y = [0] * self.output_number
        for i in range(self.output_number):
            blk, port = self.links[(None, i)]
            y[i] = u[port] if blk is None else self.yy[blk][port]
        return y

    def set_initial_state(self, state):
        raise Exception("Cannot set SuperBlock state. Set the state of its components instead ")

    @property
    def initial_state(self):
        initial_state = np.empty(self.order)
        for i, s in enumerate(self.slices):
            with suppress(AttributeError):
                initial_state[s] = self.subsystems[i].initial_state
        return initial_state

    def connect(self, outport, inport):
        insys, insig = self._parse_port(inport, direction='in')
        outsys, outsig = self._parse_port(outport, direction='out')
        self.links[(insys, insig)] = (outsys, outsig)


    def _parse_port(self, port, direction=None):
        names = [s.name for s in self.subsystems]
        if isinstance(port, str):
            blk, portsig = port.split(':')  # 'system:out1'
        else:
            blk, portsig = port             # (2, 1) or ('system','out1')

        # make outsys be either None (subperblock input) or an index:
        if isinstance(blk, str):
            blk = None if blk in ('', self.name) else names.index(blk)
        elif isinstance(blk, BaseBlock):
            blk, portsig = blk[port] = None if blk == self else self.subsystems.index(blk)

        # get index of output signal, if it is referred by name
        if isinstance(portsig, str):
            if blk is None:
                if direction == 'in':
                    portsig = self.output_names.index(portsig)
                elif direction == 'out':
                    portsig = self.input_names.index(portsig)# get superblock input
            else:
                if direction == 'in':
                    portsig = self.subsystems[blk].input_names.index(portsig)
                elif direction == 'out':
                    portsig = self.subsystems[blk].output_names.index(portsig)

        return (blk, portsig)

    def print_links(self, raw=False):
        header = '---- links of superblock "{}" ----'.format(self.name)
        print(header)

        links = {}
        for (blk_to, port_to), (blk_from, port_from) in self.links.items():
            if blk_from is None:
                s1 = self.name
                p1 = self.input_names[port_from]
            else:
                s1 = self.subsystems[blk_from].name
                p1 = self.subsystems[blk_from].output_names[port_from]

            if blk_to is None:
                s2 = self.name
                p2 = self.output_names[port_to]
            else:
                s2 = self.subsystems[blk_to].name
                p2 = self.subsystems[blk_to].input_names[port_to]

            links[f"{s1}:{p1}"] = (f"{s2}:{p2}",
                                   f'({blk_from}, {port_from}) -> ({blk_to},{port_to})')

        c1 = max(len(key) for key in links.keys())
        c2 = max(len(key[0]) for key in links.values())

        for k,v in links.items():
            print(f"{k:<{c1}}", '->', f"{v[0]:<{c2}}",
                  end=f'{" "+v[1] if raw else ""}\n')

        print('-' * len(header))



    def has_unconnected_inputs(self, recursive=False):
        for blk_idx, blk in enumerate(self.subsystems):
            if recursive:
                try:
                    if blk.has_unconnected_inputs(recursive=True):
                        return True
                except(AttributeError):
                    return False

            for input_idx, input_ in enumerate(blk.input_names):
                if (blk_idx,input_idx) not in self.links:
                    return True
        return False


    def update_internal_structures(self):
        for s in self.subsystems:
            with suppress(AttributeError):
                s.update_internal_structures()

        self.order = sum(s.order for s in self.subsystems)

        flow = nx.DiGraph()   # links between subsystems
        inout = nx.DiGraph()  # links to inputs/outputs of the superblock

        for (i, _), (o, _) in self.links.items():
            if i is None and o is None:
                # superblock has direct feedforward
                inout.add_edge("I", "O")
            elif o is None and self.subsystems[i].feedforward:
                inout.add_edge("I", i)
            elif i is None and self.subsystems[o].feedforward:
                inout.add_edge(o, "O")
            elif i is not None and self.subsystems[i].feedforward:
                flow.add_edge(o, i)

        # compute topological sort of systems with feedforward
        # and generate a list with the order the system outputs should be
        # calculated. Then permute the subsystems list to the new order
        with_ff = list(nx.topological_sort(flow))
        without_ff = list(set(range(len(self.subsystems))) - set(with_ff))
        eval_order = without_ff + with_ff
        self.subsystems = [self.subsystems[j] for j in eval_order]

        # update self.links to honor the new evaluation order indexes
        newlinks = {}
        for i, o in self.links.items():
            ii = i if i[0] is None else (eval_order.index(i[0]), i[1])
            oo = o if o[0] is None else (eval_order.index(o[0]), o[1])
            newlinks[ii] = oo
        self.links = newlinks

        # to check if superblock has feedforward we join the two graphs
        # and check if there is any path from any input to any output
        flow.add_edges_from(inout.edges())
        self.feedforward = flow.has_node("I") \
            and flow.has_node("O") \
            and nx.node_connectivity(flow, "I", "O") > 0

        # assign the slices for each subcomponent
        indices = [0] + list(accumulate(s.order for s in self.subsystems))
        self.slices = [slice(a, b) for a, b in zip(indices[:-1], indices[1:])]

        self.yy = [[None] * s.output_number for s in self.subsystems]
        self.uu = [[None] * s.input_number for s in self.subsystems]


    def run(self, time, start=0, resolution=100, record=None):
        if self.has_unconnected_inputs():
            raise Exception("Simulation cannot run with floating inputs")

        self.update_internal_structures()

        if self.order:
            update = self.state_deriv
            state = self.initial_state
        else:
            state = [0]
            def update(t,x,u):
                self.state_deriv(t, x, u)
                return [0]

        integ = integrate.ode(partial(update, u=[])).set_integrator('dopri5')
        integ.set_initial_value(state, start)


        tt, data = ([],[])
        while integ.successful() and integ.t < time:
            tt.append(integ.t)
            l = []

            y = integ.integrate(integ.t + 1/resolution)
            # recalculate input and output values
            # based on the new solution.
            self._recalculate_inputs_outputs(integ.t, y, u=[])

            if record is not None:
                for line in record:
                    blk_index, output_index = self._get_output_from_line(line)
                    l.append(self.yy[blk_index][output_index])
                data.append(l)

        self._record_data = (tt,data,record)


    def _get_output_from_line(self,line):
        s , p = line.split(':')
        if s == '':
            blk_index, output_index = self.links[(None, self.output_names.index(p))]
        else:
            blk_index = self.get_subsystem_index_by_name(s)
            output_index = self.subsystems[blk_index].output_names.index(p)
        return blk_index, output_index

    def plot(self, plot=None):
        #accept non lists as one item list
        if not isinstance(plot, list):
            plot = [plot]

        plt.figure()
        plt.plot(self._record_data[0],
                 self.values(record=plot))

        plt.legend(plot)
        plt.axis('tight')
        plt.show()

    def values(self, record=None):
        #accept non lists as one item list
        if not isinstance(record, list):
            record = [record]
        column_selector = [ self._record_data[2].index(r) for r in record ]
        return np.array(np.array(self._record_data[1]))[:,column_selector]


    def draw_diagram(self):

        g = pgv.AGraph(strict=False, directed=True)
        g.graph_attr['rankdir'] = 'TB'
        g.graph_attr['splines'] = 'true'
        g.graph_attr['overlap'] = 'false'

        g.node_attr['shape'] = 'record'


        title_template = '{{{}}} | {} | {{{}}}'

        for el in (*self.subsystems, self):

            label = title_template.format('|'.join('<{0}>{0}'.format(name) for name in el.input_names),
                                      el.name,
                                      '|'.join('<{0}>{0}'.format(name) for name in el.output_names))

            g.add_node(el.name, label = label)

        for (blk_to, port_to), (blk_from, port_from) in self.links.items():
            if blk_from is None:
                s1 = self.name
                p1 = self.input_names[port_from]
            else:
                s1 = self.subsystems[blk_from].name
                p1 = self.subsystems[blk_from].output_names[port_from]

            if blk_to is None:
                s2 = self.name
                p2 = self.output_names[port_to]
            else:
                s2 = self.subsystems[blk_to].name
                p2 = self.subsystems[blk_to].input_names[port_to]

            g.add_edge(s1, s2, tailport=p1+':e' , headport=p2+':w')


        g.subgraph(nbunch=(s.name for s in (*self.subsystems, self)),
                   name='cluster1', label=self.name)

        with BytesIO() as fp:
            plt.figure()
            g.draw(fp, format='png', prog='fdp')
            fp.seek(0)
            plt.axis("off")
            plt.imshow(plt.imread(fp, format='png'))
            plt.show()

if __name__ == '__main__':
    pass

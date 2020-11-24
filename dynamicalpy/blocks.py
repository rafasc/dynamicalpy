from .model import BaseBlock
import numpy as np


class ContinuousTimeLinearSystem(BaseBlock):

    def __init__(self, name=None, state_space=None, transfer_function=None,
                 inputs=None, outputs=None, initial_state=None):
        '''
        state_space = (A,B,C) is a state space realization.
        transfer_function = (N,D) are numerator and denominator polinomial
        coeficients, eg. [1,2,3] means s^2 + 2s + 3. (SISO only)
        The number of inputs, outputs and states are automatically inferred
        from the matrices.
        '''

        feedforward = False
        if isinstance(state_space, tuple):
            n = state_space[0].shape[0]
            if len(state_space) == 4:
                feedforward = True
                A, B, C, D = state_space
            elif len(state_space) == 3:
                A, B, C = state_space
                D = None
            elif len(state_space) == 2:
                A, B = state_space
                C = np.eye(n)
                D = None
            elif len(state_space) == 1:
                A = state_space[0]
                B = np.array([]).reshape(n, 0)  # empty column matrix
                C = np.eye(self.A.shape[0])
                D = None
        elif isinstance(transfer_function, tuple):
            num, den = transfer_function
            numsize = len(num)
            densize = len(den)
            n = len(den) - 1     # system order
            if numsize > densize:
                raise Exception('Transfer function has more zeros than poles.')

            # normalize tf so that denominator is s^n + a1 s^(n-1) + ...
            b = np.zeros(densize)
            b[-numsize:] = num / den[0]
            a = den / den[0]

            A = np.vstack((np.eye(n-1, n, 1), -a[:0:-1]))  # companion form
            B = np.array((n-1) * [0] + [1]).reshape(n, 1)
            if numsize == densize:
                C = (b - b[0] * a)[:0:-1].reshape(1, n)
                D = np.array([b[0]]).reshape(1, 1)
                feedforward = True
            else:
                C = b[:0:-1].reshape(1, n)
                D = np.zeros((1, 1))

        if inputs is None:
            inputs = B.shape[1]
        else:
            raise Exception('Not yet implemented.')

        if outputs is None:
            outputs = C.shape[0]
        else:
            raise Exception('Not yet implemented.')

        super().__init__(name, inputs=inputs, outputs=outputs)
        self.order = A.shape[0]
        self.A, self.B, self.C, self.D = A, B, C, D
        self.feedforward = feedforward
        super().set_initial_state(initial_state or np.zeros(self.order))

    def state_deriv(self, t, x, u):
        return np.dot(self.A, x) + np.dot(self.B, np.array(u))

    def output_eq(self, t, x, u):
        if self.feedforward:
            return (np.dot(self.C, x) + np.dot(self.D, np.array(u))).tolist()
        else:
            return np.dot(self.C, x).tolist()



class ContinuousTimeSystem(BaseBlock):

    def __init__(self, name=None, state_eq=None, order=0, inputs=0, outputs=0,
                 output_eq=(lambda t, x, u: x), initial_state=None,
                 feedforward=False):
        super().__init__(name, inputs, outputs)
        self.feedforward = feedforward
        self.state_deriv = state_eq
        self.order = order
        self.output_eq = output_eq
        super().set_initial_state(initial_state or np.zeros(self.order))

# Source blocks

class Sawtooth(BaseBlock):

    def __init__(self, name=None, low=0, high=1, frequency_hz=1, phase=0):
        super().__init__(name, inputs=0, outputs=1)
        self.feedforward = False
        self.order = 0

        self.phase = phase
        self.frequency_hz = frequency_hz
        self.period = 1/frequency_hz
        self.high = high
        self.low = low

    def output_eq(self, t, x, u):
        return [(t+self.phase) % self.period / self.period *
                (self.high - self.low) + self.low]


class Triangle(BaseBlock):

    def __init__(self, name=None, frequency_hz=1, amplitude=1, offset=0, phase=0):
        super().__init__(name, inputs=0, outputs=1)
        self.feedforward = False
        self.frequency_hz = frequency_hz
        self.period = 1/frequency_hz
        self.amplitude = amplitude
        self.offset = offset
        self.phase = phase

    def output_eq(self, t, x, u):
        return [2 * np.abs((t+self.phase) % self.period - self.period/2) - 1 *
                self.amplitude + self.offset]

class Square(BaseBlock):

    def __init__(self, name=None, frequency_hz=1, low=0, high=1, dutycycle=0.5, phase=0):
        super().__init__(name, inputs=0, outputs=1)
        self.feedforward = False
        self.period = 1 / frequency_hz
        self.low = low
        self.high = high
        self.dutycycle = dutycycle
        self.phase = phase

    def output_eq(self, t, x, u):
        return [((t + self.phase) % self.period < self.dutycycle *
                 self.period) * (self.high - self.low) + self.low]


class Sinusoid(BaseBlock):

    def __init__(self, name=None, frequency_hz=1, amplitude=1, offset=0, phase=0):
        super().__init__(name, inputs=0, outputs=1)
        self.feedforward = False
        self.angular_frequency = frequency_hz * 2 * np.pi
        self.amplitude = amplitude
        self.offset = offset
        self.phase = phase


    def output_eq(self, t, x, u):
        # FIXME should the sin argument the principal argument [-pi,pi[ ???
        return [self.amplitude * np.sin(self.angular_frequency * t + self.phase) + self.offset]


class Constant(BaseBlock):

    def __init__(self, name=None, value=1):
        super().__init__(name, inputs=0, outputs=1)
        self.feedforward = False
        self.value = value

    def output_eq(self, t, x, u):
        return [self.value]


class Step(BaseBlock):

    def __init__(self, name=None, step_time=1, initial_value=0, final_value=1):
        super().__init__(name, inputs=0, outputs=1)
        self.feedforward = False
        self.step_time = step_time
        self.initial_value = initial_value
        self.final_value = final_value

    def output_eq(self, t, x, u):
        return [self.initial_value if t < self.step_time else self.final_value]


# "IO" blocks

class Sum(BaseBlock):

    def __init__(self, name=None, signs='++'):
        super().__init__(name, inputs=len(signs), outputs=1)
        self.feedforward = True
        self.signs = signs

    def output_eq(self, t, x, u):
        return [sum(v if s == '+' else -v for s, v in zip(self.signs, u))]


class SumConst(Sum):

    def __init__(self, name=None, signs='+', value=0):
        super().__init__(name, signs)
        self.value = value

    def output_eq(self, t, x, u):
        return [super().output_eq(t, x, u)[0] + self.value]


class Product(BaseBlock):

    def __init__(self, name=None, inputs=2):
        super().__init__(name, inputs, outputs=1)
        self.feedforward = True

    def output_eq(self, t, x, u):
        y = u[0]
        for ui in u[1:]:
            y *= ui
        return [y]


class Function(BaseBlock):

    def __init__(self, function, inputs, outputs, name='None'):
        super().__init__(name, inputs, outputs)
        self.feedforward = True
        self.function = function

    def output_eq(self, t, x, u):
        y = self.function(*u)
        return [y] if np.isscalar(y) else y


class Saturation(BaseBlock):

    def __init__(self, name=None, inputs=1, outputs=1, low=-1, high=1):
        super().__init__(name, inputs, outputs)
        self.feedforward = True
        self.low = low
        self.high = high

    def output_eq(self, t, x, u):
        if u[0] > self.high:
            return [self.high]
        elif u[0] < self.low:
            return [self.low]
        else:
            return u


class Dummy(BaseBlock):

    def __init__(self, name, inputs, outputs):
        super().__init__(name, inputs, outputs)


if __name__ == '__main__':
    pass

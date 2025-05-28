""" Guitar effects for real-time audio processing

The following classes derived from RTProcessor implement a variety of simple
real-time guitar effects
"""

__author__ = 'Paolo Prandoni'

import numpy as np
from rtprocessor import RTProcessor, Passthru

class Echo(RTProcessor):
    """ simple echo, 1 repetition, delay 0.3 seconds
    """
    order = 10

    def __init__(self, rate, channels):
        # we will need a second's worth of buffering
        super().__init__(rate, channels, max_delay=rate)

        self.alpha = 0.7
        self.norm = 1.0 / (1 + self.alpha)
        self.D = int(0.3 * self.SF)

    def _process(self, x):
        return self.norm * (x + self.alpha * self.x.get(self.D))


class Slapback(RTProcessor):
    """ short-delay echo, 1 repetition, delay 100ms
    """
    order = 11

    def __init__(self, rate, channels):
        super().__init__(rate, channels, max_delay=rate)

        self.a = 1
        self.b = 0.4
        self.norm = 1.0 / (self.a + self.b)
        self.N = int(0.1 * self.SF)

    def _process(self, x):
        return self.norm * (self.a * x + self.b * self.x.get(self.N))


class Recursive_Echo(RTProcessor):
    """ Echo implemented as a feedback loop (like Karplus-Strong)
    """
    order = 20

    def __init__(self, rate, channels):
        # we will need a second's worth of buffering
        super().__init__(rate, channels, max_delay=rate)

        self.a = 0.7
        self.norm = 1.5 / (1 + self.a)
        self.N = int(0.3 * self.SF)

    def _process(self, x):
        # y[n] = x[n] + ay[n-N]
        return self.norm * x + self.a * self.y.get(self.N)


class Natural_Echo(RTProcessor):
    """ Echo combining a feedback loop and a simple leaky integrator
    lowpass
    """
    order = 30

    def __init__(self, rate, channels):
        # we will need a second's worth of buffering
        super().__init__(rate, channels, max_delay=rate)

        self.a = 0.8
        self.l = 0.7
        self.D = int(0.3 * self.SF)
        self.norm = 1.5 / (1 + self.a)

    def _process(self, x):
        # y [n] = x[n] + y[n-D] * h[n], h[n] leaky integrator
        return self.norm * (x - self.l * self.x.get(1)) + \
               self.l * self.y.get(1) + self.a * (1 - self.l) * self.y.get(self.D)


class Reverb(RTProcessor):
    """ Schroeder reverberation (see https://ccrma.stanford.edu/~jos/pasp/Schroeder_Reverberators.html)
        Uses a cascade of three allpass filters and a bank of four parallel comb filters
    """
    order = 40

    AP = [[0.7, 347], [0.7, 113], [0.7, 37]]     # gain and delay for allpass filters
    CF = [[0.773, 1687], [0.802, 1601], [0.753, 2053], [0.733, 2251]]  # same for comb filters
    ORIGINAL_RATE = 25000

    def __init__(self, rate, channels, mix=0.1):
        super().__init__(rate, channels)
        # mixing parameter
        self.wet = mix
        self.dry = 1 - mix
        # the wetter the mix, the more we need to scale down the wet signal
        self.norm = 1 - mix

        # initialize a separate buffer for each filter
        src = rate / self.ORIGINAL_RATE
        self.ap_filters = [{
            "a": p[0],
            "N": int(p[1] * src),
            "buf": RTProcessor(rate, channels, max_delay=int(p[1] * src))
        } for p in self.AP]
        self.cf_filters = [{
            "a": p[0],
            "N": int(p[1] * src),
            "buf": RTProcessor(rate, channels, max_delay=int(p[1] * src))
        } for p in self.CF]

    def _process(self, x):
        dry = x
        x = x * self.norm
        # bank of comb filters in parallel
        out = 0
        for cf in self.cf_filters:
            y = x + cf["a"] * cf["buf"].y.get(cf["N"])
            cf["buf"].y.push(y)
            out = out + y

        # followed by a cascade of allpass filters:
        x = out
        for ap in self.ap_filters:
            a, N = ap["a"], ap["N"]
            y = -a * x + ap["buf"].x.get(N) + a * ap["buf"].y.get(N)
            ap["buf"].x.push(x)
            ap["buf"].y.push(y)
            x = y

        return self.dry * dry + self.wet * y


class Reverb_Far(Reverb):
    """ Reverb with a lot of wet signal to simulate a far sound source"""
    order = 41
    def __init__(self, rate, channels):
        super().__init__(rate, channels, mix=0.8)


class BassBoost(RTProcessor):
    """ Simple second-order low shelf """
    order = 50

    def __init__(self, rate, channels, cutoff=300, gain=15):
        super().__init__(rate, channels, max_delay=2)

        Q = 1 / np.sqrt(2)
        w = 2 * np.pi * cutoff / rate
        A = 10 ** (gain / 40)
        alpha = np.sin(w) / (2 * Q)
        c = np.cos(w)
        a0 = (A + 1) + (A - 1) * c + 2 * np.sqrt(A) * alpha
        self.a1, self.a2 = \
            (-2 * ((A - 1) + (A + 1) * c)) / a0, \
            ((A + 1) + (A - 1) * c - 2 * np.sqrt(A) * alpha) / a0
        self.b0, self.b1, self.b2 = \
            (A * ((A + 1) - (A - 1) * c + 2 * np.sqrt(A) * alpha)) / a0,\
            (2 * A * ((A - 1) - (A + 1) * c)) / a0, \
            (A * ((A + 1) - (A - 1) * c - 2 * np.sqrt(A) * alpha)) / a0
        self.norm = .5

    def _process(self, x):
        # y[n] = x[n] + b_1x[n-1] + b_2x[n-2] - a_1y[n-1] - a_2y[n-2]
        return self.norm * (self.b0 * x + self.b1 * self.x.get(1) + self.b2 * self.x.get(2)) \
                - self.a1 * self.y.get(1) - self.a2 * self.y.get(2)


class Fuzz(RTProcessor):
    """ Very crude nonlinear limiter (hard distortion)
    """
    order = 60

    def __init__(self, rate, channels):
        # memoryless
        super().__init__(rate, channels)
        self.limit = 32767 * 0.01
        self.gain = 10

    def _process(self, x):
        # y[n] = a trunc(x[n]/a)
        return self.gain * max(min(x, self.limit), -self.limit)


class Wah(RTProcessor):
    """ Wah-wah autopedal. A slow oscillator moves the positions of
    the poles in a second-order filter around their nominal value
    The result is a time-varying bandpass filter
    """
    order = 70

    def __init__(self, rate, channels):
        # we just need a second order filter
        super().__init__(rate, channels, max_delay=2)

        self.pole_delta = 0.3 * np.pi  # max pole deviation
        self.phi = 3.0 * 2.0 * np.pi / self.SF  # LFO frequency
        self.omega = 0.0
        self.pole_mag = 0.99  # pole magnitude
        self.pole_phase = 0.04 * np.pi  # pole phase
        self.zero_mag = 0.9  # zero magnitude
        self.zero_phase = 0.06 * np.pi  # zero phase

        self.b2 = self.zero_mag * self.zero_mag
        self.a2 = self.pole_mag * self.pole_mag

    def _process(self, x):
        # current angle of the pole
        d = self.pole_delta * (1.0 + np.cos(self.omega)) / 2.0
        self.omega += self.phi

        # recompute the filter's coefficients
        self.b1 = -2.0 * self.zero_mag * np.cos(self.zero_phase + d)
        self.a1 = -2.0 * self.pole_mag * np.cos(self.pole_phase + d)

        return 0.3 * (x + self.b1 * self.x.get(1) + self.b2 * self.x.get(2)) - \
               self.a1 * self.y.get(1) - self.a2 * self.y.get(2)


class Tremolo(RTProcessor):
    """ In a tremolo, a slow sinusoidal envelope modulates the signal,
    producing a time-varying change in amplitude
    """
    order = 80

    def __init__(self, rate, channels):
        # tremolo is memoryless
        super().__init__(rate, channels, max_delay=1)

        self.depth = 0.9
        self.phi = 5 * 2 * np.pi / self.SF
        self.omega = 0

    def _process(self, x):
        self.omega += self.phi
        return ((1.0 - self.depth) + self.depth * 0.5 * (1 + np.cos(self.omega))) * x


class Flanger(RTProcessor):
    def __init__(self, rate, channels):
        super().__init__(rate, channels, max_delay=rate)

        self.maxd = 0.015 * self.SF
        self.phi = 0.2 * 2*np.pi / self.SF
        self.omega = 0
        self.a = 0.6

    def _process(self, x):
        self.omega += self.phi
        d = int(self.maxd * (1.0 - np.cos(self.omega)))
        return x if d == 0 else self.a * x + (1.0 - self.a) * self.x.get(d)

""" Signature base class for real-time audio processing

RTProcessor implements a signature class for a simple single-sample processing
unit. Initialize the calss with the sampling rate, the number of channels per
sample (default is 1, i.e. mono audio) and the maximum delay required by the
processing module (e.g. a second order filter will require max_delay=2)

"order" is an attribute that each derived class should redefine to determine
the order of the available classes in an enumeration (useful for user interface)
"""
__author__ = 'Paolo Prandoni'


class RTProcessor:
    # position in list of available classes
    order = 1e6  # menu order

    def __init__(self, rate, channels=1, max_delay=1, max_value=32767):
        self.SF = rate
        self.x = CircularBuffer(max_delay)
        self.y = CircularBuffer(max_delay)
        self.full_scale = max_value

    def process(self, samples):
        for n, x in enumerate(samples):
            y = self._process(x)
            self.x.push(x)
            self.y.push(y)
            #            if y > self.full_scale:
            #                print('overflow', y)
            #            if y < -self.full_scale:
            #                print('underflow', y)
            samples[n] = y
        return samples

    def _process(self, x):
        # this is the function to "override" for each new processor
        return x


# As an example, here is a pass-through processor
class Passthru(RTProcessor):
    order = 0

    def __init__(self, rate, channels):
        super().__init__(rate, channels)


""" Helper class: circular buffer """
import numpy as np


class CircularBuffer(object):
    def __init__(self, length):
        self.length = length
        self.buf = np.zeros(self.length).astype(float)
        self.ix = 0

    def push(self, x):
        self.buf[self.ix] = x
        self.ix = np.mod(self.ix + 1, self.length)

    def get(self, n):
        assert n > 0, 'can only access past values'
        return self.buf[np.mod(self.ix + self.length - n, self.length)]

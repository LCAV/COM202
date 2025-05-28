import numpy as np

from afsk import AFSK


class Transmitter(AFSK):
    def __init__(self, Fs=32000):
        # sampling frequency
        self.Fs = Fs
        # samples per bit
        self.spb = int(Fs / self.BPS)
        assert self.spb == Fs / self.BPS, "incompatible sampling rate, number of samples per bit must be an integer"

    def transmit(self, text):
        # convert string to bitstream
        return self.transmit_bits(''.join(format(ord(i), '08b') for i in text))

    def transmit_random(self, num_bits):
        return self.transmit_bits(''.join(str(i) for i in np.random.randint(0, 2, size=num_bits)))

    def transmit_bits(self, bits, num_times=1):
        # prepare output array
        pilot_head_len = int(self.Fs * self.PILOT_HEAD)
        pilot_tail_len = int(self.Fs * self.PILOT_TAIL)
        preamble_len = pilot_head_len + pilot_tail_len + int(self.Fs * self.GAP_LEN)
        y = np.zeros(preamble_len + len(bits) * self.spb * num_times + self.spb)

        # send pilot head; pilot head must end with phase 0 (hence the flipud)
        pilot_frequency = 2 * np.pi * self.PILOT_FREQ / self.Fs
        y[:pilot_head_len] = np.flipud(np.cos(pilot_frequency * np.arange(0, pilot_head_len)))
        # pilot tail starts with a complete phase reversal
        y[pilot_head_len:pilot_head_len + pilot_tail_len] = -np.cos(pilot_frequency * np.arange(0, pilot_tail_len))

        # send data after gap
        ix = preamble_len
        # SPACE and MARK phase increments
        phase_inc = 2 * np.pi * np.array([self.SPACE_FREQ, self.MARK_FREQ]) / self.Fs
        phase = 0
        for t in range(0, num_times):
            for k in range(0, len(bits)):
                for n in range(0, self.spb):
                    y[ix] = np.cos(phase)
                    ix += 1
                    phase += phase_inc[int(bits[k])]
                    while phase > 2 * np.pi:
                        phase -= 2 * np.pi
        return y


if __name__ == '__main__':
    import sys
    from scipy.io import wavfile

    assert len(sys.argv) == 3, 'usage: transmit.py filename text'

    transmitter = Transmitter()
    x = transmitter.transmit(sys.argv[2])
    resc = np.array(32600 * x, dtype=np.int16)
    wavfile.write(sys.argv[1], 32000, resc)
    print(f'output written to {sys.argv[1]}')
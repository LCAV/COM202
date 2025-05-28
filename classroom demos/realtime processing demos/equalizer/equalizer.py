import numpy as np
import scipy.signal as sp
import sys
import os

if os.name == 'nt':
    WINDOWS = True
    import msvcrt
else:
    import select

import wave
import pyaudio


def poll_keyboard():
    # check for key presses in a platform-independent way
    global WINDOWS
    if WINDOWS:
        key = ord(msvcrt.getch()) if msvcrt.kbhit() else 0
    else:
        key, _, _ = select.select([sys.stdin], [], [], 0)
    return key


class IIR:
    def __init__(self, B, A):
        # flip coefficients
        self.a = np.array(A)[:0:-1] # skip a_0, which we assume to be 1
        self.M = len(A) - 1
        self.ybuf = np.zeros(self.M)
        self.b = np.array(B)[::-1]
        self.N = len(B)
        self.xbuf = np.zeros(self.N - 1)

    def reset(self):
        self.xbuf = self.xbuf * 0.0
        self.ybuf = self.ybuf * 0.0

    def filter(self, x):
        data_len = len(x)
        x = np.r_[self.xbuf, np.array(x)]
        y = np.r_[self.ybuf, np.zeros(data_len)]
        for n in range(0, data_len):
            y[n+self.M] = np.sum(self.b * x[n:n+self.N]) - np.sum(self.a * y[n:n+self.M])
        # update buffer on exit
        if self.N > 0:
            self.xbuf = x[-self.N:]
        self.ybuf = y[-self.M:]
        return y[self.M:]


class EllipticLowpass(IIR):
    def __init__(self, order, cutoff, Fs, ripple=2.5, att=50):
        W = cutoff / (Fs / 2)
        super().__init__(*sp.ellip(order, ripple, att, W))


class EllipticBandpass(IIR):
    def __init__(self, order, center, bw, Fs, ripple=2.5, att=50):
        W = np.array([center - bw / 2, center + bw / 2]) / (Fs / 2)
        super().__init__(*sp.ellip(order, ripple, att, W, btype='bandpass'))


class EllipticHighpass(IIR):
    def __init__(self, order, cutoff, Fs, ripple=2.5, att=50):
        W = cutoff / (Fs / 2)
        super().__init__(*sp.ellip(order, ripple, att, W, btype='highpass'))


if __name__ == '__main__':
    wf = wave.open('resynthesis.wav', 'rb')
    Fs = wf.getframerate()
    print(f'audio sampling rate {Fs} Hz\n')

    # set up filters
    bands = [
        (IIR([1, 0, 0, 0], [1, 0]), 'all frequencies (no equalization)'),
        (EllipticLowpass(3, 150, Fs), 'frequencies < 150Hz'),
        (EllipticBandpass(3, 800, 500, Fs), 'frequencies around 500Hz'),
        (EllipticBandpass(3, 1500, 500, Fs), 'frequencies around 1500Hz'),
        (EllipticHighpass(5, 3000, Fs), 'frequencies over 3KHz'),
        (EllipticLowpass(5, 3000, Fs), 'frequencies below 3KHz'),
    ]
    selected = 0
    
    print('available equalizations:')
    for k, e in enumerate(bands):
        print(f'\t{k}) select {e[1]}')
    print(f'\ncurrently playing {bands[0][1]}')

    # create an audio object
    p = pyaudio.PyAudio()

    # open stream based on the wave object which has been input.
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    assert wf.getnchannels() == 1, 'only use mono files'

    # read data (based on the chunk size)
    chunk = 512 * 4
    data = wf.readframes(chunk)

    # play stream (looping from beginning of file to the end)
    while data != '':
        data = wf.readframes(chunk)
        npdata = np.frombuffer(data, dtype=np.int16) / 32767
        npdata = bands[selected][0].filter(npdata) * 32767
        data = npdata.astype(np.int16).tobytes()
        stream.write(data)
        key = poll_keyboard()
        if key == ord('q'):
            break
        else:
            k = key - ord('0')
            if 0 <= k < len(bands):
                selected = k
                print(f'currently playing {bands[selected][1]}')

    # cleanup stuff.
    stream.close()
    p.terminate()
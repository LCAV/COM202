import pyaudio
import time
import numpy as np
import scipy.signal as sp


class Analyzer:
    def __init__(self, Fs, fmax=20000, resolution=1):
        self.Fs = Fs # sampling rate
        self.res = resolution # frequency resolution
        self.buffer = np.zeros(int(Fs / resolution)) # length of FFT to achieve resolution
        self.kmax = int(fmax / resolution) # FFT bin for max frequency
        self.ix = 0

    def analyze(self, data):
        N = len(self.buffer)
        m = len(data)
        k = 0
        while self.ix + m > N:
            c = N - self.ix
            self.buffer[self.ix:] = data[k:k+c]
            self.ix = 0
            m -= c
            k += c
            # analyze
            X = np.abs(np.fft.fft(self.buffer))
            power = sum(X)
            peak = np.argmax(X[:self.kmax])
            if X[peak] > power / 100:
                print('{}Hz'.format(peak * self.Fs / N), flush=True)
        self.buffer[self.ix:self.ix+m] = data[k:]
        self.ix += m



def callback(in_data, frame_count, time_info, status):
    analyzer.analyze(np.frombuffer(in_data, dtype=np.int16) /32767.0)
    return None, pyaudio.paContinue


WIDTH = 2
CHANNELS = 1
RATE = 32000
CHUNK = 1000

analyzer = Analyzer(RATE)

audio_io = pyaudio.PyAudio()
stream = audio_io.open(format=audio_io.get_format_from_width(WIDTH),
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=False,
                frames_per_buffer=CHUNK,
                stream_callback=callback)

stream.start_stream()
while stream.is_active():
    time.sleep(1)
stream.stop_stream()
stream.close()

audio_io.terminate()


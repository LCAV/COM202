import numpy as np
import scipy.signal as sp

from afsk import AFSK


class Filter:
    # generic class to implement the transfer function H(z) = B(z)/A(z)
    # the filter also computes the output power via a leaky integrator
    def __init__(self, b, a, leak=0.95):
        assert len(a) == len(b), 'must have same number of coefficients in a and b -- zero pad if necessary'
        self.a = a
        self.b = b
        self.N = len(a)
        self.y = [0] * self.N
        self.x = [0] * self.N
        self.ix = 0
        self.value = 0
        self.leak = leak
        self.power = 0

    def compute(self, x):
        self.x[self.ix] = x
        self.value = self.b[0] * x
        for n in range(1, self.N):
            k = (self.ix + self.N - n) % self.N
            self.value += self.b[n] * self.x[k] - self.a[n] * self.y[k]
        self.y[self.ix] = self.value
        self.ix = (self.ix + 1) % self.N
        self.power = self.leak * self.power + (1 - self.leak) * self.value * self.value
        return self.value


class EllipticBandpass(Filter):
    def __init__(self, order, center, bw, Fs, ripple=2.5, att=50, leak=0.95):
        W = np.array([center - bw / 2, center + bw / 2]) / (Fs / 2)
        super().__init__(*sp.ellip(order, ripple, att, W, btype='bandpass'), leak=leak)


class Notch:
    # simple complex-zero-pair notch filter used for phase reversal detection
    def __init__(self, center, Fs):
        self.c = -2 * np.cos(2 * np.pi * center / Fs)
        self.x1 = 0
        self.x2 = 0
        self.value = 0

    def compute(self, x):
        self.value = x + self.c * self.x1 + self.x2
        self.x2 = self.x1
        self.x1 = x
        return self.value


class AGC:
    # automatic gain control circuit
    def __init__(self, target, alpha=0.01):
        self.max_gain = 100
        self.target = target
        self.alpha = alpha
        self.gain = 1

    def set_speed(self, alpha):
        self.alpha = alpha

    def update(self, x):
        self.gain += self.alpha * (self.target - abs(x))
        if self.gain > self.max_gain:
            self.gain = self.max_gain


class Receiver(AFSK):
    # Simple incoherent AFSK receiver. No timing recovery is implemented so keep the bitrate low
    # Correct detection hinges on proper synchronization with the timing reference in the preamble
    #  so if you have burst noise during pilot time, tough luck.

    # state machine
    WAIT_PILOT = 0
    WAIT_SYNC = 10
    WAIT_DATA = 30
    ONLINE = 40

    def __init__(self):
        # received operates at a fixed rate of 32KHz
        self.Fs = 32000
        # samples per bit
        self.spb = self.Fs / self.BPS

        # leaky integrator to compute signal power
        self.signal = Filter([1], [1], leak=0.995)

        # Automatic Gain Control
        self.agc_speed = {
            'fast': 0.005,
            'slow': 0.001,
        }
        self.agc = AGC(1, self.agc_speed['fast'])
        self.agc_wait_len = int(self.Fs * 0.3)

        # pilot and timing
        self.timing_detector = Notch(self.PILOT_FREQ, self.Fs)
        self.reference_power = 0
        self.timing_threshold = -2
        self.timing_reference = [self.timing_threshold, -1]  # threshold, time

        # bandpass filters for PILOT, MARK and SPACE
        self.pilot = EllipticBandpass(3, self.PILOT_FREQ, 700, self.Fs, leak=0.99)
        self.mark = EllipticBandpass(3, self.MARK_FREQ, 900, self.Fs)
        self.space = EllipticBandpass(3, self.SPACE_FREQ, 900, self.Fs)

        # collect incoming bits into bytes
        self.decision = 0
        self.octet = ''

        # state machine
        self.state = self.WAIT_PILOT
        self.timer = self.agc_wait_len
        self.ix = 0

    def restart(self):
        # call this after data is lost to reset the receiver
        self.agc.set_speed(self.agc_speed['fast'])
        self.signal.power = 0
        self.pilot.power = 0
        self.timing_reference = self.timing_reference = [self.timing_threshold, -1]
        self.decision = 0
        self.octet = ''
        self.state = self.WAIT_PILOT
        self.timer = self.agc_wait_len
        self.ix = 0

    def receive(self, x):
        # automatic gain control
        x *= self.agc.gain
        self.agc.update(x)
        self.signal.compute(x)

        # run all filters
        self.pilot.compute(x)
        self.mark.compute(x)
        self.space.compute(x)

        if self.state == self.WAIT_PILOT:
            self.ix = 0
            # wait until the power in the pilot band is more than a quarter of total power
            if self.pilot.power > 0.25 * self.signal.power:
                # start priming the timing detector
                self.timing_detector.compute(x)
                self.timer -= 1
                if self.timer <= 0:
                    # wait until AGC stabilizes and then move on to synch detection
                    self.reference_power = 0.25 * self.pilot.power
                    self.state = self.WAIT_SYNC
                    print(f'pilot detected ({self.ix})')
            else:
                self.timer = self.agc_wait_len

        elif self.state == self.WAIT_SYNC:
            # detect phase reversal and track location of minimum. The idea is that if you have a 180-degree
            #  phase jump, the notch filter's output will shoot down to large negative values. Of course for
            #  this to work you need the signal to be a true sinusoidal pilot.
            #  This is the most fragile step in the detector
            self.timing_detector.compute(x)
            if self.timing_detector.value < self.timing_reference[0]:
                self.timing_reference = [self.timing_detector.value, self.ix]
                # turn off AGC once we detect a sync pulse
                self.agc.set_speed(0)
                print(f'timing reference: {self.timing_reference}')
            if self.pilot.power < self.reference_power:
                if self.timing_reference[1] > 0:
                    # sync was detected; number of samples before data starts:
                    self.timer = int((self.PILOT_TAIL + self.GAP_LEN) * self.Fs) - (self.ix - self.timing_reference[1])
                    self.state = self.WAIT_DATA
                    print(f'pilot end detected at {self.ix}')
                else:
                    print(f'pilot lost')
                    self.restart()

        elif self.state == self.WAIT_DATA:
            # wait until data starts
            self.timer -= 1
            if self.timer <= 0:
                # resume AGC
                self.agc.set_speed(self.agc_speed['slow'])
                self.timer = self.spb
                self.state = self.ONLINE
                print(f'data starts ({self.ix})\n')

        elif self.state == self.ONLINE:
            # accumulate power from bandpass filters and decide on bit value at the end of the symbol period
            # the PC microphone seems to have an unstable characteristic, so we may need to boost
            # one of the filters a bit (not done here)
            self.decision += abs(self.mark.value)
            self.decision -= abs(self.space.value)
            self.timer -= 1
            if self.timer <= 0:
                self.octet += '1' if (self.decision > 0) else '0'
                if len(self.octet) == 8:
                    print(chr(int(self.octet, 2)), end='', flush=True)
                    self.octet = ''
                self.decision = 0
                self.timer += self.spb
                if self.signal.power < self.reference_power:
                    self.restart()
                    print(f'\n\ndata signal lost, waiting for new pilot tone')

        self.ix += 1


if __name__ == '__main__':
    import pyaudio
    import time

    # instantiate the receiver
    receiver = Receiver()

    # set up pyaudio for real-time processing
    def callback(in_data, frame_count, time_info, status):
        for x in np.frombuffer(in_data, dtype=np.int16):
            receiver.receive(x / 32767.0)
        return None, pyaudio.paContinue

    audio_io = pyaudio.PyAudio()
    info = audio_io.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
        if (audio_io.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ", audio_io.get_device_info_by_host_api_device_index(0, i).get('name'))

    stream = audio_io.open(
                    input_device_index=-1,
                    format=audio_io.get_format_from_width(2),   # 16 bits per sample
                    channels=1,                                 # mono input
                    rate=receiver.Fs,
                    input=True,
                    output=True,
                    frames_per_buffer=1024,
                    stream_callback=callback)

    stream.start_stream()
    while stream.is_active():
        time.sleep(1)
    stream.stop_stream()
    stream.close()
    audio_io.terminate()

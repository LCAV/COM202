import numpy as np
import matplotlib.pyplot as plt

EPS = np.finfo(float).eps

def print_array(array: np.ndarray, title=None):
    if title:
        print(title)
    print("Array shape : ", array.shape)
    print(f"Array size : {array.size}")
    print(f"Array nbytes : {array.nbytes}")
    print(f"Array dtype : {array.dtype}")
    print(f"Array max : {array.max()}")
    print(f"Array min : {array.min()}")
    print(f"Array mean : {array.mean()}")
    

def newfigure(*args, **kwargs):
    """Create a new figure with golden ratio."""

    xsize = 10
    ysize = xsize * 2 / (1 + np.sqrt(5))
    fig = plt.figure(figsize=(xsize, ysize), dpi=80)

    plottype = kwargs.get("plottype", "default")
    nsubplots = kwargs.get("nsubplots", 1)

    for nsub in range(nsubplots):
        fig.add_subplot(nsubplots, 1, nsub + 1)

    fig.subplots_adjust(hspace=0.4)

    return fig


def format_axis(ax, *args, **kwargs):
    """Format a figure axis to desired type."""

    plottype = kwargs.get("plottype", "default")
    plottitle = kwargs.get("title", "")

    if plottype == "spectrum":
        fs = kwargs.get("fs")
        ax.set_xlim([-fs / 2, fs / 2])
        ticks = [*range(0, -fs // 2, -5000), *range(5000, fs // 2 + 1, 5000)]
        ax.set_xticks(ticks)
        ax.set_title(plottitle)
        ax.grid(True, which="both")
        ax.set_xlabel("Frequency [Hz]")

    elif plottype == "positivespectrum":
        fs = kwargs.get("fs")
        ax.set_xlim([0, fs / 2])
        ticks = range(0, fs // 2 + 1, 5000)
        ax.set_xticks(ticks)
        ax.set_title(plottitle)
        ax.grid(True, which="both")
        ax.set_xlabel("Frequency [Hz]")

    elif plottype == "indices":
        xmin = kwargs.get("xmin", 0)
        xmax = kwargs.get("xmax", 512)
        ax.set_xlim(xmin - 1, xmax + 1)
        ticks = range(xmin, xmax + 1, (xmax - xmin) // 16)
        ax.set_xticks(ticks)
        ax.set_title(plottitle)
        ax.grid(True, which="both")
        
        

    return ax


def add_db(values):
  """Add power magnitude values."""
  
  powers = []
  for val in values:
    powers.append(np.power(10.0, val / 10.0))
  return 10 * np.log10( np.sum(powers) + EPS)


class SlidingWindow:
    """A fairly basic sliding window on a given numpy array.
    """
    def __init__(self, signal: np.ndarray, shift_size=32, window_size=512) -> None:
        self.signal = signal
        self.window_size = window_size
        self.shift_size = shift_size

        self.cursor = 0
        self.is_end_of_signal = False
        self.window = self.signal[self.cursor : self.cursor + self.window_size]

    def __next__(self):
        self.cursor += self.shift_size
        window = self.signal[self.cursor : self.cursor + self.window_size]
        if self.cursor + self.window_size > len(self.signal):
            # pad with 0
            window = np.pad(window, (0, self.window_size - window.size), "constant")
            self.is_end_of_signal = True
        self.window = window
        return self
    

def smr_bit_allocation(bit_available: int, smr: np.ndarray):
    """Calculate bit allocation in subbands from signal-to-mask ratio."""
    # SNR use pre recorded values. For educational purposes, would be nice to do the real computation
    # REF p. 122 DSP oppenheim
    # fmt: off
    snr = np.array(( 0.00, 7.00,16.00,25.28,31.59,37.75,43.84,49.89,
                         55.93,61.96,67.98,74.01,80.03,86.05,92.01), dtype='float32')
    # fmt: on

    bit_allocation = np.zeros(32, dtype="uint8")
    mnr = snr[bit_allocation[:]] - smr
    while bit_available > 0 and min(bit_allocation) < 15:
        subband_with_min_mnr = np.argmin(mnr)
        if bit_allocation[subband_with_min_mnr] >= 15:
            # Can't have more than 15 bit → putting it to infinity, so we don't come back to it again.
            mnr[subband_with_min_mnr] = np.inf
            continue
        bit_allocation[subband_with_min_mnr] += 1
        bit_available -= 1
        mnr[subband_with_min_mnr] = (
            snr[bit_allocation[subband_with_min_mnr] - 1] - smr[subband_with_min_mnr]
        )

    return bit_allocation

 
def tfplot(s, fs, name):
    """
    Displays a figure window with two subplots.  Above, the signal S is plotted in time domain;
    below, the signal is plotted in frequency domain.

    :param s: signal to be plotted
    :param fs: sampling frequency
    :param name: NAME is the "name" of the signal, e.g., if NAME is 's', then the labels on the y-axes will be
        's(t)' and '|s_F(f)|', respectively.
    :return:
    """
    fig, axs = plt.subplots(2)
   
    axs[0].plot(np.array(range(0, len(s)))/fs, s)
    axs[0].set_xlabel("t [s]")
    axs[0].set_ylabel(name + "(t)")
    #axs[0].set(xlim=(0, len(s)/fs), ylim=(MIN_PLOT_YAXIS, 1))

    s_f =  20*np.log10(np.abs(np.fft.fft(s)))
    freq = np.fft.fftfreq(len(s), 1/fs)
    axs[1].plot(np.fft.fftshift(freq), np.fft.fftshift(s_f * (1/fs)))
    axs[1].set_xlabel("f [Hz]")
    axs[1].set_ylabel("|"+ name + "_F(f)|")

    plt.show()


    

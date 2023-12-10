# %%
import numpy as np
from scipy.io import wavfile
import scipy.signal as sgn
import matplotlib.pyplot as plt
import psychoacoustic
import parameters
from parameters import N_SUBBANDS, SUB_SIZE, FFT_SIZE, EPS, INF, SHIFT_SIZE, FRAME_SIZE

FFT_SIZE = 512


def print_array(array: np.ndarray, name: str = ""):
    if len(name) > 0:
        print(name)
    print("Array shape : ", array.shape)
    print(f"Array size : {array.size}")
    print(f"Array nbytes : {array.nbytes}")
    print(f"Array dtype : {array.dtype}")
    print(f"Array max : {array.max()}")
    print(f"Array min : {array.min()}")
    print(f"Array mean : {array.mean()}")


class SlidingWindow:
    def __init__(self, signal: np.ndarray, shift_size=32, window_size=512) -> None:
        self.signal = signal
        self.window_size = 512
        self.shift_size = shift_size

        self.cursor = 0
        self.is_end_of_signal = False

        self.window = next(self)

    def __next__(self):
        self.cursor += self.shift_size
        window = self.signal[self.cursor : self.cursor + self.window_size]
        if self.cursor + self.window_size > len(self.signal):
            # pad with 0
            window = np.pad(window, (0, self.window_size - window.size), "constant")
            self.is_end_of_signal = True
        self.window = window
        return self

def compute_filter_bank(n_subbands, frame_size, baseband_prototype) -> np.ndarray:
    filter_bank = np.zeros((n_subbands, frame_size), dtype="float32")
    for sb in range(n_subbands):
        for n in range(frame_size):
            # Can be a good exercise to come up with that algorithm
            filter_bank[sb, n] = baseband_prototype[n] * np.cos(
                (2 * sb + 1) * (n - 16) * np.pi / 64
            )

    return filter_bank


def quantize(signal_data, num_bits):
    # Calculate the quantization range
    min_value = np.min(signal_data)
    max_value = np.max(signal_data)
    quantization_range = np.linspace(min_value, max_value, 2**num_bits)

    # Quantize the data by mapping values to the nearest quantization level
    quantized_signal = np.zeros_like(signal_data)
    for i, value in enumerate(signal_data):
        quantized_signal[i] = quantization_range[
            np.argmin(np.abs(quantization_range - value))
        ]

    return quantized_signal


def inverse_quantize(quantized_signal, quantization_range):
    # Inverse quantization by mapping quantized indices back to the original values
    reconstructed_signal = quantization_range[quantized_signal]
    return reconstructed_signal


def subband_synthesis(upsampled_subbands):
    # upsampled_subbands_filtered = np.zeros_like(upsampled_subbands)
    # for sb in range(32):
    #     upsampled_subbands_filtered[sb, :] = np.convolve(
    #         filter_bank[sb], upsampled_subbands[sb]
    #     )[:46336]
    # Cutting 46636 because of convolution padding; TODO LATER
    combined_signal = upsampled_subbands.sum(axis=0)
    return combined_signal


def subband_decompression(quantized_subbands, decimation_factor, bits_per_subband):
    # Inverse quantization
    # print(quantized_subbands[0])
    # Inverse quabtize is useless and should be removed
    num_bits_per_subband = bits_per_subband  # len(quantized_subbands[0].flatten())  # Assumes the same number of bits per subband
    reconstructed_subbands = []
    for subband in quantized_subbands:
        quantization_range = np.linspace(
            np.min(subband), np.max(subband), 2**num_bits_per_subband
        )
        reconstructed_subbands.append(inverse_quantize(subband, quantization_range))
        # reconstructed_subbands = np.asarray([inverse_quantize(subband, quantization_range) for subband in quantized_subbands])
    reconstructed_subbands = np.array(reconstructed_subbands)
    # Subband synthesis
    reconstructed_signal = subband_synthesis(reconstructed_subbands)

    return reconstructed_signal


def quantize_subband_block(
    subband_block: np.ndarray, bit_allocation: np.ndarray
) -> np.ndarray:
    subband_block_quantized = np.zeros_like(subband_block)
    for sb in range(N_SUBBANDS):
        subband_block_quantized[sb] = quantize(subband_block[sb], bit_allocation[sb])
    return subband_block_quantized

def upsample_subbands(sub_tabs: np.ndarray, filter_bank: np.ndarray, nb_sub: int = 32) -> np.ndarray:
    
    up_sub_tabs = np.asarray(
        [np.zeros(len(s) * nb_sub) for s in sub_tabs]
    )
    up_sub_tabs[:, ::nb_sub] = sub_tabs
    up_sub_tabs_filtered = np.zeros_like(up_sub_tabs)
    for sb in range(nb_sub):
        # Remove that and ? and do a convolution manually
        up_sub_tabs_filtered[sb, :] = sgn.lfilter(
            filter_bank[sb, ::-1], 1, up_sub_tabs[sb]
        )

    return nb_sub * up_sub_tabs_filtered


def subbands_filtering(signal: np.ndarray, filter_bank: np.ndarray) -> np.ndarray:
    subbands = np.zeros((32), dtype="float32")
    for sb in range(32):
        # Filter at position 0 = dot product reversed.
        subbands[sb, :] = np.dot(filter_bank, signal[::-1])
    return subbands


def smr_bit_allocation(bit_available: int, smr: np.ndarray):
    """Calculate bit allocation in subbands from signal-to-mask ratio."""
    # SNR use pre recorded values. For educational pruposes, woudl be nice to do th ereal computation
    # REF p. 122 DSP oppenheim
    # fmt: off
    snr = np.array(( 0.00, 7.00,16.00,25.28,31.59,37.75,43.84,49.89,
                         55.93,61.96,67.98,74.01,80.03,86.05,92.01), dtype='float32')
    # fmt: on

    bit_allocation = np.zeros(N_SUBBANDS, dtype="uint8")
    mnr = snr[bit_allocation[:]] - smr
    while bit_available > 0 and min(bit_allocation) < 15:
        # TODO : do an heapqueue ?
        subband_with_min_mnr = np.argmin(mnr)
        if bit_allocation[subband_with_min_mnr] >= 15:
            # Can't have more than 15 bit → putting it to infinity, so we don't come back to it again.
            mnr[subband_with_min_mnr] = np.inf
            continue
        bit_allocation[subband_with_min_mnr] += 1
        bit_available -= 1
        # Update mnr
        # QUesiton : why -1 ?
        mnr[subband_with_min_mnr] = (
            snr[bit_allocation[subband_with_min_mnr] - 1] - smr[subband_with_min_mnr]
        )

    return bit_allocation
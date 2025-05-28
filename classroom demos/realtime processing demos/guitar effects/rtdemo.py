""" Demo program showing how to process audio in real time using Python.
Requires PortAudio and pyaudio.
"""

__author__ = 'Paolo Prandoni'

import pyaudio
import time
import os
import numpy as np
if os.name == 'nt':
    WINDOWS = True
    import msvcrt
else:
    import select
import inspect
import sys


# audio format
RATE = 16000
CHANNELS = 1
CHUNK = 1024

# the module containing the processing classes
PROCESSING_MODULE = "guitar_effects"


def poll_keyboard():
    # check for key presses in a platform-independent way
    global WINDOWS
    if WINDOWS:
        key = ord(msvcrt.getch()) if msvcrt.kbhit() else 0
    else:
        key, _, _ = select.select([sys.stdin], [], [], 0)
    return key


def choice2key(ix):
    return chr(ix + (ord('0') if ix <= 9 else (ord('a') - 10)))


def key2choice(key):
    return key - (ord('0') if key <= ord('9') else (ord('a') - 10))


def print_choices(choices, current):
    # print available processing choices
    print('\n\nnow using processor ', choices[current])
    print("available choices:")
    for ix in choices:
        print(choice2key(ix), ') ', choices[ix])
    print("press 'Q' to quit")


def main():
    chunk = CHUNK
    rate = RATE
    i_device = o_device = 0

    # instantiate pyaudio
    audio_io = pyaudio.PyAudio()

    if len(sys.argv) > 3:
        rate = int(sys.argv[3])
    if len(sys.argv) > 1:
        try:
            i_device = int(sys.argv[1])
            o_device = int(sys.argv[2])
        except:
            print(f'Usage: {sys.argv[0]} [device_id [sampling_rate]]\n')
            print('Available devices:')
            info = audio_io.get_host_api_info_by_index(0)
            numdevices = info.get('deviceCount')
            for i in range(0, numdevices):
                if (audio_io.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                    print("\tInput Device id ", i, " - ", audio_io.get_device_info_by_host_api_device_index(0, i).get('name'))
            for i in range(0, numdevices):
                if (audio_io.get_device_info_by_host_api_device_index(0, i).get('maxOutputChannels')) > 0:
                    print("\tOutput Device id ", i, " - ", audio_io.get_device_info_by_host_api_device_index(0, i).get('name'))
            exit(0)

    print(f'Using devices {i_device} (I), {o_device} (O); sampling rate {rate} Hz; buffer size {chunk} samples\n')

    # scan available processing modules and build a list
    processing_module = __import__(PROCESSING_MODULE)
    p = inspect.getmembers(sys.modules[PROCESSING_MODULE], inspect.isclass)
    p = [(e[1].order, e[0]) for e in p if e[1].__module__ == PROCESSING_MODULE]
    p = sorted(p)
    # add the default class
    choices = {0: "Passthru"}
    for ix, c in enumerate(p):
        choices[ix+1] = c[1]
    print(choices)

    # callback function for the audio pipe. Process data and return
    # the proc variable is user-updated in the main loop
    def callback(in_data, frame_count, time_info, status):
        audio_in = np.array(np.frombuffer(in_data, dtype=np.int16))
        audio_out = np.int16(processor.process(audio_in))
        return audio_out, pyaudio.paContinue

    # open a bidirectional stream; a "frame" is a set of concurrent
    # samples (2 for stereo, 1 for mono) so the frames_per_buffer param
    # gives the size of the input and output buffers
    stream = audio_io.open(
        rate=rate,
        frames_per_buffer=chunk,
        input=True,
        output=True,
        input_device_index=i_device,
        output_device_index=o_device,
        format=audio_io.get_format_from_width(2),  # 16 bits per sample
        channels=1,  # mono input
        stream_callback=callback)

    print("\nstarting audio processing")
    print("press `Q` at any time to quit\n")

    # default processing module is the "no processing"
    current = 0
    processor = getattr(processing_module, choices[current])(RATE, CHANNELS)
    print_choices(choices, current)

    # start recording and playing
    stream.start_stream()
    while stream.is_active():
        key = poll_keyboard()
        if key == ord('Q'):
            break
        else:
            try:
                new_choice = key2choice(key)
                processor = getattr(processing_module, choices[new_choice])(RATE, CHANNELS)
                current = new_choice
                print_choices(choices, current)
            except KeyError:
                pass
        time.sleep(1)

    stream.stop_stream()
    stream.close()
    audio_io.terminate()


if __name__ == '__main__':
    main()


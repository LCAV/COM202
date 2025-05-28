class AFSK:
    # transmission protocol:
    #  - at least 500ms of pilot tone
    #  - phase reversal at waveform peak
    #  - 400ms pilot tail
    #  - 200ms silence
    #  - data
    # by convention, a zero is SPACE, a 1 is MARK
    MARK_FREQ = 1200
    SPACE_FREQ = 2200
    PILOT_FREQ = 400
    BPS = 100  # bits per second
    PILOT_HEAD = 0.5  # seconds
    PILOT_TAIL = 0.4  # seconds
    GAP_LEN = 0.2  # seconds

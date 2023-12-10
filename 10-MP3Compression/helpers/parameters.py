# File containing all the standard-specified data - constants, tables, parameters, as well
# as the EncoderParameters class for handling it.

import sys
import numpy as np

FRAME_SIZE = 512             # Input buffer size
FFT_SIZE   = 512             # FFT number of points
N_SUBBANDS =  32             # Number of subbands
SHIFT_SIZE =  32             # Input buffer shift size
SLOT_SIZE  =  32             # MPEG-1 Layer 1 slot size (minimum unit in a bitstream)
FRAMES_PER_BLOCK = 12        # Number of frames processed in one block
SUB_SIZE = int(FFT_SIZE/2/N_SUBBANDS)

INF = 123456                 # Large number representing infinity
EPS =   1e-6                 # Small number to avoid zero-division in log calculation etc.

DBMIN= -200

UNSET  = 0
TONE   = 1                   # Flags used to denote tonal and noise components
NOISE  = 2
IGNORE = 3





class Tables:
  """Read all the tables necessary for encoding, including the psychoacoustic model tables."""
  
  def __init__(self,fs,bitrate):
    """Select table depending on the sampling frequency. Bitrate is needed for adjustment of minimum hearing threshold."""
    
    if fs == 44100:
      thrtable = 'D1b'
      crbtable = 'D2b'
    elif fs == 32000:
      thrtable = 'D1a'
      crbtable = 'D2a'
    elif fs == 48000:
      thrtable = 'D1c'
      crbtable = 'D2c'


    # Read ISO psychoacoustic model 1 tables containing critical band rates,
    # absolute thresholds and critical band boundaries
    freqband = np.loadtxt('data/tables/' + thrtable, dtype='float32')
    critband = np.loadtxt('data/tables/' + crbtable, dtype='uint16'  )

    self.cbnum  = critband[-1,0] + 1
    self.cbound = critband[:,1]
    
    self.subsize = freqband.shape[0]
    self.line    = freqband[ :,1].astype('uint16')
    self.bark    = freqband[ :,2]
    self.hear    = freqband[ :,3]
    if bitrate >= 96:
      self.hear -= 12

    self.map = np.zeros(FFT_SIZE // 2 + 1, dtype='uint16')
    for i in range(self.subsize - 1):
      for j in range(self.line[i],self.line[i+1]):
        self.map[j] = i
    for j in range(self.line[self.subsize - 1], FFT_SIZE // 2 + 1):
      self.map[j] = self.subsize - 1


def filter_coeffs():
  """Baseband subband filter prototype coefficients."""
  return np.loadtxt('data/tables/LPfilterprototype', dtype='float32')




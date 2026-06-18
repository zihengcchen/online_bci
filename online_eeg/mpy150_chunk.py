# -*- coding: utf-8 -*-
# MPy150 – Chunk-based dynamic channel acquisition using receiveMPData

import os, time
from ctypes import windll, c_int, c_double, c_uint, byref
import numpy as np

# Load the BIOPAC DLL
try:
    mpdev = windll.LoadLibrary('mpdev.dll')
except:
    mpdev = windll.LoadLibrary(os.path.join(os.path.dirname(__file__), 'mpdev.dll'))

def check_returncode(rc):
    return "MPSUCCESS" if rc == 1 else f"ERROR_CODE_{rc}"

class MP150:
    def __init__(self, samplerate=200, channels=None):
        self._samplerate = samplerate
        self._channels = channels or [1, 2, 3]
        self._num_ch = len(self._channels)

        rc = mpdev.connectMPDev(c_int(101), c_int(11), b'auto')
        if check_returncode(rc) != "MPSUCCESS":
            raise Exception(f"Connect failed (rc={rc})")

        rc = mpdev.setSampleRate(c_double(1000.0 / self._samplerate))
        if check_returncode(rc) != "MPSUCCESS":
            raise Exception(f"Set samplerate failed (rc={rc})")

        mask = (c_int * 16)(*[
            1 if (i+1) in self._channels else 0
            for i in range(16)
        ])
        rc = mpdev.setAcqChannels(mask)
        if check_returncode(rc) != "MPSUCCESS":
            raise Exception(f"Set channels failed (rc={rc})")

        # Start acquisition daemon for streaming
        rc = mpdev.startMPAcqDaemon()
        if check_returncode(rc) != "MPSUCCESS":
            raise Exception(f"Start daemon failed (rc={rc})")

        rc = mpdev.startAcquisition()
        if check_returncode(rc) != "MPSUCCESS":
            raise Exception(f"Start acquisition failed (rc={rc})")

    def get_chunk(self, duration_sec=1.0):
        """
        Acquire a data block of length ‘duration_sec’ seconds across all channels.

        Returns:
            np.ndarray of shape (n_samples, num_channels)
        """
        if duration_sec <= 0:
            raise ValueError("duration_sec must be positive.")
        n_samples = max(1, int(round(self._samplerate * duration_sec)))
        total_vals = n_samples * self._num_ch
        buf = (c_double * total_vals)()
        received = c_uint(0)

        rc = mpdev.receiveMPData(buf, c_uint(total_vals), byref(received))
        if check_returncode(rc) != "MPSUCCESS":
            raise Exception(f"receiveMPData failed (rc={rc})")

        if received.value != total_vals:
            print(f"Warning: Only received {received.value}/{total_vals} values")

        received_vals = min(int(received.value), total_vals)
        complete_vals = received_vals - (received_vals % self._num_ch)
        if complete_vals != received_vals:
            print(
                f"Warning: Dropping {received_vals - complete_vals} incomplete channel values"
            )
        if complete_vals == 0:
            return np.empty((0, self._num_ch), dtype=np.float64)

        arr = np.ctypeslib.as_array(buf)[:complete_vals].copy()
        return arr.reshape((-1, self._num_ch))

    def close(self):
        rc = mpdev.stopAcquisition()
        if check_returncode(rc) != "MPSUCCESS":
            print(f"Warning: stopAcquisition failed (rc={rc})")

        rc = mpdev.disconnectMPDev()
        if check_returncode(rc) != "MPSUCCESS":
            raise Exception(f"Disconnect failed (rc={rc})")

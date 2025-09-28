"""
Created on June 2024

@author: a23marmo

Functions to listen to audio signals and spectrograms.
Also contains a function to compute the SDR between two audio signals.
These wrappers are useful to avoid importing IPython.display and mir_eval in the main scripts and rewrite the same code.
"""
import IPython.display as ipd
import mir_eval
import numpy as np

import base_audio.spectrogram_to_signal as spec_to_sig

# %% Audio listening
def listen_to_this_spectrogram(spectrogram, feature_object, phase_retrieval = "griffin_lim", original_phase = None):
    """
    Inverts the spectrogram using the istft method, and plots the audio using IPython.diplay.audio.

    Parameters
    ----------
    spectrogram : numpy array
        The spectrogram to be inverted.
    feature_object : object
        The feature object used to compute the spectrogram.
        See signal_to_spectrogram.py for more details about the Feature object.
    phase_retrieval : string
        The method used to retrieve the phase information.
        Must be either "griffin_lim" or "original_phase".
        - "griffin_lim": uses the Griffin-Lim algorithm to retrieve the phase.
        - "original_phase": uses the original phase information of the signal.
    original_phase : numpy array
        The original phase information of the signal.
        Necessary if phase_retrieval is set to "original_phase".
        Useless if phase_retrieval is set to "griffin_lim".

    Returns
    -------
    IPython.display audio
        The audio signal from the inverted spectrogram.
    """
    signal = spec_to_sig.spectrogram_to_audio_signal(spectrogram, feature_object=feature_object, phase_retrieval = phase_retrieval, original_phase=original_phase)
    return listen_to_this_signal(signal, feature_object.sr)

def listen_to_this_signal(signal, sr=44100):
    """
    Returns the IPython wrapper for listening to the audio signal.
    """
    return ipd.display(ipd.Audio(data=signal, rate=sr))

# %% Audio evaluation
def compute_sdr(audio_ref, audio_estimate):
    """
    Function encapsulating the SDR computation in mir_eval.
    SDR is computed between 'audio_ref' and 'audio_estimate'.
    """
    if (audio_estimate == 0).all():
        return -np.inf
    return mir_eval.separation.bss_eval_sources(np.array([audio_ref]), np.array([audio_estimate]))[0][0]
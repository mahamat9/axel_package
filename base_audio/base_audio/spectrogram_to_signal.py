"""
Created on June 2024

@author: a23marmo

Functions to convert spectrograms to audio signals.
These functions are really useful to listen to what the spectrogram represents.
"""

import base_audio.errors as err
import librosa

# TODO: handle CQT

# %% Audio to signal conversion
def spectrogram_to_audio_signal(spectrogram, feature_object, phase_retrieval = "griffin_lim", original_phase=None):
    """
    Converts a spectrogram to an audio signal using the inverse Short-Time Fourier Transform (iSTFT).

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
    numpy array
        The audio signal from the inverted spectrogram.
    """
    if feature_object.feature in ["stft", "stft_complex"]:
        spectrogram_stft = spectrogram
    elif feature_object.feature in ["mel", "log_mel", "nn_log_mel"]:
        spectrogram_stft = feature_object.get_stft_from_mel(spectrogram)
    else:
        raise err.InvalidArgumentValueException(f"Feature representation not handled for audio reconstruction: {feature_object.feature}.")

    if phase_retrieval == "griffin_lim":
        return librosa.griffinlim(spectrogram_stft, hop_length = feature_object.hop_length, random_state = 0)
        
    elif phase_retrieval == "original_phase":
        assert original_phase is not None, "Original phase is necessary for phase retrieval using the original_phase."
        assert original_phase.shape == spectrogram_stft.shape, "Original phase and spectrogram must have the same shape."
        complex_spectrogram_to_inverse = spectrogram_stft * original_phase
        return complex_stft_to_audio(complex_spectrogram_to_inverse, feature_object.hop_length)

    else:
        raise err.InvalidArgumentValueException(f"Phase retrieval method not understood: {phase_retrieval}.")
    
def complex_stft_to_audio(stft_to_inverse, hop_length):
    """
    Wrapper to the inverse Short-Time Fourier Transform (iSTFT) from librosa.
    """
    return librosa.istft(stft_to_inverse, hop_length = hop_length)


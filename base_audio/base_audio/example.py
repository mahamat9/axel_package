"""
Example file to compute a spectrogram.
"""
# %% Import the spectrogram computation library
import base_audio.signal_to_spectrogram as sig_to_spec

# %% Load the file
file_path = "/home/a23marmo/datasets/Chanson perso/All Them Witches/All Them Witches - 1X1/All Them Witches - 1X1 - 01 1X1.flac" #TODO: To change

# %% Define the feature object
feature_object = sig_to_spec.FeatureObject(sr=44100, feature="mel", n_fft=2048, hop_length=512, n_mels=128, fmin=0, fmax=22050)

# %% Compute the spectrogram
spectrogram = feature_object.load_file_compute_spectrogram(file_path)

## Can similarly be done using:
# signal, sr = librosa.load(file_path, sr=44100, mono=True)
# spectrogram = feature_object.get_spectrogram(signal)

print(spectrogram.shape)

# %% Plot the spectrogram
from base_audio.common_plot import *
plot_me_this_spectrogram(spectrogram)

# %% Audio to signal conversion
import base_audio.spectrogram_to_signal as spec_to_sig  
import base_audio.audio_helper as audio_helper

recons_signal = spec_to_sig.spectrogram_to_audio_signal(spectrogram, feature_object, phase_retrieval="griffin_lim")
audio_helper.listen_to_this_signal(recons_signal) # Doesn't work in VScode, use a Jupyter notebook instead.

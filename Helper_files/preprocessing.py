import pandas as pd
import os
import scipy.io
import numpy as np
from scipy.signal import find_peaks
##################################### HELPER FUNCTIONS RELATED TO DATA PREPROCESSING ##############################################
### Function to Import the mat files in the folders -  

def load_mat_files_to_dfs(folder_path, start_with='damaged', end_with='.mat'):
    data_frames = {}

    for file_name in os.listdir(folder_path):
        if file_name.startswith(start_with) and file_name.endswith(end_with):
            parts = file_name.split('_')
            if len(parts) < 3:
                continue  # Skip files that do not follow expected naming convention
            
            damage_state = parts[1]  # Extract damage state from filename

            if damage_state not in data_frames:
                data_frames[damage_state] = []

            file_path = os.path.join(folder_path, file_name)
            mat = scipy.io.loadmat(file_path)
            # print(mat.values())
            data = {key: value for key, value in mat.items() if not key.startswith('__')}
            # print(data)
            
            first_key = list(data.keys())[0]  # Get the first variable name
            df = pd.DataFrame(data[first_key])  # Convert to DataFrame
            # print(f"Loaded {file_name} ({first_key}): shape {df.shape}")
            data_frames[damage_state].append(df)

    # Concatenate lists of DataFrames into single DataFrame per damage state
    df_dict = {key: pd.concat(value, ignore_index=True, axis=1) for key, value in data_frames.items() if value}

    return df_dict

# Function Check
# folder_path = 'car_new_with_correct_indices\\mat_files'
# dfs = load_mat_files_to_dfs(folder_path)
# undamaged = dfs["0"]
# damaged_1 = dfs["1"]
# damaged_2 = dfs["2"]
# print(undamaged.shape, damaged_1.shape, damaged_2.shape)


# function to segment the signals into equal windows of length window_size and return a dataframe of the segmented signals
def segment_signals(df, window_size=2000, step_size=10):
  segmented_data = []
  for col in df.columns:
    segments = []
    for i in range(0, df.shape[0] - window_size + 1, step_size):
      segment = df[col].iloc[i:i + window_size]
      if segment.isnull().values.any():
        continue  # Skip segments with NaN values
      segments.append(segment.values)
    segmented_data.append(pd.DataFrame(segments).T)
  return pd.concat(segmented_data, axis=1)

#Function Check
# undamaged_segmented = segment_signals(undamaged)
# print(undamaged_segmented.shape)

### Function to get the stats of the dataframes, any dataframe with the examples along the axis =1, 
# return the stats at each time step or the frequency point
def get_stats(df):
  stats = {
    'Peak': df.abs().max(axis=1),
    'Mean': df.mean(axis=1),
    'Variance': df.var(axis=1),
    'Standard Deviation': df.std(axis=1),
    'Median': df.median(axis=1)
  }
  return stats



#Function Check
# stats = get_stats(df)
# print(stats)

### Function to convert a single signal to FFT
def convert_to_fft(signal, Fs):
  n = len(signal)
  fft = np.fft.fft(signal)
  fft = np.abs(fft[:n // 2])  # Take the magnitude of the first half of the FFT
  freqs = np.fft.fftfreq(n, d=1/Fs)[:n // 2]  # Corresponding frequencies
  return freqs, fft

#Function Check
# freqs, fft = convert_to_fft(df.iloc[:,0],2000)
# print(freqs.shape, fft.shape)

### Function to normalize the signals in a dataframe using the mean and standard deviation from the get_stats function
# the output is a df with mean 0 and std 1
# stats parameter is the output of the get_stats function
def normalize(df, stats):
  for i in range(df.shape[1]):
    df.iloc[:, i] = (df.iloc[:, i] - stats['Mean']) / stats['Standard Deviation']
  return df

#Function Check
# df = normalize(df, stats)
# print(df.shape)

### Function to find the first n peaks from the FFT of the signals in a dataframe of ffts
# the output is a list of the frequencies of the peaks
# Inputs are the ffts and the corresponding frequencies from the convert_to_fft function and n = no. of peaks to find
# by default n=5 for our case
def find_peaks(fft, freqs, n=5, distance = 1):
  peak_indices, _ = find_peaks(fft, distance=distance)
  peak_indices = peak_indices[np.argsort(fft[peak_indices])][-n:]
  frequency_peaks = freqs[peak_indices]
  return frequency_peaks

#Function Check
# fft = np.random.rand(100)
# freqs = np.random.rand(100)
# peaks = find_peaks(fft, freqs)
# print(peaks)
# def resample(df, new_length):
#   old_length = df.shape[0]
#   return df.iloc[::old_length//new_length]


# def spectrogram(df, Fs, nperseg=256, noverlap=128):
#   f, t, Sxx = scipy.signal.spectrogram(df, fs=Fs, nperseg=nperseg, noverlap=noverlap)
#   return f, t, Sxx

from scipy.signal import spectrogram

def generate_spectrogram(data, fs=2000, nperseg=100, noverlap=99):
    """
    Converts time-series data into spectrogram representation.
    
    Parameters:
        data (numpy array): Shape (2000, num_samples), time-series data.
        fs (float): Sampling frequency, default is 2000 Hz.
        nperseg (int): Length of each segment for STFT.
        noverlap (int): Overlapping points between segments.
        
    Returns:
        numpy array: Spectrogram representation (frequency, time, samples)
    """
    num_samples = data.shape[1]
    spectrograms = []
    
    for i in range(num_samples):
        f, t, Sxx = spectrogram(data[:, i], fs=fs, nperseg=nperseg, noverlap=noverlap)
        valid_idx = (f >= 1)  # Keep frequencies from 1 Hz to Nyquist frequency
        f = f[valid_idx]
        Sxx = Sxx[valid_idx, :]
        spectrograms.append(Sxx)
    
    spectrograms = np.array(spectrograms)  # Shape (num_samples, freq_bins, time_bins)
    spectrograms = np.moveaxis(spectrograms, 0, -1)  # Shape (freq_bins, time_bins, num_samples)
    
    print(f"Frequency range: {f.min()} Hz to {f.max()} Hz, Total bins: {len(f)}")
     
    return [f, t ,spectrograms]

# # Example usage
# time_series_data = np.random.randn(2000, 100)  # Dummy data (2000 timesteps, 100 samples)
# spectrogram_data = generate_spectrogram(time_series_data)
# print("Spectrogram shape:", spectrogram_data.shape)


### List the functions in this module when someone imports it
__all__ = [
  'load_mat_files_to_dfs',  # folder_path, start_with='damaged', end_with='.mat'
  'get_stats',  # df
  'convert_to_fft',  # signal, Fs
  'normalize',  # df, stats
  'find_peaks', # fft, freqs, n=5
  'segment_signals', # df, window_size=2000, step_size=10
  'generate_spectrogram' # data, fs=2000, nperseg=100, noverlap=99
]
print(__all__)

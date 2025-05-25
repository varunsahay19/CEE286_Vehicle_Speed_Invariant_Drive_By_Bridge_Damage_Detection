import preprocessing as PP
import os
import numpy as np
import pandas as pd


folder_path = "CEE286_Project\segmented_npy_files"
undamaged_df = None
damaged_1_df = None
damaged_2_df = None

for file in os.listdir(folder_path):
  if file.endswith(".npy"):
    if file.startswith("undamaged"):
        undamaged_df = np.load(os.path.join(folder_path, file))
    elif file.startswith("damaged_1"):
        damaged_1_df = np.load(os.path.join(folder_path, file))
    elif file.startswith("damaged_2"):
        damaged_2_df = np.load(os.path.join(folder_path, file))

print(undamaged_df.shape, damaged_1_df.shape, damaged_2_df.shape)

# [f, t ,spectrograms] = def generate_spectrogram(data, fs=2000, nperseg=100, noverlap=99)

# Function to generate spectrograms from the segmented signals
WINDOW = 500
OVERLAP = 400
undamaged_list = PP.generate_spectrogram(undamaged_df, fs=2000, nperseg=WINDOW, noverlap=OVERLAP)
damaged_1_list = PP.generate_spectrogram(damaged_1_df, fs=2000, nperseg=WINDOW, noverlap=OVERLAP)
damaged_2_list = PP.generate_spectrogram(damaged_2_df, fs=2000, nperseg=WINDOW, noverlap=OVERLAP)

undamaged_spectrograms = undamaged_list[2]
damaged_1_spectrograms = damaged_1_list[2]
damaged_2_spectrograms = damaged_2_list[2]

f_undamaged = undamaged_list[0]
t_undamaged = undamaged_list[1]
f_damaged_1= damaged_1_list[0]
t_damaged_1 = damaged_1_list[1]
f_damaged_2 = damaged_2_list[0]
t_damaged_2 = damaged_2_list[1]

print("Pre-Saving Shapes: ", undamaged_spectrograms.shape, damaged_1_spectrograms.shape, damaged_2_spectrograms.shape)


save_path = "CEE286_Project\\segmented_spectrograms"
if not os.path.exists(save_path):
    os.makedirs(save_path)

np.save(os.path.join(save_path, "undamaged_spectrograms_w_500_o_400.npy"), undamaged_spectrograms)
print("Saved Undamaged Spectrograms of Shape: ", undamaged_spectrograms.shape)
np.save(os.path.join(save_path, "damaged_1_spectrograms_w_500_o_400.npy"), damaged_1_spectrograms)
print("Saved Undamaged Spectrograms of Shape: ", undamaged_spectrograms.shape)
np.save(os.path.join(save_path, "damaged_2_spectrograms_w_500_o_400.npy"), damaged_2_spectrograms)
print("Saved Undamaged Spectrograms of Shape: ", undamaged_spectrograms.shape)

np.save(os.path.join(save_path, "t_undamaged_w_500_o_400.npy"), t_undamaged)
np.save(os.path.join(save_path, "f_undamaged_w_500_o_400.npy"), f_undamaged)
np.save(os.path.join(save_path, "t_damaged_1_w_500_o_400.npy"), t_damaged_1)
np.save(os.path.join(save_path, "f_damaged_1_w_500_o_400.npy"), f_damaged_1)
np.save(os.path.join(save_path, "t_damaged_2_w_500_o_400.npy"), t_damaged_2)
np.save(os.path.join(save_path, "f_damaged_2_w_500_o_400.npy"), f_damaged_2)


# check shapes on loading 
undamaged_spectrograms = np.load(os.path.join(save_path, "undamaged_spectrograms_w_500_o_400.npy"))
damaged_1_spectrograms = np.load(os.path.join(save_path, "damaged_1_spectrograms_w_500_o_400.npy"))
damaged_2_spectrograms = np.load(os.path.join(save_path, "damaged_2_spectrograms_w_500_o_400.npy"))

t_undamaged = np.load(os.path.join(save_path, "t_undamaged_w_500_o_400.npy"))
f_undamaged = np.load(os.path.join(save_path, "f_undamaged_w_500_o_400.npy"))

t_damaged_1 = np.load(os.path.join(save_path, "t_damaged_1_w_500_o_400.npy"))
f_damaged_1 = np.load(os.path.join(save_path, "f_damaged_1_w_500_o_400.npy"))

t_damaged_2 = np.load(os.path.join(save_path, "t_damaged_2_w_500_o_400.npy"))
f_damaged_2 = np.load(os.path.join(save_path, "f_damaged_2_w_500_o_400.npy"))

print("Loaded Spectrogram Files: " , undamaged_spectrograms.shape, damaged_1_spectrograms.shape, damaged_2_spectrograms.shape)
print("Loaded t and f files: " , t_undamaged.shape, f_undamaged.shape, t_damaged_1.shape, f_damaged_1.shape, t_damaged_2.shape, f_damaged_2.shape)
import pandas as pd
import os
import shutil

#directories, paths, etc
base_dir = r'C:\Users\khafa\Downloads\UrbanSound8K\UrbanSound8K'
audio_dir = os.path.join(base_dir, 'audio')
metadata_path = os.path.join(base_dir, 'metadata', 'UrbanSound8K.csv')
output_dir = os.path.join(base_dir, 'street_music_samples')

#output directory
os.makedirs(output_dir, exist_ok=True)

#load metadata
metadata = pd.read_csv(metadata_path)

#grab the entries that have the class tag "street_music"
street_music_data = metadata[metadata['class'] == 'street_music']

#log for errors/missing files :/
#no such missing files because of destination fold path workaround
missing_files = []

#copy each file to the new directory
for _, row in street_music_data.iterrows():
    fold = f"fold{row['fold']}" #search our folders
    file_name = row['slice_file_name'] #grabs the name of the file

    #source and destination paths
    source_path = os.path.join(audio_dir, fold, file_name)

    #actually creates 10 folders inside street_music_samples
    #that should each have 100 samples
    #created to get around an error where a mismatch in the csv file
    #and the actual data would break the script
    destination_fold_path = os.path.join(output_dir, fold)
    destination_path = os.path.join(output_dir, fold, file_name)

    #make sure the target directory is there
    os.makedirs(destination_fold_path, exist_ok=True)

    #copy the sought after files
    if os.path.exists(source_path):
        shutil.copy(source_path, destination_path)
        print(f"Copied {file_name} to {destination_path}")
    else:
        print(f"File {file_name} not found in {source_path}")
        missing_files.append(source_path)


print(f"Finished copying {len(street_music_data) - len(missing_files)} street music samples to {output_dir}")
print(f"{len(missing_files)} missing files were missing and couldn't be copied")

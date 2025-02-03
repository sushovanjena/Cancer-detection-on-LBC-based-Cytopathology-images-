import csv
import os
import numpy as np
import struct
from natsort import natsorted


def read_32i_file(file_path):
    """Function to read 32-bit unsigned integers from a .32i file."""
    with open(file_path, 'rb') as file:
        data = file.read()
        num_integers = len(data) // 4
        integers = struct.unpack('I' * num_integers, data)
    return integers

def create_csv_from_folders(root_folder_path, csv_path, metadata_csv_path, features_folder_path):  
    # Create a dictionary to store binary labels
    binary_labels_mapping = {}

    # Load the metadata CSV file with binary labels
    with open(metadata_csv_path, 'r') as metadata_csv:
        metadata_reader = csv.DictReader(metadata_csv)
        for row in metadata_reader:
            # Remove the file extension from the image_id
            image_id_without_extension = row['CASE-ID']
            # binary_labels_mapping[image_id_without_extension] = row['LABEL']

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row
        writer.writerow(['Folder_Name', 'Label', 'WebP_Image_Path'])

        # Iterate over the root folder and its subfolders
        for parent_folder in os.listdir(root_folder_path):
            parent_folder_path = os.path.join(root_folder_path, parent_folder)
            print("parent_folder_path",parent_folder_path)
            if not os.path.isdir(parent_folder_path):
                continue

            # Iterate over the subfolders inside the parent folder
            for subfolder1 in os.listdir(parent_folder_path):
                subfolder1_path = os.path.join(parent_folder_path, subfolder1)
                print("subfolder1_path",subfolder1_path)
                if not os.path.isdir(subfolder1_path):
                    continue


                for subfolder2 in os.listdir(subfolder1_path):
                    subfolder2_path = os.path.join(subfolder1_path, subfolder2)
                    print("subfolder2_path",subfolder2_path)
                    if not os.path.isdir(subfolder2_path):
                        continue
                
                    # for subfolder3 in os.listdir(subfolder2_path):
                    #     subfolder3_path = os.path.join(subfolder2_path, subfolder3)
                    #     print("subfolder3_path",subfolder3_path)
                    #     if not os.path.isdir(subfolder3_path):
                    #         continue

                    target_folders = natsorted(os.listdir(subfolder2_path))
                    target_folders.reverse()  # Sort target folders in descending order
                    
                    # Get the highest target folder path
                     # Get the highest target folder path that is a directory
                    highest_target_folder_path = None
                    for target_folder in target_folders:
                        potential_target_folder_path = os.path.join(subfolder2_path, target_folder)
                        # print("potential_target_folder_path",potential_target_folder_path)
                        if os.path.isdir(potential_target_folder_path):
                            highest_target_folder_path = potential_target_folder_path
                            print("highest_target_folder_path",highest_target_folder_path)
                            break
                    # if target_folders:
                    #     highest_target_folder = target_folders[0]
                    #     highest_target_folder_path = os.path.join(subfolder2_path, highest_target_folder)

                        # Get all .webp images in the highest target folder path
                    if highest_target_folder_path is not None:
                        # Get all .jpeg images in the highest target folder path
                        jpeg_images = [f for f in os.listdir(highest_target_folder_path) if f.endswith('.jpeg')]
                        # print("jpeg_images:", jpeg_images)
                        jpeg_images_paths = natsorted([os.path.join(highest_target_folder_path, img) for img in jpeg_images])
                    else:
                        print(f"Not a directory: {highest_target_folder_path}")
                        

                    # # Get the feature file path based on folder name
                    # feature_file_name = parent_folder + ".npy"  # Assuming feature file extension is ".npy"
                    # feature_file_path = os.path.join(features_folder_path, parent_folder, subfolder1, highest_target_folder)
                    # print(feature_file_path)
                    # Find the file ending with annot_idxs.32i in the 3rd nested folder
                    # Find the annot_idxs.32i file path for the current folder
                    

                        # Find the annot_idxs.32i file path for the current folder
                    # feature_subfolder_path = os.path.join(features_folder_path, parent_folder, subfolder1, 'features')
                    annot_file_path = None
                    for file in os.listdir(subfolder2_path):
                        if file.endswith('annot_idxs.32i'):
                            annot_file_path = os.path.join(subfolder2_path, file)
                            break

                    # Read 32-bit unsigned integer labels from the found annot_idxs.32i file
                    if annot_file_path and os.path.exists(annot_file_path):
                        labels = read_32i_file(annot_file_path)
                    else:
                        labels = ['N/A'] * len(jpeg_images_paths)

                    # Write rows for each WebP image path
                    for idx, webp_image_path in enumerate(jpeg_images_paths):
                        label = labels[idx] if idx < len(labels) else 'N/A'
                        writer.writerow([parent_folder, label, webp_image_path])
    print(f"CSV file created successfully at {csv_path}.")

# Specify the root folder path, metadata CSV file path, and output CSV file path
root_folder_path = '/media/dataset/annot_cases_cpap/data/cpap/May10_2018'
metadata_csv_path = '/media/dataset/annot_cases_cpap/data/caseid_file.csv'
features_folder_path = '/media/dataset/annot_cases_cpap/data/cpap/Pap_Smear_slides.svs_format-11th_July_2018-100_slides'
csv_path = '/media/dataset/annot_cases_cpap/data/meta_data6.csv'

# Call the function to create the CSV file
create_csv_from_folders(root_folder_path, csv_path, metadata_csv_path, features_folder_path)

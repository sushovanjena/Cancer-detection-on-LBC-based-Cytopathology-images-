# import pandas as pd
# from sklearn.model_selection import train_test_split

# # Load the CSV file
# df = pd.read_csv('/media/dataset/annot_cases_cpap/data/final_metadata_otsu.csv')

# # Splitting the data into train and test sets while stratifying by the label column
# train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['Label'], random_state=42)

# # Save the test dataset to a new CSV file
# test_df.to_csv('/media/dataset/annot_cases_cpap/data/final_test_data_otsu.csv', index=False)

# # Save the train dataset to a new CSV file
# train_df.to_csv('/media/dataset/annot_cases_cpap/data/final_train_data_otsu.csv', index=False)

import pandas as pd
import numpy as np
import cv2

def filter_add_and_save_with_thresholding(csv_file, output_file, image_path_column):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Ensure the 'Label' column exists
    if 'Label' not in df.columns:
        print("The 'label' column does not exist in the CSV file.")
        return

    # Filter rows where 'label' is 1
    ones_df = df[df['Label'] == 1]
    
    # Shuffle zero-label rows
    zeros_df = df[df['Label'] == 0].sample(frac=1).reset_index(drop=True)

    # Initialize a list to collect valid zero-label samples
    valid_zeros = []

    # Number of zeros needed to match ones
    needed_zeros_count = len(ones_df)
    
    # Iterate through shuffled rows with 'Label' 0
    for _, row in zeros_df.iterrows():
        if len(valid_zeros) >= needed_zeros_count:
            break  # Stop processing if enough valid zeros have been found

        img_path = row[image_path_column]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            if np.std(img) < 10:  # Check if the image has very low variance
                continue  # Skip this image if almost no variation

            _, thresh_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            percent_above_thresh = np.sum(thresh_img == 255) / thresh_img.size

            # Include images with a high percentage of foreground or very low background
            if percent_above_thresh > 0.5 or percent_above_thresh < 0.05:
                valid_zeros.append(row)

    # Convert list of valid rows to DataFrame
    valid_zeros_df = pd.DataFrame(valid_zeros)
    
    # Randomly select an equal number of rows with 'label' 0, if more are collected than needed
    sampled_zeros_df = valid_zeros_df.sample(n=len(ones_df), random_state=1)

    # Concatenate the filtered rows
    final_df = pd.concat([ones_df, sampled_zeros_df])
    
    # Save the final DataFrame to a new CSV file
    final_df.to_csv(output_file, index=False)
    print(f"Filtered and sampled rows saved to {output_file}")

# Example usage
csv_file = '/media/dataset/annot_cases_cpap/data/FINAL_MERGED_DATA_new.csv'
output_file = '/media/dataset/annot_cases_cpap/data/final_metadata_otsu.csv'
image_path_column = 'WebP_Image_Path'  # Assuming the column name that contains image paths
filter_add_and_save_with_thresholding(csv_file, output_file, image_path_column)






# import pandas as pd

# def count_ones_in_label(csv_file):
#     # Read the CSV file
#     df = pd.read_csv(csv_file)
    
#     # Ensure the 'label' column exists
#     if 'Label' in df.columns:
#         # Count the number of 1's in the 'label' column
#         count = df['Label'].sum()
#         print(f"Number of 1's in the 'label' column: {count}")
#     else:
#         print("The 'label' column does not exist in the CSV file.")

# # Example usage
# csv_file = '/media/dataset/annot_cases_cpap/data/FINAL_TEST_DATA_new.csv'
# count_ones_in_label(csv_file)

# import pandas as pd
# import numpy as np

# def filter_add_and_save(csv_file, output_file):
#     # Read the CSV file
#     df = pd.read_csv(csv_file)
    
#     # Ensure the 'label' column exists
#     if 'Label' in df.columns:
#         # Filter rows where 'label' is 1
#         ones_df = df[df['Label'] == 1]
        
#         # Filter rows where 'label' is 0
#         zeros_df = df[df['Label'] == 0]
        
#         # Check if there are enough rows with 'label' 0
#         if len(zeros_df) >= len(ones_df):
#             # Randomly select an equal number of rows with 'label' 0
#             sampled_zeros_df = zeros_df.sample(n=len(ones_df), random_state=1)
#         else:
#             print("Not enough rows with 'label' 0 to match the number of rows with 'label' 1.")
#             return
        
#         # Concatenate the filtered rows
#         final_df = pd.concat([ones_df, sampled_zeros_df])
        
#         # Save the final DataFrame to a new CSV file
#         final_df.to_csv(output_file, index=False)
#         print(f"Filtered and sampled rows saved to {output_file}")
#     else:
#         print("The 'label' column does not exist in the CSV file.")

# # Example usage
# csv_file = '/media/dataset/annot_cases_cpap/data/FINAL_MERGED_DATA_new.csv'
# output_file = '/media/dataset/annot_cases_cpap/data/final_metadata.csv'
# filter_add_and_save(csv_file, output_file)

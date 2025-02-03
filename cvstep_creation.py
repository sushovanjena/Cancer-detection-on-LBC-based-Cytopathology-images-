import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os
from sklearn.utils import resample

# Load the metadata
metadata = pd.read_csv('/media/dataset/annot_cases_cpap/data/final_train_data_otsu.csv')

# Separate positive and negative cases
positive_cases = metadata[metadata['Label'] == 1]
negative_cases = metadata[metadata['Label'] == 0]

# Initialize StratifiedKFold for positive cases
skf_positive = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Specify the folder to save the CSV files
folder_name = '/media/dataset/annot_cases_cpap/data/train_folds_otsu'
os.makedirs(folder_name, exist_ok=True)

# Split the positive cases into folds
for fold, (_, test_index) in enumerate(skf_positive.split(positive_cases, positive_cases['Label'])):
    # Split the positive cases
    positive_cases_fold = positive_cases.iloc[test_index]
    
    # Combine positive cases with all negative cases
    balanced_train_data = pd.concat([positive_cases_fold, negative_cases])
    
    # Sample a subset of negative cases equal to the size of the positive cases
    sampled_negative_cases = resample(negative_cases, 
                                      replace=True,  # sample with replacement
                                      n_samples=len(positive_cases_fold),  # equal to the size of the positive cases in this fold
                                      random_state=123)  # reproducible results
    
    # Combine positive and resampled negative cases
    balanced_train_data = pd.concat([positive_cases_fold, sampled_negative_cases])
    
    # Specify the path for the CSV file
    file_path = os.path.join(folder_name, f'train_fold_{fold+1}.csv')
    
    # Save the split data into CSV files
    balanced_train_data.to_csv(file_path, index=False)



# meta_data_csv = pd.read_csv('/media/Data/lung_cancer_dataset/meta_data.csv')

# CV_Step_prostate_cancer_path = '/media/Data/lung_cancer_dataset/testcsv'

# num_rows_per_file = len(meta_data_csv) // 5

# smaller_dfs = [meta_data_csv[i*num_rows_per_file:(i+1)*num_rows_per_file]for i in range(5)]

# for i,df in enumerate(smaller_dfs):
#     df.to_csv(os.path.join(CV_Step_prostate_cancer_path, f'prostrate_cvstep_without_mask_fold_{i}.csv'), index = False)
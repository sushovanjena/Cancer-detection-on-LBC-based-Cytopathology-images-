import pandas as pd
import sftp_utils
import os

csv_path = os.path.join(sftp_utils.sftp_path, "media/dataset/annot_cases_cpap/data/FINAL_MERGED_DATA.csv")
# Load the CSV file
df = pd.read_csv(csv_path)

# Filter the rows based on Folder_Name, e.g., 'C15809-14.svs'
folder_name_to_filter = 'C19538-15.svs'
filtered_df = df[df['Folder_Name'] == folder_name_to_filter]
output_path = os.path.join(sftp_utils.sftp_path, "media/dataset/annot_cases_cpap/data/filtered_output.csv")
# Save the filtered data to a new CSV file
filtered_df.to_csv(output_path, index=False)

print(f"Filtered data saved to 'filtered_output.csv'.")

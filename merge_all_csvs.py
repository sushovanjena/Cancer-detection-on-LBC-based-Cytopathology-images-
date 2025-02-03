import csv
import os

# Directory containing the source CSV files
source_dir = '/media/dataset/annot_cases_cpap/data/seperate_folder_csvs'

# Destination CSV file
destination_file = '/media/dataset/annot_cases_cpap/data/FINAL_TEST_DATA_new.csv'

# Names of the columns you want to extract and append
# columns_to_keep = ['SLIDE-ID', 'LABEL','SAMPLE-PATH','Path to Max Level Feature File']
columns_to_keep = ['Folder_Name','Label','WebP_Image_Path']

# List all CSV files in the source directory
source_files = [f for f in os.listdir(source_dir) if f.endswith('.csv')]

# print(source_files)
# Assume the destination file needs a header until we check
needs_header = True

# Loop through each source file to append its data
for source_file in source_files:
    # Construct the full path to the source file
    full_path = os.path.join(source_dir, source_file)
    
    # Open the current source CSV file
    with open(full_path, mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        
        # Open the destination CSV file in append mode
        with open(destination_file, mode='a', newline='', encoding='utf-8') as dest_file:
            # Initialize a DictWriter
            writer = csv.DictWriter(dest_file, fieldnames=columns_to_keep)
            
            # Write the header if needed
            if needs_header:
                # Check if the destination file is empty to decide on writing the header
                dest_file.seek(0, 2)  # Seek to the end of the file
                if dest_file.tell() == 0:  # Check if the file is empty
                    writer.writeheader()
                needs_header = False  # Update flag since header is written or not needed
            
            # Iterate through each row in the current source CSV
            for row in reader:
                # Select and write only the desired columns to the destination file
                writer.writerow({key: row[key] for key in columns_to_keep})



import os
import pandas as pd

# Paths to the two datasets
path_7days1 = r'C:\Users\Michael\Desktop\Y2 S2\DATA7903-Data Science Capstone Project 2B\dataset and pre-trained model\TrajAir A General Aviation Trajectory Dataset\Train and OOD\raw_data(7days1)'
path_7days2 = r'C:\Users\Michael\Desktop\Y2 S2\DATA7903-Data Science Capstone Project 2B\dataset and pre-trained model\TrajAir A General Aviation Trajectory Dataset\Train and OOD\raw_data(7days2)'

# Function to extract IDs from all CSVs in a folder
def extract_ids_from_folder(folder_path, id_column):
    all_ids = set()
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.csv'):
                csv_path = os.path.join(root, file)
                df = pd.read_csv(csv_path)
                if id_column in df.columns:
                    ids = set(df[id_column].unique())
                    all_ids.update(ids)
    return all_ids

# Replace 'ID_column' with the actual name of the ID column (e.g., 'aircraft_id', 'flight_id')
id_column = 'ID'

# Extract IDs from 7days1
ids_7days1 = extract_ids_from_folder(path_7days1, id_column)

# Extract IDs from 7days2
ids_7days2 = extract_ids_from_folder(path_7days2, id_column)

# Check for matching IDs
matching_ids = ids_7days1.intersection(ids_7days2)

# Output results
print(f"Number of matching IDs: {len(matching_ids)}")
print(f"Matching IDs: {matching_ids}")

# If you want, you can also check for unique IDs in each dataset
unique_to_7days1 = ids_7days1 - ids_7days2
unique_to_7days2 = ids_7days2 - ids_7days1

print(f"Unique IDs in 7days1: {len(unique_to_7days1)}")
print(f"Unique IDs in 7days2: {len(unique_to_7days2)}")

import pandas as pd

# Read the Excel file (or CSV file)
df = pd.read_excel('C:/Users/Michael/Desktop/Y2 S2/DATA7903-Data Science Capstone Project 2B/dataset and pre-trained model/pre-trained model/synthetic data/df_traj_tr_data_export_20241012_1047541.xlsm', engine='openpyxl')

# Get the unique Aircraft IDs
unique_ids = df['Aircraft ID'].unique()

# Set the output folder with a trailing slash
output_folder = "C:/Users/Michael/Desktop/Y2 S2/DATA7903-Data Science Capstone Project 2B/dataset and pre-trained model/pre-trained model/dataset/synthetic data/processed_data/train/"

# Function to format each row
def format_row(row):
    # First two columns (Frame and Aircraft ID) as integers
    first_two = [f'{int(row[0])}', f'{int(row[1])}']
    # Remaining columns as floats (with full precision)
    remaining_columns = [f'{x:.6f}' if isinstance(x, float) else str(x) for x in row[2:]]
    return ' '.join(first_two + remaining_columns)

# Iterate through the unique Aircraft IDs and collect the first continuous path
file_counter = 1
for aircraft_id in unique_ids:
    # Filter the rows that belong to this Aircraft ID
    aircraft_rows = df[df['Aircraft ID'] == aircraft_id].reset_index(drop=True)
    
    # Find the first discontinuity in the 'Frame' column
    discontinuity_index = (aircraft_rows['Frame'].diff().fillna(1) > 1).idxmax()
    
    # If there's a discontinuity, slice the dataframe up to that point
    if discontinuity_index != 0:
        aircraft_rows = aircraft_rows.iloc[:discontinuity_index]
    
    # Create a new file for each Aircraft ID
    file_name = output_folder + str(file_counter) + ".txt"
    
    with open(file_name, 'w') as f:
        # Write each row of the filtered data to the file with correct formatting
        for _, row in aircraft_rows.iterrows():
            f.write(format_row(row.values) + '\n')
    
    # Increment the file counter
    file_counter += 1

print("Export complete!")

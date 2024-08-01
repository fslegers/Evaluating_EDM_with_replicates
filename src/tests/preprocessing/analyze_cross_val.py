import os
import re
import pandas as pd

# Define the folder containing the CSV files
folder_path = "C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/cross_validation/tukey"

# List to store data from each file
data_list = []

# Define a regex pattern to extract the needed information
pattern = re.compile(r'rho=(\d+), length=(\d+), noise=([\d.]+)')

# Iterate over each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        # Get the full path of the file
        file_path = os.path.join(folder_path, filename)

        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Extract information from the filename using regex
        match = pattern.search(filename)
        if match:
            rho, length, noise = match.groups()
            df['rho'] = rho
            df['length'] = length
            df['noise'] = noise

        # Append the DataFrame to the list
        data_list.append(df)

# Concatenate all DataFrames in the list into a single DataFrame
final_df = pd.concat(data_list, ignore_index=True)

# Drop unnecessary columns
final_df.drop(columns=['lower', 'upper'], inplace=True)
final_df = final_df.sort_values(by=['group1', 'group2'])
final_df = final_df.style.format(precision=3)

# Define the folder to save the LaTeX file
path_name = "C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/cross_validation/cv_table.tex"

# Save DataFrame to LaTeX without row indices
final_df.hide().to_latex(path_name)

print(f"LaTeX table saved to {path_name}")

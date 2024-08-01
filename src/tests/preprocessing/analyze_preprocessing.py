import os
import re
import pandas as pd

# Define the folder containing the CSV files
folder_path = "C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/preprocessing/tukey"

# List to store data from each file
data_list = []

# Define a regex pattern to extract the needed information
pattern = re.compile(r'rho=(\d+), length=(\d+), n_repl=(\d+), noise=([\d.]+)')

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
            rho, length, n_repl, noise = match.groups()
            df['rho'] = rho
            df['length'] = length
            df['noise'] = noise
            df['n_repl'] = n_repl

        # Append the DataFrame to the list
        data_list.append(df)

# Concatenate all DataFrames in the list into a single DataFrame
final_df = pd.concat(data_list, ignore_index=True)

# Drop unnecessary columns
final_df.drop(columns=['lower', 'upper', 'group1', 'group2'], inplace=True)

# Rename columns
final_df.rename(columns={'n_repl': 'number of replicates'}, inplace=True)

# Define the folder to save the LaTeX file
output_folder_path = "C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/preprocessing"

# Save DataFrame to LaTeX without row indices
final_df.to_latex(f'{output_folder_path}/preprocessing_table.tex', index=False)

print(f"LaTeX table saved to {output_folder_path}/preprocessing_table.tex")

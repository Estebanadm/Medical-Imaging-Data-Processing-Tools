import pandas as pd
import glob
import os

def extract_csv_files(folder_path):
    """
    Extract all CSV files from the given folder.
    """
    # Use glob to find all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    return csv_files


def compare_csv(file1, file2):
    """
    Compare two CSV files and print differences, if any.
    """
    # Load the CSV files into dataframes
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Compare the two dataframes
    comparison = df1.compare(df2)

    # Check if there are any differences
    if comparison.empty:
        print(f"The two CSV files '{file1}' and '{file2}' are identical.")
    else:
        print(f"Differences between '{file1}' and '{file2}':")
        print(comparison)
        return comparison


def compare_csv_files_by_name(folder_path1, folder_path2):
    """
    Compare CSV files with the same names from two different folders.
    """
    # Extract CSV files from both folders
    csv_files_1 = extract_csv_files(folder_path1)
    csv_files_2 = extract_csv_files(folder_path2)

    # Create a dictionary to map file names to their paths for both folders
    file_map_1 = {os.path.basename(file): file for file in csv_files_1}
    file_map_2 = {os.path.basename(file): file for file in csv_files_2}

    print(f"Found {len(file_map_1)} CSV files in folder 1.")
    print(f"Found {len(file_map_2)} CSV files in folder 2.")

    # Find common file names in both folders
    common_files = set(file_map_1.keys()) & set(file_map_2.keys())

    if not common_files:
        print("No common CSV files to compare.")
        return

    # Compare each pair of files with the same name
    for file_name in common_files:
        file1 = file_map_1[file_name]
        file2 = file_map_2[file_name]
        print(f"\nComparing '{file_name}':")
        compare_csv(file1, file2)

def extract_xlsx_files(folder_path):
    """
    Extract all XLSX files from the given folder.
    """
    # Use glob to find all XLSX files in the folder
    xlsx_files = glob.glob(os.path.join(folder_path, "*.xlsx"))
    return xlsx_files

def compare_xlsx(file1, file2):
    """
    Compare two XLSX files and print differences, if any.
    """
    # Load the XLSX files into dataframes
    df1 = pd.read_excel(file1)
    df2 = pd.read_excel(file2)

    # Compare the two dataframes
    comparison = df1.compare(df2)

    # Check if there are any differences
    if comparison.empty:
        print(f"The two XLSX files '{file1}' and '{file2}' are identical.")
    else:
        print(f"Differences between '{file1}' and '{file2}':")
        print(comparison)
        return comparison

def compare_xlsx_files_by_name(folder_path1, folder_path2):
    """
    Compare XLSX files with the same names from two different folders.
    """
    # Extract XLSX files from both folders
    xlsx_files_1 = extract_xlsx_files(folder_path1)
    xlsx_files_2 = extract_xlsx_files(folder_path2)

    # Create a dictionary to map file names to their paths for both folders
    file_map_1 = {os.path.basename(file): file for file in xlsx_files_1}
    file_map_2 = {os.path.basename(file): file for file in xlsx_files_2}

    print(f"Found {len(file_map_1)} XLSX files in folder 1.")
    print(f"Found {len(file_map_2)} XLSX files in folder 2.")

    # Find common file names in both folders
    common_files = set(file_map_1.keys()) & set(file_map_2.keys())

    if not common_files:
        print("No common XLSX files to compare.")
        return

    # Compare each pair of files with the same name
    for file_name in common_files:
        file1 = file_map_1[file_name]
        file2 = file_map_2[file_name]
        print(f"\nComparing '{file_name}':")
        compare_xlsx(file1, file2)



# Example usage:
folder_path1 = "../Excel TSV"  # Path to the first folder
folder_path2 = "../New Excel TSV"  # Path to the second folder

xlsx_1_path='../Excel Files/Recording 4'
xlsx_2_path='../New Excel Files/Recording 4'



# compare_csv_files_by_name(folder_path1, folder_path2)
compare_xlsx_files_by_name(xlsx_1_path,xlsx_2_path)  
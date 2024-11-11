import pandas as pd

def compare_image_numbers(first_excel_path, second_excel_path):
    # Load the two Excel sheets into dataframes
    df1 = pd.read_excel(first_excel_path)
    df2 = pd.read_excel(second_excel_path)

    mismatched_rows = []  # Store row numbers where image numbers do not match

    # Iterate over the rows in the first dataframe
    for index, row in df1.iterrows():
        dicom_path_first_sheet = row['DICOM Path']  # Extract the DICOM Path from the first sheet
        image_number_first_sheet = row['Image Number']  # Extract Image Number from the first sheet

        # Search for this exact DICOM Path in the second Excel sheet
        match = df2[df2['DICOM Path'] == dicom_path_first_sheet]

        if not match.empty:
            # Get the corresponding image number from the second sheet
            image_number_second_sheet = match.iloc[0]['Image Number']

            # Compare the image numbers from both sheets
            if image_number_first_sheet != image_number_second_sheet:
                print(f"Row {index + 1}: Image Numbers DO NOT Match ({image_number_first_sheet} != {image_number_second_sheet})")
                mismatched_rows.append(index + 1)  # Store row number (1-based index)
        else:
            print(f"Row {index + 1}: No matching DICOM Path found in the second sheet.")

    # Display all mismatched rows at the end
    if mismatched_rows:
        print("\nMismatched Rows: ", mismatched_rows)
        import pdb; pdb.set_trace()
    else:
        print("\nAll Image Numbers match.")

def compare_series_numbers(first_excel_path):
    # Load the Excel sheet into a dataframe
    df = pd.read_excel(first_excel_path)

    mismatched_rows = []  # Store row numbers where series numbers do not match

    # Iterate over the rows in the dataframe
    for index, row in df.iterrows():
        series_number_dicom = row['Series Number DICOM']  # Series Number from DICOM Path row
        series_number_image = row['Series Number Image']  # Series Number from Image row

        # Compare the series numbers
        if series_number_dicom != series_number_image:
            print(f"Row {index + 1}: Series Numbers DO NOT Match ({series_number_dicom} != {series_number_image})")
            mismatched_rows.append(index + 1)  # Store row number (1-based index)

    # Display all mismatched rows at the end
    if mismatched_rows:
        print("\nMismatched Rows (Series Numbers): ", mismatched_rows)
        import pdb; pdb.set_trace()
    else:
        print("\nAll Series Numbers match.")


def detect_empty_zoom_window_level(first_excel_path):
    # Load the Excel sheet into a dataframe
    df = pd.read_excel(first_excel_path)

    empty_rows = []  # Store row numbers where Zoom, Window, or Level are empty

    # Iterate over the rows in the dataframe
    for index, row in df.iterrows():
        zoom = row['Zoom']  # Value from Zoom column
        window = row['Window']  # Value from Window column
        level = row['Level']  # Value from Level column

        # Check if any of the fields are empty (NaN)
        if pd.isna(zoom) or pd.isna(window) or pd.isna(level):
            print(f"Row {index + 1}: Empty value in Zoom, Window, or Level")
            empty_rows.append(index + 1)  # Store row number (1-based index)

    # Display all rows with empty values at the end
    if empty_rows:
        print("\nRows with empty Zoom, Window, or Level: ", empty_rows)
        import pdb; pdb.set_trace()
    else:
        print("\nNo empty values found in Zoom, Window, or Level.")


start=1
end=24
for current_folder in range(start, end+1):

    # Example usage
    first_excel = '../Final MD-1/Recording '+ str(current_folder) +'/Updated_matched_data_R'+ str(current_folder) +'.xlsx' # The first Excel file with DICOM Path and Image Number
    second_excel = '../Final MD-1/Recording '+ str(current_folder) +'/dicom_data_R'+ str(current_folder) +'.xlsx'  # The second Excel file with DICOM Path and Image Number
    print("Validating "+ str(current_folder))
    if current_folder != 2:
        detect_empty_zoom_window_level(first_excel)
        compare_image_numbers(first_excel, second_excel)
        compare_series_numbers(first_excel)

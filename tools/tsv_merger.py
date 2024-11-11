import pandas as pd
import re

def extract_frame_id(image_path):
    """Extract the frame id from the image path (e.g., 'frame006631.jpg' -> '006631')."""
    match = re.search(r'frame(\d+)\.jpg', image_path)
    if match:
        return match.group(1)  # Return the extracted frame ID as a string
    return None

def compare_and_copy_data(excel_path, csv_path, output_path):
    # Load the Excel file
    excel_df = pd.read_excel(excel_path)

    # Load the CSV file
    csv_df = pd.read_csv(csv_path)

    # Loop through each row in the Excel file
    for index, row in excel_df.iterrows():
        # Extract frame_id from the image path column
        image_path = row['Image Path']
        frame_id = extract_frame_id(image_path)

        # If frame_id is valid, compare it to the Captured_time(frame_id) in the CSV
        if frame_id:
            matching_row = csv_df[csv_df['Captured_time(frame_id)'] == int(frame_id)]

            # If a match is found, copy values from the CSV row to the Excel row
            if not matching_row.empty:
                csv_data = matching_row.iloc[0].to_dict()  # Get the first matching row as a dictionary
                for col, value in csv_data.items():
                    excel_df.loc[index, f'{col}'] = value  # Create new columns in Excel file for CSV data

    # Save the new Excel file with copied data
    excel_df.to_excel(output_path, index=False)
    print(f"Updated Excel file saved at {output_path}")

start=3
end= 24
for current_data in range(start, end+1):

    mainPath=f'../Final Excel MD-1/Recording {current_data}'
    excel_file = mainPath+f'/Updated_matched_data_R{current_data}.xlsx'
    csv_file = f'../MD-1 TSV/Final_R{current_data}.csv'
    output_file = mainPath+f'/Final_R{current_data}.xlsx'

    compare_and_copy_data(excel_file, csv_file, output_file)
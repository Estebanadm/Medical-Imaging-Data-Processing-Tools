import os
import pydicom
import cv2
import easyocr
import pandas as pd
import re
import numpy as np
from openpyxl import load_workbook
import matplotlib.pyplot as plt


# Initialize EasyOCR reader (for English language)
reader = easyocr.Reader(['en'], gpu=False)

def extract_series_and_slice_location(dicom_file):
    try:
        dicom_data = pydicom.dcmread(dicom_file)
        
        # Extract Series Number
        series_number = str(dicom_data.get("SeriesNumber", "Unknown"))
        
        # Extract Slice Location
        slice_location = dicom_data.get("SliceLocation", "Unknown")
        
        if slice_location != "Unknown":
            # Ensure slice_location is a float and format to 2 decimal places
            try:
                slice_location = float(slice_location)
                # Format the slice location to 2 decimal points
                slice_location = "{:.2f}".format(slice_location)
            except ValueError:
                print(f"Slice location could not be converted to float for {dicom_file}")
                slice_location = "Unknown"
                import pdb; pdb.set_trace()
        
        # Debugging: print the series number and slice location for verification
        print(f"DICOM {series_number}, {slice_location}")

        return series_number, slice_location

    except Exception as e:
        print(f"Error reading {dicom_file}: {e}")
        import pdb; pdb.set_trace()
        return None, None

def enhance_image_for_ocr(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding (Binarization)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Apply gentle blurring to reduce noise
    blurred = cv2.GaussianBlur(binary, (3, 3), 0)
    
    # Sharpen the image slightly without overdoing it
    sharpening_kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
    sharpened_image = cv2.filter2D(blurred, -1, sharpening_kernel)
    
    return sharpened_image


def extract_text_from_image(image_path):
    try:
        enhanced_image = enhance_image_for_ocr(image_path)
        result = reader.readtext(enhanced_image, detail=0)
        return result
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        import pdb; pdb.set_trace() 
        return None

def match_series_and_location(ocr_series, ocr_location, dicom_series, dicom_location):
    try:
        return str(ocr_series) == str(dicom_series) and float(ocr_location) == float(dicom_location)
    except Exception as e:
        print(f"Error during comparison: {e}")
        import pdb; pdb.set_trace()
        return False
import re

def extract_series_and_slice_from_ocr(ocr_text):
    series_number, slice_location, image_number = None, None, None
    combined_text = ' '.join(ocr_text)

    # Debugging: Print the combined OCR text
    print(f"Combined OCR Text: {combined_text}")

    # Search for series number (e.g., "Ser 12")
    series_match = re.search(r'Ser[:\s\.-]*\s*(\d+)', combined_text, re.IGNORECASE)
    if series_match:
        series_number = int(series_match.group(1))

    # Search for image number (e.g., "Img 34")
    img_match = re.search(r'Img[:\s\.-]*\s*(\d+)', combined_text, re.IGNORECASE)
    if img_match:
        image_number = int(img_match.group(1))

    # Initial regex to capture slice location
    loc_match = re.search(r'Loc[:\s]*(-?\d+)[,\s]*(\d+)', combined_text, re.IGNORECASE)

    # 1. If a match is found, attempt to clean up and format the result
    if loc_match:
        slice_location_str = loc_match.group(1) + '.' + loc_match.group(2)
        # Remove any trailing text like 'mm' or non-numeric characters
        slice_location_str = re.sub(r'[^\d\.-]', '', slice_location_str)
        try:
            slice_location = float(slice_location_str)
        except ValueError:
            slice_location = None
        print(f"First attempt: {slice_location}")
    
    # 2. Fallback: Handle if the first attempt failed (e.g., extra spaces or misread characters)
    if slice_location is None:
        # Try matching for cases where the decimal is replaced by a space (e.g., "565 71" instead of "565.71")
        loc_match = re.search(r'Loc[:\s]*(-?\d+)\s+(\d+)', combined_text, re.IGNORECASE)
        if loc_match:
            slice_location_str = loc_match.group(1) + '.' + loc_match.group(2)
            try:
                slice_location = float(slice_location_str)
            except ValueError:
                slice_location = None
        print(f"Second attempt (space between numbers): {slice_location}")

    # 3. Fallback: Handle misread characters in the middle of the slice location (like "O" instead of "0")
    if slice_location is None:
        # Replace common OCR mistakes (e.g., "O" -> "0", "I" -> "1")
        cleaned_text = combined_text.replace('O', '0').replace('I', '1')
        loc_match = re.search(r'Loc[:\s]*(-?\d+)[,\s]*(\d+)', cleaned_text, re.IGNORECASE)
        if loc_match:
            slice_location_str = loc_match.group(1) + '.' + loc_match.group(2)
            try:
                slice_location = float(slice_location_str)
            except ValueError:
                slice_location = None
        print(f"Third attempt (correcting misread characters): {slice_location}")

    # 4. Fallback: Handle cases where "mm" or other units might be mixed into the slice location
    if slice_location is None:
        # Look for "Loc: 565 71mm" and similar cases, and remove "mm" or other trailing text
        loc_match = re.search(r'Loc[:\s]*(-?\d+)\s*(\d+)', combined_text, re.IGNORECASE)
        if loc_match:
            slice_location_str = loc_match.group(1) + '.' + loc_match.group(2)
            # Remove units like "mm" and keep only numeric values
            slice_location_str = re.sub(r'[^\d\.-]', '', slice_location_str)
            try:
                slice_location = float(slice_location_str)
            except ValueError:
                slice_location = None
        print(f"Fourth attempt (removing units): {slice_location}")

    # Return the extracted values (series_number, slice_location, image_number)
    return series_number, slice_location, image_number


def truncate_float(value, decimals):
    str_value = str(value)
    if '.' in str_value:
        integer_part, decimal_part = str_value.split('.')
        truncated_decimal_part = decimal_part[:decimals]
        return float(f"{integer_part}.{truncated_decimal_part}")
    else:
        return str(value)


def process_image_and_dicom_folders(image_folder_path, dicom_folder_path, output_excel):
    # Define the columns for the Excel sheet
    columns = ["Image Path", "DICOM Path", "Image Number", "Series Number", "Slice Location"]
    
    # Create the Excel file if it doesn't exist
    if not os.path.exists(output_excel):
        print(f"Creating {output_excel} with headers...")
        df = pd.DataFrame(columns=columns)
        df.to_excel(output_excel, index=False)

    # Collect all image file paths into a list
    image_files = []
    for root, dirs, files in os.walk(image_folder_path):
        for file in files:
            if file.endswith(".jpg"):
                image_path = os.path.join(root, file)
                image_files.append(image_path)

    # Sort the image files and start from index 5379
    image_files.sort()
    start_index = 5864
    image_files = image_files[start_index:]

    # Process each image file
    for image_path in image_files:
        print(f"\nProcessing Image: {image_path}")
        image = cv2.imread(image_path)

        if image is None:
            print(f"Failed to load image from {image_path}")
            continue

        # Extract text using OCR
        ocr_text = extract_text_from_image(image)
        ocr_series, ocr_location, image_number = None, None, None

        if ocr_text:
            ocr_series, ocr_location, image_number = extract_series_and_slice_from_ocr(ocr_text)
            # print(f"OCR Extracted -> Series: {ocr_series}, Location: {ocr_location}, Image Number: {image_number}")
        else:
            print(f"No text extracted from {image_path}")
            continue

        # Loop through all DICOM files to find a match
        for root_dicom, _, dicom_files in os.walk(dicom_folder_path):
            for dicom_file in dicom_files:
                if dicom_file.endswith(".dcm"):
                    found=False
                    dicom_file_path = os.path.join(root_dicom, dicom_file)
                    print("Image",ocr_series, ocr_location)
                    series_number, slice_location = extract_series_and_slice_location(dicom_file_path)

                    if series_number is None or slice_location is None:
                        continue

                    # Truncate the DICOM slice location to 2 decimal places without rounding
                    slice_location = truncate_float(slice_location, 2)

                    # Match the OCR extracted data with DICOM metadata
                    # print(f"Comparing OCR Series: {ocr_series} with DICOM Series: {series_number}")
                    # print(f"Comparing OCR Location: {ocr_location} with DICOM Location: {slice_location}")
                    
                    if match_series_and_location(ocr_series, ocr_location, series_number, slice_location):
                        print(f"Match found! Writing data to Excel.")
                        df = pd.DataFrame([[image_path, dicom_file_path, image_number, ocr_series, ocr_location]], columns=columns)
                        append_df_to_excel(output_excel, df, header=False, index=False)
                        found=True
                        # import pdb; pdb.set_trace()  
                        break 
            if not found:

                print(f"No match found for {image_path}")
                import pdb; pdb.set_trace()     

    print(f"\nProcessing complete. Data saved to {output_excel}")

# Set paths and process the folders
image_folder_path = "../recording/recording 2/crop__Recording-2"
dicom_folder_path = "../recording/recording 2/184554064 (trauma from auto accident)"
output_excel = "../Excel files/Recording2.1.xlsx"

process_image_and_dicom_folders(image_folder_path, dicom_folder_path, output_excel)
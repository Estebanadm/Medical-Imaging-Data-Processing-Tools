import os
import pydicom
import cv2
import easyocr
import pandas as pd
import re
import numpy as np
from openpyxl import load_workbook
import matplotlib.pyplot as plt
import tensorflow as tf

reader = easyocr.Reader(['en'], gpu=True)
doubleCheck=[]


def debug(image_path):
    print(f"Debugging OCR and extraction for image: {image_path}")

    # Extract OCR text
    ocr_text = extract_ocr_data(image_path)
    
    if ocr_text:
        zoom, window, level = extract_zoom_window_level_from_ocr(ocr_text, image_path)
        print(f"Extracted from OCR -> Zoom: {zoom}, Window: {window}, Level: {level}")
    else:
        print("No OCR text found.")



def extract_window_level_from_dicom(dicom_file):
    try:
        # Read the DICOM file
        dicom_data = pydicom.dcmread(dicom_file)

        # Extract Window Width and Window Level
        window_width = dicom_data.get("WindowWidth", "Not Available")
        window_level = dicom_data.get("WindowCenter", "Not Available")

        print(f"Window Width: {window_width}")
        print(f"Window Level: {window_level}")

        return window_width, window_level
    except Exception as e:
        print(f"Error reading DICOM file {dicom_file}: {e}")
        return None, None


def enhance_image_for_ocr(image_path, crop_coords=None):
    # Load the original image
    original_image = cv2.imread(image_path)
    
    if original_image is None:
        print(f"Error: Could not load image from {image_path}")
        return None

    # Crop the image if crop coordinates are provided
    if crop_coords is not None:
        x_start, y_start, x_end, y_end = crop_coords
        cropped_image = original_image[y_start:y_end, x_start:x_end]
    else:
        cropped_image = original_image

    # Initialize OpenCV's DNN Super-Resolution model
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel('FSRCNN_x4.pb') 
    sr.setModel("fsrcnn", 4)

    # Apply super-resolution to the cropped image
    upscaled_image = sr.upsample(cropped_image)

    # Convert the upscaled image to grayscale
    grayscale_image = cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2GRAY)

    return grayscale_image

def check_negative_sign(image_path, crop_coords):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return False

    y_start, y_end = crop_coords[1], crop_coords[3]

    # Flag for character detection
    in_character = False
    first_white_pixel_x = None
    character_start_x = None
    character_end_x = None

    # Iterate horizontally to detect individual characters
    for x_start in range(crop_coords[0], crop_coords[2]):
        # Define a thin vertical slice (column) to check at the current x position
        roi_column = image[y_start:y_end, x_start:x_start + 1]

        # Apply thresholding to isolate white pixels
        ret, thresh = cv2.threshold(roi_column, 150, 255, cv2.THRESH_BINARY)

        # Check for any white pixels in this column
        white_pixels = np.where(thresh == 255)

        if len(white_pixels[0]) > 0:  # White pixels detected, we're in a character
            if not in_character:
                # Start of a new character
                character_start_x = x_start
                in_character = True
        else:
            if in_character:
                # End of the character detected (when there's a column with no white pixels)
                character_end_x = x_start
                in_character = False  # Character ended

                # Visualize the character
                roi_character = image[y_start:y_end, character_start_x:character_end_x]
                thresh_character = cv2.threshold(roi_character, 150, 255, cv2.THRESH_BINARY)[1]

                white_pixel_count = np.sum(thresh_character == 255)

                # print(f"White pixel count in the cropped area: {white_pixel_count}")
                if white_pixel_count <=4 and white_pixel_count > 2:
                    print("Negative sign detected!")
                    return True
        
                
            

    print("No negative sign detected.")

    return False  # Function returns after analyzing all characters


def extract_zoom_window_level_from_ocr(ocr_text, image_path):
    zoom_value = None
    window_value = None
    level_value = None

    # Combine OCR text into one string for easier pattern matching
    combined_text = ' '.join(ocr_text)
    print(f"Combined OCR Text: {combined_text}")

    # Extract Zoom value (e.g., "Zoom 153%" or "Zoom: 153")
    zoom_match = re.search(r'Zoom\s*[_\s\-:]*\s*(\d+)\s*%?', combined_text, re.IGNORECASE)
    if zoom_match:
        zoom_value = int(zoom_match.group(1))
        print(f"Zoom Value: {zoom_value}")
    else:
        print("Zoom Value not found")

    # Extract Window value (e.g., "W392")
    window_match = re.search(r'W\s*[_\-:]*\s*(\d+)', combined_text, re.IGNORECASE)
    if window_match:
        window_value = int(window_match.group(1))
        print(f"Window (W) Value: {window_value}")
    else:
        print("Window (W) Value not found")

    # Extract Level value (allowing for underscores or hyphens like "L_321" or "L-321")
    level_match = re.search(r'L\s*[:_\-\.\s]*\s*([-]?\d+\.?\d*)', combined_text, re.IGNORECASE)
    if level_match:
        level_value = int(level_match.group(1))

        # Use the check_negative_sign function to confirm negative sign in the image
        crop_coords = (1765, 980, 1920, 998)  # Define region of interest for negative sign
        if check_negative_sign(image_path, crop_coords):
            level_value = -abs(level_value)
            print(f"Corrected Level (L) Value: {level_value} (Negative sign confirmed)")
        else:
            # Check second location
            #check 3rd location
            print(f"Level (L) Value: {level_value}")
    else:
        print("Level (L) Value not found")

    if zoom_value is None or window_value is None or level_value is None:
        doubleCheck.append( os.path.basename(image_path))

    if zoom_value == 1539:
        zoom_value = 153

    return zoom_value, window_value, level_value



def extract_series_number_from_dicom(dicom_file):
    try:
        dicom_data = pydicom.dcmread(dicom_file)
        series_number = dicom_data.get("SeriesNumber", "Not Available")
        return series_number
    except Exception as e:
        print(f"Error reading DICOM file {dicom_file}: {e}")
        return None


def extract_series_number_from_image(image_path, excel_file_path):
    try:
        # Load the Excel file into a DataFrame
        df = pd.read_excel(excel_file_path)

        # Search for the row where the 'Image Path' column matches the provided image_path
        matching_row = df[df['Image Path'].str.contains(os.path.basename(image_path), case=False, na=False)]

        # If a matching row is found, extract the 'Series Number' from the second column
        if not matching_row.empty:
            series_number = matching_row['Series Number'].values[0]
            print(f"Series Number from Image: {series_number}")
            return series_number
        else:
            print(f"No matching series number found for image: {image_path}")
            return None
    except Exception as e:
        print(f"Error reading Excel file or extracting data for {image_path}: {e}")
        return None
    


def extract_ocr_data(image_path):
    try:
        crop_coords = (1578, 950, 2000, 1000)
        enhanced_image = enhance_image_for_ocr(image_path, crop_coords)

        ocr_result = reader.readtext(enhanced_image, detail=0)
        print(f"Image Path: {image_path}")
        print(f"OCR Result: {ocr_result}")

        return ocr_result 
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def process_and_save_to_new_excel(image_folder_path, excel_file_path, new_output_directory, file_name,image_output_excel):
    if not os.path.exists(new_output_directory):
        os.makedirs(new_output_directory)

    new_excel_file = os.path.join(new_output_directory, file_name)
    df = pd.read_excel(excel_file_path)

    for root, _, files in os.walk(image_folder_path):
        for file in sorted(files):
            if file.endswith(".jpg"):
                image_path = os.path.join(root, file)

                ocr_text = extract_ocr_data(image_path)
                if ocr_text:
                    zoom_value, window_value, level_value = extract_zoom_window_level_from_ocr(ocr_text, image_path)

                for index, row in df.iterrows():
                    excel_image_path = row['Image Path']
                    excel_dicom_path = row['DICOM Path']
                    
                    if os.path.basename(excel_image_path) == os.path.basename(image_path):
                        # Extract series number from DICOM file
                        series_number_dicom = extract_series_number_from_dicom(excel_dicom_path)
                        
                        # Extract series number from Image file (if available)
                        series_number_image = extract_series_number_from_image(image_path,image_output_excel)

                        # Update the DataFrame with the extracted values
                        df.at[index, 'Series Number DICOM'] = series_number_dicom  # Third column
                        df.at[index, 'Series Number Image'] = series_number_image  # Fourth column
                        df.at[index, 'Zoom'] = zoom_value  # Fifth column
                        df.at[index, 'Window'] = window_value  # Sixth column
                        df.at[index, 'Level'] = level_value  # Seventh column

                        print(f"Updated row {index} for image {excel_image_path} with DICOM Series: {series_number_dicom}, "
                              f"Image Series: {series_number_image}, Zoom: {zoom_value}, Window: {window_value}, Level: {level_value}")

    df.to_excel(new_excel_file, index=False)
    print(f"Updated Excel file saved at {new_excel_file}")
    print(f"Images with missing OCR data:")
    print(doubleCheck)


image_folder_path = "../Recordings/recording 7/pnuemothorax pain(7) agrawal"
output_directory = "../New Excel Files/Recording 17"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

curr_excel = "R7"
dicom_output_excel = os.path.join(output_directory, "dicom_data_"+curr_excel+".xlsx")
image_output_excel = os.path.join(output_directory, "image_data_"+curr_excel+".xlsx")
final_output_excel = os.path.join(output_directory, "matched_data_"+curr_excel+".xlsx")

new_output_directory = os.path.join(output_directory)
file_name = "Updated_matched_data_"+curr_excel+".xlsx"
# process_and_save_to_new_excel(image_folder_path, final_output_excel, new_output_directory,file_name,image_output_excel)
debug("../Recordings/recording 15/chest 41 (15) agrawal/frame000146.jpg")







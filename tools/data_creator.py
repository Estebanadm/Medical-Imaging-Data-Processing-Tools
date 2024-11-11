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

def debug_first_run(image_path):
    print("Debugging OCR first run")
    crop_coords = (315, 900, 560, 1021)
    ocr_text = extract_ocr_data(image_path,crop_coords)
    
    if ocr_text:
        series_number, slice_location, image_number = extract_series_and_slice_from_ocr(ocr_text,image_path)          
        if slice_location is None or slice_location == "":
            print("image_path", image_path)
            print("OCR Text", ocr_text)
            print("Slice location", slice_location)
            slice_location = "Missing" 
        print(f"Series Number: {series_number}, Slice Location: {slice_location}, Image Number: {image_number}")

def debug_second_run(image_path):
    print("Debugging OCR Second run")

    crop_coords = (1578, 950, 2000, 1000)
    ocr_text = extract_ocr_data(image_path, crop_coords)
    
    if ocr_text:
        zoom, window, level = extract_zoom_window_level_from_ocr(ocr_text, image_path)
        print(f"Extracted from OCR -> Zoom: {zoom}, Window: {window}, Level: {level}")
    else:
        print("No OCR text found.")

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

                # Visualize the detected character
                roi_character = image[y_start:y_end, character_start_x:character_end_x]
                thresh_character = cv2.threshold(roi_character, 150, 255, cv2.THRESH_BINARY)[1]

                white_pixel_count = np.sum(thresh_character == 255)

                # # Show detected character and its thresholded version
                # plt.figure(figsize=(8, 4))
                
                # plt.subplot(1, 2, 1)
                # plt.imshow(roi_character, cmap='gray')
                # plt.title("Detected Character (Grayscale)")

                # plt.subplot(1, 2, 2)
                # plt.imshow(thresh_character, cmap='gray')
                # plt.title(f"Character Thresholded\nWhite pixel count: {white_pixel_count}")
                # plt.show()

                if 2 < white_pixel_count <= 4:  # Likely a negative sign
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
        crop_coords = (1765, 982, 1920, 992)  # Define region of interest for negative sign
        if check_negative_sign(image_path, crop_coords):
            level_value = -abs(level_value)
            print(f"Corrected Level (L) Value: {level_value} (Negative sign confirmed)")
        else:
            # Check second location
            #check 3rd location
            print(f"Level (L) Value: {level_value}")
    else:
        print("Level (L) Value not found")
    
    if zoom_value == 1539:
        zoom_value = 153
    if zoom_value == 1569:
        zoom_value = 156 

    if zoom_value is None or window_value is None or level_value is None:
        doubleCheck.append( os.path.basename(image_path))

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
    


def extract_ocr_data(image_path,crop_coords = None):
    try:
        
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
    print(new_excel_file)
    df = pd.read_excel(excel_file_path)

    for root, _, files in os.walk(image_folder_path):
        for file in sorted(files):
            if file.endswith(".jpg"):
                image_path = os.path.join(root, file)
                crop_coords = (1578, 950, 2000, 1000)
                ocr_text = extract_ocr_data(image_path, crop_coords)
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


def extract_dicom_metadata(dicom_file):
    try:
        dicom_data = pydicom.dcmread(dicom_file)
        series_number = str(dicom_data.get("SeriesNumber", "Unknown"))
        slice_location = dicom_data.get("SliceLocation", "Unknown")
        image_number = dicom_data.get("InstanceNumber", "Unknown")  # Extract the image number (InstanceNumber)
        print(f"Dicom Series Number: {series_number}, Slice Location: {slice_location}, Image Number: {image_number}")
        if slice_location != "Unknown":
            try:
                slice_location = float(slice_location)
                slice_location = "{:.2f}".format(slice_location)
            except ValueError:
                slice_location = "Unknown"

        return dicom_file, series_number, slice_location, image_number
    except Exception as e:
        print(f"Error reading {dicom_file}: {e}")
        return dicom_file, "Unknown", "Unknown", "Unknown"

def save_dicom_to_excel(dicom_folder_path, output_excel):
    dicom_data = []
    for root, _, files in os.walk(dicom_folder_path):
        for file in files:
            if file.endswith(".dcm"):
                dicom_file_path = os.path.join(root, file)
                dicom_data.append(extract_dicom_metadata(dicom_file_path))
    
    df = pd.DataFrame(dicom_data, columns=["DICOM Path", "Series Number", "Slice Location", "Image Number"])
    df.to_excel(output_excel, index=False)
    print(f"DICOM metadata saved to {output_excel}")

    
def extract_series_and_slice_from_ocr(ocr_text, image_path):
    series_number = None
    slice_location = None
    image_number = None 

    # Combine the OCR text into one string
    combined_text = ' '.join(ocr_text)
    print(f"Combined OCR Text: {combined_text}")
    
    series_match = re.search(r'Ser[:\s\.-]*\s*(\d+)', combined_text, re.IGNORECASE)
    if series_match:
        series_number = int(series_match.group(1))
    else:
        print("Series Number not found")

    # Extract Image Number (Img)
    img_match = re.search(r'Img[:\s\.-]*\s*(\d+)', combined_text, re.IGNORECASE)
    if img_match:
        image_number = int(img_match.group(1))
        # print(f"Found Image Number: {image_number}")
    else:
        print("Image Number not found")

    # Check for different "Loc" patterns in a case-insensitive way
    loc_index = combined_text.lower().find('loc')
    if loc_index != -1:
        # Extract substring starting from the 'Loc' keyword
        loc_substring = combined_text[loc_index:]
        
        # Update the regex to match the number right after 'Loc'
        location_match = re.search(r'loc\s*[:\-]?\s*(-?\d+\.?\d*)', loc_substring, re.IGNORECASE)
        if location_match:
            slice_location_str = location_match.group(1).replace(" ", "")
            slice_location = float(slice_location_str)
            print("slice_location_str", slice_location_str)

        else:
            print("No valid slice location found after 'Loc'")
    else:
        print("No 'Loc' keyword found in text")
    
    combined_text = ' '.join(ocr_text)

    if series_number is None:
        last_item_match = re.search(r'(\d+)$', combined_text)
        if last_item_match:
            series_number = int(last_item_match.group(1))
            print(f"Found standalone Series Number: {series_number}")


    crop_coords = (320,1001,535,1011)
    if slice_location is not None and check_negative_sign(image_path, crop_coords) and slice_location >= 0:
        slice_location = -slice_location
        print(f"Corrected Slice Location: {slice_location}")

    print(f"Series Number: {series_number}, Slice Location: {slice_location}, Image Number: {image_number} ")
    print("--------------------------------------------------")

    if series_number is None or slice_location is None:
        print("Error locating OCR text")

    return series_number, slice_location, image_number 

def save_images_to_excel(image_folder_path, output_excel):
    
    image_data = []
    for root, _, files in sorted(os.walk(image_folder_path)):
        for file in sorted(files): 
            if file.endswith(".jpg"):
                image_path = os.path.join(root, file)
                # We need to crop to desired square here
                # crop image to desired location
                crop_coords = (315, 900, 560, 1021)
                ocr_text = extract_ocr_data(image_path, crop_coords)
                
                if ocr_text:
                    series_number, slice_location, image_number = extract_series_and_slice_from_ocr(ocr_text,image_path)
                      
                    image_data.append([image_path, series_number, slice_location, image_number])
    
    df = pd.DataFrame(image_data, columns=["Image Path", "Series Number", "Slice Location", "Image Number"])
    df.to_excel(output_excel, index=False)
    print(f"Image OCR data saved to {output_excel}")
    print("Double Check",len(doubleCheck))
    print(doubleCheck)
    
def match_dicom_and_images(dicom_excel, image_excel, output_excel):
    dicom_df = pd.read_excel(dicom_excel)
    image_df = pd.read_excel(image_excel)

    match_data = []

    for _, image_row in image_df.iterrows():
        image_series = image_row["Series Number"]
        image_location = image_row["Slice Location"]
        image_path = image_row["Image Path"]
        image_number = image_row["Image Number"]

        dicom_path = None

        # Loop through DICOM files to find a match
        for _, dicom_row in dicom_df.iterrows():
            dicom_series = dicom_row["Series Number"]
            dicom_location = dicom_row["Slice Location"]
            dicom_image_number = dicom_row["Image Number"]
            dicom_path_candidate = dicom_row["DICOM Path"]

            # Check if both dicom_location and image_location are numeric
            try:
                # Debugging: Print the values before comparison
                # print(f"Comparing Image Series: {image_series} with DICOM Series: {dicom_series}")
                # print(f"Comparing Image Location: {image_location} with DICOM Location: {dicom_location}")
                # print(f"Image Number: {image_number}, DICOM Image Number: {dicom_image_number}")
                
                if dicom_location != "Unknown" and image_location is not None and image_location != "Missing":
                    dicom_location_float = float(dicom_location)
                    image_location_float = float(image_location)

                    # Match by series number and slice location
                    if dicom_series == image_series and dicom_location_float == image_location_float:
                        dicom_path = dicom_path_candidate
                        print(f"Match found by slice location: {dicom_path}")
                        break  # Stop once a match is found
                
            except ValueError:
                # Handle case where location cannot be converted to float
                print(f"Skipping location comparison for non-numeric values. DICOM: {dicom_location}, Image: {image_location}")

            # If location comparison is skipped or invalid, match by image number
            if dicom_path is None and dicom_series == image_series and dicom_image_number == image_number : 
                dicom_path = dicom_path_candidate
                print("Image path", image_path)
                print(f"Match found by image number: {dicom_path}")
                break

        # Append the match data
        if dicom_path is None:
            print(f"No match found for image: {image_path}")
            doubleCheck.append([os.path.basename(image_path)])

        match_data.append([image_path, dicom_path if dicom_path else '', image_number])

    # Create a DataFrame and save the results to an Excel file
    df = pd.DataFrame(match_data, columns=["Image Path", "DICOM Path", "Image Number"])
    df.to_excel(output_excel, index=False)
    print(f"Matched data saved to {output_excel}")

def find_dicom_folder(root_directory):
    # Walk through the root directory to look for DICOM files
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            file_path = os.path.join(root, file)

            # Check if the file ends with .dcm extension or is a valid DICOM file
            if file.lower().endswith(".dcm"):
                return root  # Return the folder path if a DICOM file is found
            else:
                try:
                    # Try reading the file to check if it's a valid DICOM file
                    pydicom.dcmread(file_path, stop_before_pixels=True)
                    return root  # Return the folder path if it's a valid DICOM file
                except Exception:
                    # If not a valid DICOM file, continue
                    continue

    # If no DICOM files are found, return None
    return None

def find_jpg_folders(root_directory):
    jpg_folders = []

    # Walk through the root directory to look for JPG files
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            if file.lower().endswith('.jpg'):  # Check if the file has a .jpg extension
                if root not in jpg_folders:  # Ensure each folder is added only once
                    jpg_folders.append(root)
                break  # Exit the inner loop once a JPG file is found in the folder

    return jpg_folders

def find_crop_and_other_folder(folders):
    crop_folder = None
    other_folder = None

    for folder in folders:
        if 'crop' in folder.lower():
            crop_folder = folder
        else:
            other_folder = folder

    return crop_folder, other_folder

def first_run(dicom_folder_path, image_folder_path, dicom_output_excel, image_output_excel):
    save_dicom_to_excel(dicom_folder_path, dicom_output_excel)
    save_images_to_excel(image_folder_path, image_output_excel)
    match_dicom_and_images(dicom_output_excel, image_output_excel, final_output_excel)

def second_run(image_folder_path, excel_file_path, new_output_directory, file_name,image_output_excel):
    process_and_save_to_new_excel(image_folder_path, excel_file_path, new_output_directory, file_name,image_output_excel)


directory = '../Data_collection-MD1/Fixation Data+3D Images'
directory_items = os.listdir(directory)
folder_count = sum(1 for item in directory_items if os.path.isdir(os.path.join(directory, item)))
start=1
end=19
for i in range(start, end+1):
    curr=str(i)
    root_directory = directory+"/recording"+curr
    dicom_folder_path = find_dicom_folder(root_directory)
    image_folder_paths = find_jpg_folders(root_directory)

    crop_folder_path,complete_image_folder_path = find_crop_and_other_folder(image_folder_paths)
    output_directory = "../New Dataset/Recording"+curr


    dicom_output_excel = os.path.join(output_directory, "dicom_data_R"+curr+".xlsx")
    image_output_excel = os.path.join(output_directory, "image_data_R"+curr+".xlsx")
    final_output_excel = os.path.join(output_directory, "matched_data_R"+curr+".xlsx")
    file_name = "Updated_matched_data_R"+curr+".xlsx"

    print(complete_image_folder_path,crop_folder_path)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    if(complete_image_folder_path is not None and dicom_folder_path is not None):
        first_run(dicom_folder_path, complete_image_folder_path, dicom_output_excel, image_output_excel)
        second_run(complete_image_folder_path, final_output_excel, output_directory,file_name,image_output_excel)

    # debug_first_run('../Recordings/recording 3/crop__fell from roof(3) agrawal/frame000424.jpg')
# debug_second_run('../Recordings/recording 15/chest 41 (15) agrawal/frame001338.jpg')
# debug_first_run('../recording/recording 17/chest 42 (17) agrawal/frame000500.jpg')

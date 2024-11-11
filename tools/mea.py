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
    ocr_text = extract_ocr_data(image_path)
    if ocr_text:
        series_number, slice_location, image_number = extract_series_and_slice_from_ocr(ocr_text,image_path)          
        if slice_location is None or slice_location == "":
            print("image_path", image_path)
            print("OCR Text", ocr_text)
            print("Slice location", slice_location)
            slice_location = "Missing" 
        print(f"Series Number: {series_number}, Slice Location: {slice_location}, Image Number: {image_number}")

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

def enhance_image_for_ocr(image_path):
    original_image = cv2.imread(image_path)
    
    if original_image is None:
        print(f"Error: Could not load image from {image_path}")
        return None

    # Initialize OpenCV's DNN Super-Resolution model (FSRCNN in this example)
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    
    # Load the pre-trained super-resolution model (FSRCNN x4 in this case)
    sr.readModel('FSRCNN_x4.pb') 

    # Set the model and scale factor (e.g., 4x upscaling)
    sr.setModel("fsrcnn", 4)

    # Apply super-resolution to the original image
    upscaled_image = sr.upsample(original_image)
    
    # Convert the upscaled image to grayscale
    grayscale_image = cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2GRAY)

    # # 5. Display the upscaled and grayscale images for comparison
    # plt.figure(figsize=(10, 5))

    # # Display original image
    # plt.subplot(1, 2, 1)
    # plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    # plt.title("Original Image")
    # plt.axis('off')

    # # Display the upscaled grayscale image
    # plt.subplot(1, 2, 2)
    # plt.imshow(grayscale_image, cmap='gray')
    # plt.title("Upscaled Grayscale Image")
    # plt.axis('off')

    # # Show the comparison
    # plt.tight_layout()
    # plt.show()

    # Return the upscaled grayscale image for OCR processing
    return grayscale_image

def extract_ocr_data(image_path):
    try:
        enhanced_image = enhance_image_for_ocr(image_path)


        # Convert the image to text using EasyOCR
        ocr_result = reader.readtext(enhanced_image, detail=0)  # detail=0 returns only the detected text
        print(f"Image Path: {image_path}")
        print(f"OCR Result: {ocr_result}")

        return ocr_result 
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None
    
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


    # Check for negative sign in the slice location
    if slice_location is not None and check_negative_sign(image_path) and slice_location >= 0:
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
                ocr_text = extract_ocr_data(image_path)
                
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
            print(dicom_path)
            print(int(dicom_series), int(image_series), int(dicom_image_number), int(image_number))
            if dicom_path is None and int(dicom_series) == int(image_series) and int(dicom_image_number) == int(image_number) : 
                print(dicom_path)
                print(dicom_series, image_series, dicom_image_number, image_number)
                import pdb; pdb.set_trace()
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

def check_negative_sign(image_path):
    # 1. Load the image in grayscale without equalization
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return False

    # 2. Define the Region of Interest (ROI) manually based on visual observation
    x_start, x_end = 45, 54
    y_start, y_end = 105, 115  
    roi = image[y_start:y_end, x_start:x_end]

    # 3. Apply thresholding to isolate white pixels that may indicate the negative sign
    ret, thresh = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY)

    # 4. Check if there are any white pixels (255) which may represent the negative sign
    white_pixels = np.where(thresh == 255)
    print(f"White Pixels: {len(white_pixels[0])}")
    negative_sign_present = len(white_pixels[0]) <=4 and len(white_pixels[0]) > 0

    if negative_sign_present:
        print("Negative sign detected in the selected ROI.")
    else:
        print("No negative sign detected in the selected ROI.")

    return negative_sign_present

dicom_folder_path = "../Recordings/recording 6/184786730(Leukocytosis chest wall pain)"
image_folder_path = "../Recordings/recording 6/crop__leukocytosis chest wall pain(6) agrawal"
output_directory = "../Excel Files Rec 2/Recording 2"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

curr_excel= "R2"
dicom_output_excel = os.path.join(output_directory, "dicom_data_"+curr_excel+".xlsx")
image_output_excel = os.path.join(output_directory, "image_data_"+curr_excel+".xlsx")
final_output_excel = os.path.join(output_directory, "matched_data_"+curr_excel+".xlsx")

# # # # Save DICOM and image metadata to separate Excel files
save_dicom_to_excel(dicom_folder_path, dicom_output_excel)
# save_images_to_excel(image_folder_path, image_output_excel)

# Match DICOM and image data and save the results to another Excel
# match_dicom_and_images(dicom_output_excel, image_output_excel, final_output_excel)

# debug("../Recordings/recording 15/crop__chest 41 (15) agrawal/frame000146.jpg")
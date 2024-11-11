import cv2
import pandas as pd
import os
import SimpleITK as sitk
from tqdm import tqdm
import re
import numpy as np

def read_dicom(path):
    dicom = sitk.ReadImage(path)
    # Convert the pixel data to a numpy array
    pixel_array = sitk.GetArrayFromImage(dicom)[0]
    # Normalize the pixel array to the range 0-255
    pixel_array = (
        (pixel_array - pixel_array.min())
        / (pixel_array.max() - pixel_array.min())
        * 255
    ).astype(np.uint8)
    return pixel_array

def create_video(final_excel, final_path, og_frames):


    # Ensure the directory exists
    os.makedirs(final_path, exist_ok=True)

    df_xlsx = pd.read_excel(final_excel)
    final_excel_csv = final_excel.replace("xlsx", "csv")
    df_xlsx.to_csv(final_excel_csv)


    combined_df= pd.read_csv(final_excel_csv)

    print(combined_df)
    middle_w = 723 + 391
    middle_h = 258 + 391

    keep_column = [
    "Fixation point X",
    "Fixation point Y",
    "Captured_time(micro_seconds)",
    "Captured_time(frame_id)",
    "fixation_alive",
    "Image Path",
    "DICOM Path",
    "Image Number",
    "Series Number DICOM",
    ]

    keep_combined_df = combined_df[keep_column]

    new_df = []
    # Iterate over each DICOM path in the dataframe and save as JPG
    for index, row in tqdm(keep_combined_df.iterrows(), total=len(keep_combined_df)):
        dicom_path = row["DICOM Path"]
        # Convert the pixel data to a numpy array
        pixel_array = read_dicom(
            dicom_path.replace(
                "../Recordings",
                "../recording",
            )
        )
        # Calculate the aspect ratio and new width
        (h, w) = pixel_array.shape[:2]
        aspect_ratio = w / h
        new_height = 781
        new_width = int(new_height * aspect_ratio)
        new_row = {col: row[col] for col in row.index.values.tolist()}
        fx, fy = row["Fixation point X"], row["Fixation point Y"]
        miny, maxy, minx, maxx = (
            258,
            258 + 781 - 1,
            middle_w - new_width // 2 - 1,
            middle_w + new_width // 2 - 1,
        )
        if fx > minx and fx < maxx and fy > miny and fy < maxy:
            new_row["valid"] = 1
            new_row["FX"] = (fx - minx) / new_width * w
            new_row["FY"] = (fy - miny) / new_height * h
        else:
            new_row["valid"] = 0
            new_row["FX"] = -1
            new_row["FY"] = -1
        new_df.append(new_row)

    new_dfs = pd.DataFrame(new_df)
    new_dfs.to_csv(final_path+"keep_combined_df_keep_invalid.csv", index=False)

    new_df2 = []
    # Iterate over each DICOM path in the dataframe and save as JPG
    for index, row in tqdm(new_dfs.iterrows(), total=len(new_dfs)):
        new_row = {col: row[col] for col in row.index.values.tolist()}
        if new_row["valid"] == 0:
            continue
        new_df2.append(new_row)

    new_dfs2 = pd.DataFrame(new_df2)
    new_dfs2.to_csv(final_path+"keep_combined_df_discard_invalid.csv", index=False)

    for _, row in tqdm(new_dfs2.iterrows(), total=len(new_dfs2)):
        if row["valid"] == 1:
            pixel_array = read_dicom(
                row["DICOM Path"].replace(
                    "../Recordings",
                    "../recording",
                )
            )
            # Draw a yellow circle at the fixation point
            pixel_array = cv2.cvtColor(pixel_array, cv2.COLOR_GRAY2BGR)
            circle_radius = 133 / 2  # max circle is 130 pixel in diameters
            # that circle is for 781 image height on 1080p, let's scale to 512
            circle_radius = circle_radius / 781 * 512
            radius = int(circle_radius * (0.5 + row["fixation_alive"] / 2 / 1_000_000))
            radius = min(radius, int(133 // 2 / 781 * 512))
            radius = max(radius, int(60 // 2 / 781 * 512))
            cv2.circle(
                pixel_array, (int(row["FX"]), int(row["FY"])), radius, (0, 255, 255), -1
            )
            # Save the image as a JPG file
            index = row["Captured_time(frame_id)"]
            output_path = os.path.join(
                final_path,"notebook_tmp_data4", f"frame{str(int(index)).zfill(6)}.jpg"
            )
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, pixel_array)

    # Define paths
    original_frames_path = og_frames
    notebook_frames_path = final_path+"notebook_tmp_data4/"
    output_path = final_path+"comparison_frames/"
    os.makedirs(output_path, exist_ok=True)

    # Get list of frame files
    original_frames = sorted(os.listdir(original_frames_path))
    notebook_frames = sorted(os.listdir(notebook_frames_path))

    # Define the size for scaling
    target_size = (1080, 1080)

    # Iterate over each frame in the original frames directory
    for frame_file in tqdm(original_frames):
        # Read the original frame
        original_frame_path = os.path.join(original_frames_path, frame_file)
        original_frame = cv2.imread(original_frame_path)

        # Resize the original frame to the target size
        original_frame_resized = original_frame

        # Check if the corresponding frame exists in notebook_tmp_data4
        notebook_frame_path = os.path.join(notebook_frames_path, frame_file)
        
        if os.path.exists(notebook_frame_path):
            # Read and resize the notebook frame
            notebook_frame = cv2.imread(notebook_frame_path)
            notebook_frame_resized = cv2.resize(notebook_frame, target_size)
        else:
            # Create a blank image if the frame is missing
            notebook_frame_resized = np.zeros((1080, 1080, 3), dtype=np.uint8)

        # Concatenate the frames side by side
        concatenated_frame = np.concatenate(
            (original_frame_resized, notebook_frame_resized), axis=1
        )

        # Save the concatenated frame
        output_frame_path = os.path.join(output_path, frame_file)
        cv2.imwrite(output_frame_path, concatenated_frame)

    # Define the path to the frames and the output video file
    frames_path = final_path+"comparison_frames/"
    output_video_path = final_path+"output_video.mp4"

    # Get the list of frame files
    frame_files = sorted([f for f in os.listdir(frames_path) if f.endswith(".jpg")])

    # Read the first frame to get the frame size
    first_frame = cv2.imread(os.path.join(frames_path, frame_files[0]))
    height, width, layers = first_frame.shape
    frame_size = (width, height)

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 30, frame_size)

    # Iterate over each frame file and write it to the video
    for frame_file in frame_files:
        frame = cv2.imread(os.path.join(frames_path, frame_file))
        video_writer.write(frame)

    # Release the VideoWriter object
    video_writer.release()

def find_crop_and_other_folder(folders):
    crop_folder = None
    other_folder = None

    for folder in folders:
        if 'crop' in folder.lower():
            crop_folder = folder
        else:
            other_folder = folder

    return crop_folder, other_folder

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

start=2
end=19
for i in range(start, end+1):
    curr=str(i)
    root_directory = "../recording/recording "+curr
    image_folder_paths = find_jpg_folders(root_directory)
    crop_folder_path,complete_image_folder_path = find_crop_and_other_folder(image_folder_paths)
    final_excel = "../Final Excel/Recording "+curr+"/Final_R"+curr+".xlsx"
    final_path = "../Processed/Recording "+curr+"/"
    print("--------------------------------------------------------------")
    print(root_directory)
    print(crop_folder_path)
    print(complete_image_folder_path)
    create_video(final_excel,final_path,complete_image_folder_path)

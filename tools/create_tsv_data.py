import pandas as pd 
import numpy as np 
import cv2 
from matplotlib import pyplot as plt 
from pathlib import Path
from tqdm import tqdm
import os


def create_tsv (root, dataframe, final_datasheet):
    print(root, dataframe, final_datasheet)
    dataframe.columns
    dataframe['Average calibration accuracy (degrees)']
    dataframe['Recording timestamp'].unique()
    dataframe['Recording start time'].unique()
    dataframe['Recording duration'].unique()
    chosen_cols = ['Recording timestamp', 'Recording start time',
       'Recording duration', 'Recording resolution height', 'Recording resolution width',
       'Recording monitor latency', 'Eyetracker timestamp',
       'Gaze point X', 'Gaze point Y',
       'Gaze point left X', 'Gaze point left Y', 'Gaze point right X',
       'Gaze point right Y', 'Gaze direction left X', 'Gaze direction left Y',
       'Gaze direction left Z', 'Gaze direction right X',
       'Gaze direction right Y', 'Gaze direction right Z', 'Validity left', 'Validity right',
       'Eye movement type', 'Gaze event duration',
       'Eye movement type index', 'Fixation point X', 'Fixation point Y',
    ]

    # for c in chosen_cols:
    #     print('Uniques for ', c)
    #     print(dataframe[c].unique())

    chosen_cols = ['Recording timestamp',       'Gaze point X', 'Gaze point Y',
       'Gaze point left X', 'Gaze point left Y', 'Gaze point right X',
       'Gaze point right Y', 'Gaze direction left X', 'Gaze direction left Y',
       'Gaze direction left Z', 'Gaze direction right X',
       'Gaze direction right Y', 'Gaze direction right Z', 'Validity left', 'Validity right',
       'Eye movement type', 'Gaze event duration', 'Fixation point X', 'Fixation point Y',
    ]

    recording_starttime = dataframe['Recording start time'].unique()[0]
    recording_duration = dataframe['Recording duration'].unique()[0]
    height = dataframe['Recording resolution height'].unique()[0]
    width = dataframe['Recording resolution width'].unique()[0]
    latency = dataframe['Recording monitor latency'].unique()[0]

    new_data = []
    FPS = 25
    for index, row in dataframe.iterrows():
        new_row = {col: row[col] for col in chosen_cols if col != 'Recording timestamp'}
        new_row['Captured_time(micro_seconds)'] = row['Recording timestamp']
        new_row['Captured_time(frame_id)'] = int(row['Recording timestamp']/1_000_000*FPS)
        if row['Sensor'] == 'Mouse': continue
        if new_row['Validity left'] == 'Valid' and new_row['Validity right'] == 'Valid':
            new_data.append(new_row)

    new_dataframe = pd.DataFrame(new_data)

    new_dataframe.to_csv('new_dataframe.csv', index=False) 
    new_dataframe.head(50)
    (255200-105302)/1000
    len(new_dataframe)

    # # Open the video file
    # video = cv2.VideoCapture(video_path)

    # # Check if the video opened successfully
    # if not video.isOpened():
    #     print("Error: Could not open video.")
    # else:
    #     # Retrieve the FPS of the video
    #     fps = video.get(cv2.CAP_PROP_FPS)
    #     print(f"Frames per second (FPS): {fps}")

    # # Release the video capture object
    # video.release()

    new_dataframe['Captured_time(frame_id)'].max()

    new_dataframe['Captured_time(micro_seconds)'].max()

    prev_x, prev_y = -1, -1
    prev_dur = -1
    first_met = -1
    new_data2 = []

    for index, row in new_dataframe.iterrows():
        if row["Eye movement type"] == "Fixation":
            x, y = row["Fixation point X"], row["Fixation point Y"]
            duration = row["Gaze event duration"]
            frame_id = row["Captured_time(frame_id)"]
            new_row = {col: row[col] for col in row.index.values.tolist()}
            new_row["fixation_alive"] = 0
            new_data2.append(new_row)
            if prev_x == -1 and prev_y == -1:
                prev_y = y
                prev_x = x
                prev_dur = row["Gaze event duration"]
                first_met = row["Captured_time(micro_seconds)"]
            else:
                if x != prev_x or y != prev_y or duration != prev_dur:
                    if new_data2[-2]["fixation_alive"] == 0:
                        new_data2[-2]["fixation_alive"] = (
                            new_data2[-2]["Gaze event duration"] * 1000
                        )
                        if (
                            new_data2[-2]["Fixation point Y"]
                            == new_data2[-3]["Fixation point Y"]
                            and new_data2[-2]["Fixation point X"]
                            == new_data2[-3]["Fixation point X"]
                            and new_data2[-2]["fixation_alive"]
                            < new_data2[-3]["fixation_alive"]
                        ):  # sanity check
                            print(prev_x, prev_y)
                            print(first_met)
                            print(new_data2[-3])
                            print(new_data2[-2])
                            raise NameError(
                                "new_dataframe2 is not defined or is not a DataFrame"
                            )
                    first_met = row["Captured_time(micro_seconds)"]
                    prev_y = y
                    prev_x = x
                    prev_dur = row["Gaze event duration"]
                else:
                    new_data2[-2]["fixation_alive"] = (
                        row["Captured_time(micro_seconds)"] - first_met
                    )


    new_data2[-1]["fixation_alive"] = (
        row["Gaze event duration"] * 1000
    )  # last point get the whole duration.
    new_dataframe2 = pd.DataFrame(new_data2)

    new_dataframe2.to_csv('new_dataframe2.csv', index=False)

    new_data3 = []
    # collapse to keep only one fixation per frame
    for index, row in new_dataframe2.iterrows():
        new_row = {col: row[col] for col in row.index.values.tolist()}
        if len(new_data3) != 0:
            prev = new_data3[-1]
            if prev['Fixation point X'] == new_row['Fixation point X'] and prev['Fixation point Y'] == new_row['Fixation point Y'] and prev['Captured_time(frame_id)'] == new_row['Captured_time(frame_id)']:
                new_data3[-1] = new_row
            else:
                new_data3.append(new_row)
        else:
            new_data3.append(new_row)


    new_dataframe3 = pd.DataFrame(new_data3)
    new_dataframe3.to_csv(final_datasheet, index=False)

    # # Define the height and width of the frames
    # height = 1080
    # width = 1920

    # frame_list = {i:[] for i in range(new_dataframe3['Captured_time(frame_id)'].max()+1)}
    # root_ = root
    # for index, row in new_dataframe3.iterrows():
    #     if row['Eye movement type'] == 'Fixation':
    #         x,y = row['Fixation point X'], row['Fixation point Y']
    #         duration = row['fixation_alive']
    #         frame_id = row['Captured_time(frame_id)'] 
    #         frame_list[frame_id].append((x,y,duration))
    #     # Define the circle properties
    #     circle_radius = 133 / 2 # max circle is 130 pixel in diameters
    #     circle_color = (0, 255, 255)  # Yellow color in BGR
    #     circle_thickness = -1  # Filled circle
    #     opacity = 0.2  # Opacity of the circle

    #     # Iterate through the total number of frames
    #     for frame_id,fixation_list in tqdm(frame_list.items(), total=len(frame_list)):
    #         # Create a black image
    #         if os.path.exists(os.path.join(root_, f"frame{frame_id:06d}.jpg")):
    #             frame = cv2.imread(os.path.join(root_, f"frame{frame_id:06d}.jpg"))
    #             overlay = frame.copy()
    #             if len(fixation_list) != 0: 
    #                 # Draw a yellow circle on the black image
    #                 for x,y,duration in fixation_list:
    #                     radius = int(circle_radius * (0.5+duration/2/1_000_000))
    #                     radius = min(radius, 133//2)
    #                     radius = max(radius,60//2)
    #                     circle_center = (int(x), int(y))
    #                     cv2.circle(overlay, circle_center, radius, circle_color, circle_thickness)
    #                 cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)
    #             # Save the frame (optional)
    #             # cv2.imwrite(f'./notebook_tmp_data/frame{frame_id:06d}.jpg', frame)
    #             if frame_id == 500: break 
    # os.path.join(root_, f"frame_{frame_id:06d}.jpg")

 # List all files and folders in the given directory
directory = '../Final Excel MD-1'
directory_items = os.listdir(directory)

# Count only directories
folder_count = sum(1 for item in directory_items if os.path.isdir(os.path.join(directory, item)))
start=24
end= 25
for i in range(start, folder_count+start):
    root='../MD1/Recording '+str(i)
    folder = Path(root)
    filepath = str(next(folder.glob("*.tsv"), None))
    dataframe = pd.read_csv(filepath, sep='\t')
    final_datasheet= "../MD-1 TSV/Final_R"+str(i)+".csv"
    create_tsv(root, dataframe, final_datasheet)


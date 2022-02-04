import h5py
import numpy as np
import os
import pandas as pd
import cv2

"""

gesture_event.hdf5
    - data
        - test
            - user24
                - event
                    -> [total(total of frames), x(width of frame), y(height of frame), 3(channel)]
                - label
            - user25
        - train
            - user01
            ...
            - user23


"""


def get_event_info(filename):
    video = cv2.VideoCapture(filename)

    # width of frame, height of frame, frames per second, total of frames in the video file
    x = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    y = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = video.get(cv2.CAP_PROP_FPS)
    total = video.get(cv2.CAP_PROP_FRAME_COUNT)

    # dimensions of each frame
    channel = 3 #is it correct?
    return np.array([int(total), int(x), int(y), channel])

def get_files_count(folder_path):
    # total of files in user's folder
    dirList = os.listdir(folder_path)
    return len(dirList)

def get_filename(root, user, num, filesNum):
    n = str(num)
    if filesNum < 100 and num < 10:
        n = '0' + n
    elif filesNum > 100 and num < 10:
        n = '00' + n
    elif filesNum > 100 and num >= 10 and num < 100:
        n = '0' + n
    if filesNum >= 100:
        return root + f'\\{user}_data_rgb\\{user}-{n}.mp4'
    return root + f'\\{user}_data_rgb\\{user}_data-{n}.mp4'

def get_user_name(user):
    u = str(user)
    if user < 10:
        return '0'+u
    return u

def create_hdf5(hdf_filename, csv_filename, root):
    label_file = pd.read_csv(csv_filename)

    USER = label_file['User']
    NUM = label_file['Sample Number']
    LABEL = label_file['Label']

    # list of total of files in user's folder
    files_num_list = []
    for user in set(USER):
        u = get_user_name(user)
        files_num_list.append(get_files_count(root + f'\\{u}_data_rgb'))

    with h5py.File(hdf_filename, 'w') as file:
        file.clear()
        f = file.create_group("data")

        current_user = 1
        u = '01'
        # width of frame, height of frame, total of frames in the video file, channel
        event_list = []
        # label
        label_list = []

        train_grp = f.create_group("train")
        test_grp = f.create_group("test")
        for user, num, label in zip(USER, NUM, LABEL):

            u = get_user_name(user)
            video_filename = get_filename(root, u, num, files_num_list[user-1])
            #print(video_filename, os.path.isfile(video_filename))
            # double check if the video file exists
            if os.path.isfile(video_filename):
                data = get_event_info(video_filename)
            else:
                print(video_filename, os.path.isfile(video_filename))
                continue

            event_list.append(data)
            # if label is nothing
            label_list.append(label if len(label)!=0 else 'other gestures')

            # if it finishes getting the information about all videos of a user,
            # create datasets under user's folder
            if num == files_num_list[user-1]:
                g = train_grp if user <= 23 else test_grp
                user_grp = g.create_group(f'user{get_user_name(user)}')
                event_dset = user_grp.create_dataset('event', data=event_list, dtype=np.uint8)
                label_dset = user_grp.create_dataset('label', data=label_list)

                event_list.clear()
                label_list.clear()

if __name__ == "__main__":
    #get_event_info(r'C:\Users\yeieu\Downloads\All_RGB_Data\01_data_rgb\01_data-01.mp4')
    create_hdf5("gesture_event.hdf5", 'Gesture_Data_Labels - Sheet1.csv', r'C:\Users\yeieu\Downloads\All_RGB_Data')

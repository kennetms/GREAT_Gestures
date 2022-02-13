import h5py
import numpy as np
import os
import pandas as pd
import cv2
from dv import AedatFile
from dv import LegacyAedatFile
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
    timestamp, polarity, xaddr, yaddr = [], [], [], []

    with LegacyAedatFile(filename) as f:
        for e in f:
            timestamp.append(e.timestamp)
            polarity.append(e.polarity)
            xaddr.append(e.x)
            yaddr.append(e.y)

    return np.column_stack([
        np.array(timestamp, dtype=np.uint32),
        np.array(polarity, dtype=np.uint8),
        np.array(xaddr, dtype=np.uint16),
        np.array(yaddr, dtype=np.uint16)])


def get_files_count(folder_path):
    # total of files in user's folder
    dirList = os.listdir(folder_path)
    return len(dirList)

def get_filename(root, user, num, filesNum):
    n = str(num)
    if filesNum < 100 and num < 10:
        n = '0' + n
    elif filesNum >= 100 and num < 10:
        n = '00' + n
    elif filesNum >= 100 and num >= 10 and num < 100:
        n = '0' + n
    if filesNum >= 100:
        if user in ['25', '24', '23', '21', '10', '03']:
            return root + f'\\{user}_data_events\\{user}_data-{n}\\{user}.aedat'
        return root + f'\\{user}_data_events\\{user}-{n}\\{user}.aedat'
    if user in ['19', '05', '01']:
        return root + f'\\{user}_data_events\\{user}_data-{n}\\{user}.aedat'
    if user == '11':
        return root + f'\\{user}_data_events\\{user}_data-{n}\\HB.aedat'
    return root + f'\\{user}_data_events\\{user}-{n}\\{user}.aedat'


def get_user_name(user):
    u = str(user)
    if user < 10:
        return '0'+u
    return u

def create_hdf5(hdf_filename, csv_filename, root):
    label_file = pd.read_csv(csv_filename)

    TRAIN_NUM = 23
    USER = label_file['User']
    NUM = label_file['Sample Number']
    LABEL = label_file['Label']

    event_list = []
    label_list = []
    test_keys = []
    train_keys = []

    # list of total of files in user's folder
    files_num_list = []
    for user in set(USER):
        u = get_user_name(user)
        files_num_list.append(get_files_count(root + f'\\{u}_data_events'))

    with h5py.File(hdf_filename, 'w') as file:
        file.clear()
        metas = []
        data_grp = file.create_group("data")
        extra_grp = file.create_group("extra")
        train_grp = data_grp.create_group("train")
        test_grp = data_grp.create_group("test")

        u = '01'
        for user, num, label in zip(USER, NUM, LABEL):
            istrain = 1 if user <= TRAIN_NUM else 0
            if num == 1:
                g = train_grp if istrain else test_grp
                user_grp = g.create_group(f'user{get_user_name(user)}')

            u = get_user_name(user)
            aedat_filename = get_filename(root, u, num, files_num_list[user-1])
            # double check if the video file exists
            if os.path.isfile(aedat_filename):
                data = get_event_info(aedat_filename)
            else:
                print(aedat_filename, os.path.isfile(aedat_filename))
                continue

            times = data[:,0]
            addrs = data[:,1:]
            event_list.append(data)
            # if label is nothing
            label_list.append(label if len(label)!=0 else 'other gestures')
            if istrain:
                train_keys.append((user, num))
                #train_label_list[label_dict[label]].append(num)
            else:
                test_keys.append((user, num))
                #test_label_list[label_dict[label]].append(num)

            # if it finishes getting the information about all videos of a user,
            # create datasets under user's folder
            metas.append({'num': str(num), 'training sample': istrain})
            sub_grp = user_grp.create_group(str(get_user_name(num)))
            time_dset = sub_grp.create_dataset('times', data=times, dtype=np.uint32)
            addr_dset = sub_grp.create_dataset('addrs', data=addrs, dtype=np.uint8)
            label_dset = sub_grp.create_dataset('labels', data=label_list)
            sub_grp.attrs['meta_info'] = str(metas[-1])
            event_list.clear()
            label_list.clear()

            #extra_sub_grp = extra_grp.create_group(f'user{get_user_name(user)}')
            #extra_sub_grp.create_dataset('train_keys', data = train_keys)
            #extra_grp.create_dataset('train_keys_by_label', data = train_label_list, dtype=np.uint8)
            #extra_grp.create_dataset('test_keys_by_label', data = test_label_list, dtype=np.uint8)
            #extra_sub_grp.create_dataset('test_keys', data = test_keys)
        extra_grp.create_dataset('train_keys', data = train_keys)
        extra_grp.create_dataset('test_keys', data=train_keys)
        extra_grp.attrs['N'] = len(train_keys) + len(test_keys)
        extra_grp.attrs['Ntrain'] = len(train_keys)
        extra_grp.attrs['Ntest'] = len(test_keys)

if __name__ == "__main__":
    #get_event_info(r'C:\Users\yeieu\Downloads\All_RGB_Data\01_data_rgb\01_data-01.mp4')
    create_hdf5("gesture_event.hdf5", 'Gesture_Data_Labels - Sheet1.csv', r'C:\Users\yeieu\Downloads\All_Event_Data')

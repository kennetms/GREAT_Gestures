import numpy as np
from torch import fill_
from tqdm import tqdm
import h5py
import os
import pandas as pd
import cv2
import glob
import pickle

def create_hdf5(directory, hdf5_filename, split_number, csv_path):
    '''
    Make Hdf5 file with RGB data

            Parameters : 
                    directory(str) : all data directories common path directory
                    hdf5_filename(str) : file name which is maken when this function performed
                    split_number(int) : which user id is smaller than split number belongs to train, the rest belong to test

    I wish below directory tree

    hdf5_filename
    |
    |--data
        |
        |-train ( group ) # user 1-23
            |
            |- user01 ( group )
                |- meta (attribute)
                |- data_001(dataset) # 001 mean sample number
                |- data_002
                |  .......
                |- labels
            |- user02
            |   ......
            |- user03
            |- user04
            | .......
        |-test  ( group ) # user split_numer - 25


        Why I construct this file tree is all gesture data(.mp4) has different length, so I think each has to have differnt dataset
        But It took too much time and stopped with OSError
        : Can't write data  (errno = 28, error message = 'No space left on device', buf = 000002BA16309020, total write size = 7121, bytes this sub-write = 7121, bytes actually written = 18446744073709551615, offset = 181744050443)
    '''

    Label_csv = pd.read_csv(csv_path) # 'Gesture_Data_Labels - Sheet1.csv'
    label_map = {}
    for value, key in enumerate(Label_csv.Label.unique()):
        label_map[key] = value

    # save label_mapping
    with open('label_map.pickle', 'wb') as f:
        pickle.dump(label_map, f, pickle.HIGHEST_PROTOCOL)

    with h5py.File(hdf5_filename, 'w') as f:
        f.clear()

        metas = []
        data_grp = f.create_group('data')
        train_grp = data_grp.create_group('train')
        test_grp = data_grp.create_group('test')
        for i in range(1, 26):
            fns = gather_dat(directory, i)
            # print(fns)
            user = i
            # print(user)
            is_train = i < split_number
            user_num = fill_zero(i, 2)
            if is_train:
                subgroup = train_grp.create_group(f'user{user_num}')
            else:
                subgroup = test_grp.create_group(f'user{user_num}')


            for file_d in tqdm(fns):
                # frames
                frame = []
                cap = cv2.VideoCapture(file_d)

                total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                frames_step = total_frames//20

                for i in range(20):
                    cap.set(1, i*frames_step)
                    success, image = cap.read()
                    image_re = cv2.resize(image, dsize=(224,224), interpolation = cv2.INTER_AREA)
                    frame.append(image_re)

                cap.release()


            # sample_num setting
                temp = file_d.split('-')[-1][:-4]
                sample_num = int(temp)

                sample_fill = fill_zero(sample_num, 3)

                subgroup.create_dataset(
                    f'data_{sample_fill}', data=frame, dtype=np.float64, compression='gzip')

            label = Label_csv[(Label_csv['User'] == user)]['Label'].values

            labels = list(map(lambda x: label_map[x], label))

            subgroup.create_dataset(
                f'labels', data=labels, dtype=np.int16)

            metas.append({'user': str(user), 'training sample': is_train})
            subgroup.attrs['meta_info'] = str(metas[-1])


def gather_dat(directory, id):
    '''
    Return file names in directory

            Parameters :
                    directory  (str) : data directory
                    id (int) : user number for directory
    '''

    fns = []
    user_num = fill_zero(id, 2)
    search_mask = user_num + directory
    glob_out = glob.glob('./'+search_mask+'/*')


    if len(glob_out) > 0:
        fns += glob_out
    return fns


def fill_zero(num, l):
    '''
    Return the number which is filled with zero for sorting file name

            Parameters:
                    num (int) : original number
                    l (int) : return number's expected length 

    '''
    fill_num = str(num).zfill(l)
    return fill_num


# if __name__ == "__main__":
#     directory = "_data_rgb"
#     hdf5_filename = "example2.hdf5"
#     create_hdf5(directory, hdf5_filename, 20, 'Gesture_Data_Labels - Sheet1.csv')

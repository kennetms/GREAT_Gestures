import numpy as np
from tqdm import tqdm
import h5py
import os
import pandas as pd
import cv2
import glob


def create_hdf5(directory, hdf5_filename, split_number):
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
                |- label_001 (dataset)
                |- data_002
                |- label_002
                |  .......
            |- user02
            |   ......
            |- user03
            |- user04
            | .......
        |-test  ( group ) # user 24-25


        Why I construct this file tree is all gesture data(.mp4) has different length, so I think each has to have differnt dataset
        But It took too much time and stopped with OSError
        : Can't write data  (errno = 28, error message = 'No space left on device', buf = 000002BA16309020, total write size = 7121, bytes this sub-write = 7121, bytes actually written = 18446744073709551615, offset = 181744050443)
    '''

    Label_csv = pd.read_csv('Gesture_Data_Labels - Sheet1.csv')
    label_map = {}
    for value, key in enumerate(Label_csv.Label.unique()):
        label_map[key] = value

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
            is_train = i < split_number
            user_num = fill_zero(i, 2)
            if is_train:
                subgroup = train_grp.create_group(f'user{user_num}')
            else:
                subgroup = test_grp.create_group(f'user{user_num}')

            frames = []
            for file_d in tqdm(fns):
                # frames
                frame = []
                cap = cv2.VideoCapture(file_d)

                while True:
                    retval, image = cap.read()

                    if not retval:
                        break
                    frame.append(image)

                temp = file_d.split('-')[-1][:-4]
                sample_num = int(temp)

                label = Label_csv[(Label_csv['User'] == user) & (
                    Label_csv['Sample Number'] == sample_num)]['Label'].values[0]
                label = label_map[label]

                sample_fill = fill_zero(sample_num, 3)
                subgroup.create_dataset(
                    f'data_{sample_fill}', data=frame, dtype=np.float64, compression='gzip')
                subgroup.create_dataset(
                    f'label_{sample_fill}', data=label, dtype=np.int16)

            metas.append({'user': str(user), 'training sample': is_train})
            subgroup.attrs['meta_info'] = str(metas[-1])

# c : int = 3


def gather_dat(directory, id):
    '''
    Return file names in directory

            Parameters :
                    directory  (str) : data directory
                    id (int) : user number for directory
    '''

    fns = []
    search_mask = str(id) + directory
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
    length = l - len(str(num))
    fill_num = str(num).zfill(length)
    return fill_num

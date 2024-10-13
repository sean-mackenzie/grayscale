from os.path import join, isdir
from os import listdir

import numpy as np
import pandas as pd


# ------------------------------------------------------------------------------------------------------------------
# I/O: READ FUNCTIONS - Processes and Details


def read_process_flow(fp):

    df = pd.read_excel(fp)

    pflow = {}
    for i in range(len(df)):

        step_dict = {}
        for k, v in df.to_dict().items():
            v = v[i]

            if k == 'details':
                v = eval(v)
            elif v == 'None':
                v = eval(v)

            step_dict.update({k: v})

        pflow.update({i + 1: step_dict})

    return pflow


def read_measurement_methods(profilometry):
    """
    measurement_methods = io.read_measurement_methods(profilometry='KLATencor-P7')

    :param profilometry:
    :return:
    """
    if profilometry == 'KLATencor-P7':
        data_profile = {'header': 'KLATencor-P7',
                        'filetype_read': '.txt', 'x_units_read': 1e-6, 'y_units_read': 1e-9,
                        'filetype_write': '.xlsx', 'x_units_write': 1e-6, 'y_units_write': 1e-6,
                        }
    elif profilometry == 'Dektak':
        data_profile = {'header': 'Dektak',
                        'filetype_read': '.txt', 'x_units_read': 1e-6, 'y_units_read': 1e-10,
                        'filetype_write': '.xlsx', 'x_units_write': 1e-6, 'y_units_write': 1e-6,
                        }
    else:
        raise ValueError("Profilometry tools are 'KLATencor-P7' or 'Dektak'.")

    data_etch_monitor = {'header': 'DSEiii-LaserMon', 'filetype_read': '.csv', 'filetype_write': '.xlsx'}
    data_optical = {'header': 'FluorScope', 'filetype_read': '.jpg', 'filetype_write': '.png'}
    data_misc = {'header': 'MLA150', 'filetype_read': '.png', 'filetype_write': '.png'}

    measurement_methods = {'Profilometry': data_profile,
                           'Etch Monitor': data_etch_monitor,
                           'Optical': data_optical,
                           'Misc': data_misc,
                           }

    return measurement_methods


# ------------------------------------------------------------------------------------------------------------------
# I/O: READ FUNCTIONS - Tools


def collect_Dektak(graycart_process):
    """
    process_files = io.collect_Dektak(self)

    :param graycart_process:
    :return:
    """

    files = [f for f in listdir(graycart_process.ppath) if f.endswith(graycart_process.pread)]
    files.sort()

    # drop filetype from filename
    drop_len = len(graycart_process.pread)

    # create a list of lists: [feature ids (to find peaks of), feature labels, file path to data]
    process_files = []
    for f in files:
        fn = f[:-drop_len]  # e.g., 'a1.csv' --> 'a1'; or, 'e1-a1.csv' --> 'e1-a1'

        if len(fn) == 2:
            print("Single feature scan: {}".format(fn))

            process_files.append([[graycart_process.features[fn].fid], [fn], f])

        else:
            f_start, f_stop = f[0:2], f[-drop_len - 2:-drop_len]
            print("Multiple feature scan (start, stop): ({}, {})".format(f_start, f_stop))

            if f_start in graycart_process.feature_labels:
                if f_start.startswith('a'):
                    process_files.append([graycart_process.feature_ids, graycart_process.feature_labels, f])
                else:
                    process_files.append([np.flip(graycart_process.feature_ids), np.flip(graycart_process.feature_labels), f])

    return process_files


def collect_KLATencor(graycart_process):
    """
    process_files = io.collect_KLATencor(self)

    :param graycart_process:
    :return:
    """

    files = [f[:-len(graycart_process.pread)] for f in listdir(graycart_process.ppath) if
             f.endswith(graycart_process.pread)]
    files.sort()

    # create a list of lists: [feature ids (to find peaks of), feature labels, file path to data]
    process_files = []
    for f in files:
        if f in graycart_process.feature_labels:
            process_files.append([[graycart_process.features[f].fid], [f], f + graycart_process.pread])

    return process_files


def collect_KLATencor_3D(graycart_process):

    # "normal" files (i.e., 0 degree rotation)
    files = [f[:-len(graycart_process.pread)] for f in listdir(graycart_process.ppath) if f.endswith(graycart_process.pread)]
    file_ids = files.copy()

    # rotation files
    files_rot = [f[:-len(graycart_process.pread)] for f in listdir(graycart_process.ppath + '/3d') if
                 f.endswith(graycart_process.pread)]

    files.extend(files_rot)
    files.sort()

    dict_process_files = {}
    for f in file_ids:
        f_ids = [ff for ff in files if ff.startswith(f)]

        rot_dict = {}
        for f_rot in f_ids:
            if 'rot' in f_rot:
                rot = f_rot.split('_rot')[1]
                rot_dict.update({int(rot): f_rot + graycart_process.pread})
            else:
                rot_dict.update({0: f_rot + graycart_process.pread,
                                 'fid': graycart_process.features[f].fid},
                                )

        dict_process_files.update({f: rot_dict})
    """
    process_dict_files = {'a1': {'fid': 1,
                                 '0': path_to_rot0,
                                 '45': path_to_rot45,
                                 '90': path_to_rot90,
                         }
    """

    return dict_process_files


# ------------------------------------------------------------------------------------------------------------------
# I/O: READ FUNCTIONS - Files


def read_scan(filepath, tool):
    # fp_PR = join(base_dir, fn_dir, fn + filetype_read)
    if tool == 'Dektak':
        df = pd.read_csv(filepath, skiprows=36)
        cols = df.columns
        df = df.drop(columns=cols[-1])
    elif tool == 'KLATencor-P7':
        df = pd.read_csv(filepath, names=['x', 'z'])
    return df

# ------------------------------------------------------------------------------------------------------------------
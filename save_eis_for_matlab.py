import scipy.io as sio
import numpy as np

def save_eis_for_matlab(ei_list, save_path):
    """
    Save a list of EI matrices [512 x 81 each] to a MATLAB .mat file.
    
    Args:
        ei_list: list of numpy arrays, each [512 x 81]
        save_path: string, path to .mat file
    """
    mat_dict = {'eis': np.empty((len(ei_list),), dtype=object)}
    for i, ei in enumerate(ei_list):
        mat_dict['eis'][i] = ei.astype(np.float32)
    sio.savemat(save_path, mat_dict)
    print(f"Saved {len(ei_list)} EIs to: {save_path}")
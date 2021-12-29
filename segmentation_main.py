# -*- coding: utf-8 -*-
"""
@author: Shahin Heidarian
Concordia Intelligent Signal & Information Processing (I-SIP)
The lung segmentation model is developed and implemented here: https://github.com/JoHof/lungmask
"""
# Note: Execute each section separately, otherwise there might be issues in importing the segmenation model.
# In the case of any error related to the execution of the segmentation model on your system, you can refer to:
# https://github.com/JoHof/lungmask
from lung_segmentation_module import DICOM_Segmentation

#%%
SAVE_OUTPUT = True
NORMALIZATION = True
SORT_ORDER = 'descending'
OUTPUT_SIZE = (512,512)

data_path = r'path containing DICOM folders' # path containing DICOM folders
lung_save_path = r'directory to save extracted lung regions as a numpy file'
mask_save_path = r'directory to save identified lung masks as a binary numpy file'

# improting the segmentaiton model
seg = DICOM_Segmentation(path = data_path,
                     lung_save_path = lung_save_path,
                     mask_save_path = mask_save_path,
                     segmentation_mode = 'multiple', # multiple studies (folders)
                     sort_order = SORT_ORDER,
                     output_size = OUTPUT_SIZE,
                     save_output = SAVE_OUTPUT,
                     normalization = NORMALIZATION)

#%% Run the segmentation
"""
Run each section specified by #%% seperately.
(e.g., use "run selection" button in Spyder instead of running the whole section. Or use different sections in Jupyter Notebook)
"""
seg.segment_files()

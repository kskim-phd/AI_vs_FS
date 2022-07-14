import pdb
import os
import numpy as np
import nibabel as nib
from hjy_center_crop import *
from pickle_load import *
from ResampleLinear import *
import cv2


BIOMARKER = 'E:\\FreeSurfer\\github\\pickle_data'

BRAIN_path = 'E:\\FreeSurfer\\github\\FS_seg\\BRAINMASK'
PUTAMEN_path = 'E:\\FreeSurfer\\github\\FS_seg\\PUTAMEN'
PALLIDUM_path = 'E:\\FreeSurfer\\github\\FS_seg\\PALLIDUM'
PONS_path = 'E:\\FreeSurfer\\github\\FS_seg\\PONS'
MIDBRAIN_path = 'E:\\FreeSurfer\\github\\FS_seg\\MIDBRAIN'
CAUDATE_path = 'E:\\FreeSurfer\\github\\FS_seg\\CAUDATE'
V3_path = 'E:\\FreeSurfer\\github\\FS_seg\\V3'

dirList_BRAIN = sorted(os.listdir(BRAIN_path))
dirList_PUTAMEN = sorted(os.listdir(PUTAMEN_path))
dirList_PALLIDUM = sorted(os.listdir(PALLIDUM_path))
dirList_PONS = sorted(os.listdir(PONS_path))
dirList_MIDBRAIN = sorted(os.listdir(MIDBRAIN_path))
dirList_CAUDATE = sorted(os.listdir(CAUDATE_path))
dirList_V3 = sorted(os.listdir(V3_path))

DCM_vol, PUTAMEN_vol, PALLIDUM_vol, PONS_vol, MIDBRAIN_vol, CAUDATE_vol, V3_vol = [], [], [], [], [], [], []

for subdir_dcm in dirList_BRAIN:
    print(subdir_dcm)
    filepath_dcm = "%s\\%s" % (BRAIN_path, subdir_dcm)
    DCM = nib.load(filepath_dcm).get_fdata()
    img = np.transpose(DCM, (2, 0, 1))
    DCM_vol.append(img)

for subdir_gt in dirList_PUTAMEN:
    print(subdir_gt)
    filepath_gt = "%s\\%s" % (GT_path, subdir_gt)
    GT = nib.load(filepath_gt).get_fdata()
    gt = np.transpose(GT, (2, 0, 1))
    PUTAMEN_vol.append(gt)

for subdir_gt in dirList_PALLIDUM:
    print(subdir_gt)
    filepath_gt = "%s\\%s" % (GT_path, subdir_gt)
    GT = nib.load(filepath_gt).get_fdata()
    gt = np.transpose(GT, (2, 0, 1))
    PALLIDUM_vol.append(gt)

for subdir_gt in dirList_PONS:
    print(subdir_gt)
    filepath_gt = "%s\\%s" % (GT_path, subdir_gt)
    GT = nib.load(filepath_gt).get_fdata()
    gt = np.transpose(GT, (2, 0, 1))
    PONS_vol.append(gt)

for subdir_gt in dirList_MIDBRAIN:
    print(subdir_gt)
    filepath_gt = "%s\\%s" % (GT_path, subdir_gt)
    GT = nib.load(filepath_gt).get_fdata()
    gt = np.transpose(GT, (2, 0, 1))
    MIDBRAIN_vol.append(gt)

for subdir_gt in dirList_CAUDATE:
    print(subdir_gt)
    filepath_gt = "%s\\%s" % (GT_path, subdir_gt)
    GT = nib.load(filepath_gt).get_fdata()
    gt = np.transpose(GT, (2, 0, 1))
    CAUDATE_vol.append(gt)

for subdir_gt in dirList_V3:
    print(subdir_gt)
    filepath_gt = "%s\\%s" % (GT_path, subdir_gt)
    GT = nib.load(filepath_gt).get_fdata()
    gt = np.transpose(GT, (2, 0, 1))
    V3_vol.append(gt)


DCM_vol = np.stack(DCM_vol, axis=0)
PUTAMEN_vol = np.stack(PUTAMEN_vol, axis=0)
PALLIDUM_vol = np.stack(PALLIDUM_vol, axis=0)
PONS_vol = np.stack(PONS_vol, axis=0)
MIDBRAIN_vol = np.stack(MIDBRAIN_vol, axis=0)
CAUDATE_vol = np.stack(CAUDATE_vol, axis=0)
V3_vol = np.stack(V3_vol, axis=0)



img_stack2_dcm = np.expand_dims(DCM_vol, axis=4)  # add axis
img_stack2_PUTAMEN = np.expand_dims(PUTAMEN_vol, axis=4)
img_stack2_PALLIDUM = np.expand_dims(PALLIDUM_vol, axis=4)
img_stack2_PONS = np.expand_dims(PONS_vol, axis=4)
img_stack2_MIDBRAIN = np.expand_dims(MIDBRAIN_vol, axis=4)
img_stack2_CAUDATE = np.expand_dims(CAUDATE_vol, axis=4)
img_stack2_V3 = np.expand_dims(V3_vol, axis=4)


img_stack2_dcm = img_stack2_dcm.astype('float32')
img_stack2_PUTAMEN = img_stack2_PUTAMEN.astype('float32')
img_stack2_PALLIDUM = img_stack2_PALLIDUM.astype('float32')
img_stack2_PONS = img_stack2_PONS.astype('float32')
img_stack2_MIDBRAIN = img_stack2_MIDBRAIN.astype('float32')
img_stack2_CAUDATE = img_stack2_CAUDATE.astype('float32')
img_stack2_V3 = img_stack2_V3.astype('float32')

save_pickle(
    BIOMARKER + '\\BRAINMASK_PUTAMEN_' + str(img_stack2_PUTAMEN.shape[0]) + '_' + str(img_stack2_PUTAMEN.shape[1]) + '_' + str(
        img_stack2_PUTAMEN.shape[2]) + '_' + str(img_stack2_PUTAMEN.shape[3]) + '.pkl', [img_stack2_dcm, img_stack2_PUTAMEN])

save_pickle(
    BIOMARKER + '\\BRAINMASK_PALLIDUM_' + str(img_stack2_PALLIDUM.shape[0]) + '_' + str(img_stack2_PALLIDUM.shape[1]) + '_' + str(
        img_stack2_PALLIDUM.shape[2]) + '_' + str(img_stack2_PALLIDUM.shape[3]) + '.pkl', [img_stack2_dcm, img_stack2_PALLIDUM])

save_pickle(
    BIOMARKER + '\\BRAINMASK_PONS_' + str(img_stack2_PONS.shape[0]) + '_' + str(img_stack2_PONS.shape[1]) + '_' + str(
        img_stack2_PONS.shape[2]) + '_' + str(img_stack2_PONS.shape[3]) + '.pkl', [img_stack2_dcm, img_stack2_PONS])

save_pickle(
    BIOMARKER + '\\BRAINMASK_MIDBRAIN_' + str(img_stack2_MIDBRAIN.shape[0]) + '_' + str(img_stack2_MIDBRAIN.shape[1]) + '_' + str(
        img_stack2_MIDBRAIN.shape[2]) + '_' + str(img_stack2_MIDBRAIN.shape[3]) + '.pkl', [img_stack2_dcm, img_stack2_MIDBRAIN])

save_pickle(
    BIOMARKER + '\\BRAINMASK_CAUDATE_' + str(img_stack2_CAUDATE.shape[0]) + '_' + str(img_stack2_CAUDATE.shape[1]) + '_' + str(
        img_stack2_CAUDATE.shape[2]) + '_' + str(img_stack2_CAUDATE.shape[3]) + '.pkl', [img_stack2_dcm, img_stack2_CAUDATE])

save_pickle(
    BIOMARKER + '\\BRAINMASK_V3_' + str(img_stack2_V3.shape[0]) + '_' + str(img_stack2_V3.shape[1]) + '_' + str(
        img_stack2_V3.shape[2]) + '_' + str(img_stack2_V3.shape[3]) + '.pkl', [img_stack2_dcm, img_stack2_V3])


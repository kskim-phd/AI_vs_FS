#!/usr/bin/env python
# coding: utf-8

# In[1]:


#settings

import nibabel as nib 
import numpy as np
import cv2 
import pdb
import math
import cv2
from scipy import ndimage 
import os


# In[2]:


import argparse 
parser = argparse.ArgumentParser(description='python Implementation')
parser.add_argument('--label', type = str, default =None, help='input_dir') 
args = parser.parse_args()
print('label:',args.label) 

label = args.label

# In[3]:


final_output_DIR = '/media/smc/Plus/FreeSurfer/'

FS_DIR = '/home/smc/workspace/parkinson_20210802/' + label + '_seg/mri/' #path for T1,aparc,brainstem
FIN_DIR = final_output_DIR + label + '/' #final_output/#patients

t1,t2,t3 = (256,256,128)

#call  aparc.nii
aparc_path = FS_DIR + 'aparc+aseg.nii.gz'
aparc = nib.load(aparc_path).get_fdata()

#call  brainstemSsLabels.nii
brainstem_path = FS_DIR + 'brainstemSsLabels.v12.FSvoxelSpace.nii.gz'
brainstem = nib.load(brainstem_path).get_fdata()

#call  brainmask.nii
brainmask_path = FS_DIR + 'brainmask.nii.gz'
brainmask = nib.load(brainmask_path).get_fdata()
brainmask_pixel_num = len(np.where(brainmask>0)[0])


#caudate
caudate = np.zeros(aparc.shape)
caudate[np.where((aparc==11) | (aparc==50))] = 1.0 

#putamen
putamen = np.zeros(aparc.shape)
putamen[np.where((aparc==12) | (aparc==51))] = 1.0 

#pallidum
pallidum = np.zeros(aparc.shape)
pallidum[np.where((aparc==13) | (aparc==52))] = 1.0 

#V3
V3 = np.zeros(aparc.shape)
V3[np.where(aparc==14)] = 1.0 

#midbrain
midbrain = np.zeros(brainstem.shape)
midbrain[np.where(brainstem==173)] = 1.0 

#pons
pons = np.zeros(brainstem.shape)
pons[np.where(brainstem==174)] = 1.0 



# In[4]:


T1_path = FS_DIR + 'T1_' + label + '.nii.gz'

#contents of orig_nii: FreeSurfer/mri/T1.mgz origin -not set to 0x0x0 AND original.nii(dicom->nii)
img_arr = nib.load(T1_path).get_fdata()  #shape: 256x256x256
brainmask_arr = brainmask[:] 


    
# z-axis based degree rotation 
a,b,c = img_arr.shape

d_range = 25 # <30
threshold = 20
upper_limit = 5

temp = np.sum(img_arr[:,:,c//2-d_range:c//2+d_range],axis=2)

x1 = a//2-d_range
y1 = np.max(np.where(temp[x1,:]>2*d_range*threshold))
x2 = a//2
y2 = np.max(np.where(temp[x2,:]>2*d_range*threshold))
x3 = a//2+d_range
y3 = np.max(np.where(temp[x3,:]>2*d_range*threshold))
dx1 = x2 - x1
dy1 = -(y2 - y1)
dx2 = x3 - x2
dy2 = -(y3 - y2)


if abs(dy1-dy2) < upper_limit: # and abs(y3 - y1) > upper_limit :
    dx = x3 - x1
    dy = -(y3 - y1)
    alpha = math.degrees(math.atan2(dy, dx))
    rotation1 = alpha 
 
    for ii in range(c):
        img_arr[:,:,ii] = ndimage.rotate(img_arr[:,:,ii], rotation1, reshape=False)
        brainmask_arr[:,:,ii] = ndimage.rotate(brainmask_arr[:,:,ii], rotation1, reshape=False) 
        caudate[:,:,ii] = ndimage.rotate(caudate[:,:,ii], rotation1, reshape=False)
        putamen[:,:,ii] = ndimage.rotate(putamen[:,:,ii], rotation1, reshape=False)
        pallidum[:,:,ii] = ndimage.rotate(pallidum[:,:,ii], rotation1, reshape=False)
        V3[:,:,ii] = ndimage.rotate(V3[:,:,ii], rotation1, reshape=False)
        midbrain[:,:,ii] = ndimage.rotate(midbrain[:,:,ii], rotation1, reshape=False)
        pons[:,:,ii] = ndimage.rotate(pons[:,:,ii], rotation1, reshape=False)


# x-axis based degree rotation 
temp = np.sum(img_arr[a//2-d_range:a//2+d_range,:,:],axis=0)
x1 = c//2-d_range
y1 = np.max(np.where(temp[:,x1]>2*d_range*threshold))
x2 = c//2
y2 = np.max(np.where(temp[:,x2]>2*d_range*threshold))
x3 = c//2+d_range
y3 = np.max(np.where(temp[:,x3]>2*d_range*threshold))
ndx1 = x2 - x1
ndy1 = -(y2 - y1)
ndx2 = x3 - x2
ndy2 = -(y3 - y2)


if abs(ndy1-ndy2) < upper_limit:# and abs(y3 - y1) > upper_limit : 
    dx = x3 - x1
    dy = -(y3 - y1)
    alpha = math.degrees(math.atan2(dy, dx))
    rotation2 = - alpha 
 
    for ii in range(a):
        img_arr[ii,:,:] = ndimage.rotate(img_arr[ii,:,:], rotation2, reshape=False)
        brainmask_arr[ii,:,:] = ndimage.rotate(brainmask_arr[ii,:,:], rotation2, reshape=False) 
        caudate[ii,:,:] = ndimage.rotate(caudate[ii,:,:], rotation2, reshape=False)
        putamen[ii,:,:] = ndimage.rotate(putamen[ii,:,:], rotation2, reshape=False)
        pallidum[ii,:,:] = ndimage.rotate(pallidum[ii,:,:], rotation2, reshape=False)
        V3[ii,:,:] = ndimage.rotate(V3[ii,:,:], rotation2, reshape=False)
        midbrain[ii,:,:] = ndimage.rotate(midbrain[ii,:,:], rotation2, reshape=False)
        pons[ii,:,:] = ndimage.rotate(pons[ii,:,:], rotation2, reshape=False)

# 90 degree rotation
for ii in range(a):
    img_arr[ii,:,:] = ndimage.rotate(img_arr[ii,:,:], -90, reshape=False)
    brainmask_arr[ii,:,:] = ndimage.rotate(brainmask_arr[ii,:,:], -90, reshape=False) 
    caudate[ii,:,:] = ndimage.rotate(caudate[ii,:,:], -90, reshape=False)
    putamen[ii,:,:] = ndimage.rotate(putamen[ii,:,:], -90, reshape=False)
    pallidum[ii,:,:] = ndimage.rotate(pallidum[ii,:,:], -90, reshape=False)
    V3[ii,:,:] = ndimage.rotate(V3[ii,:,:], -90, reshape=False)
    midbrain[ii,:,:] = ndimage.rotate(midbrain[ii,:,:], -90, reshape=False)
    pons[ii,:,:] = ndimage.rotate(pons[ii,:,:], -90, reshape=False)

        
# flip
for ii in range(c):
    img_arr[:,:,ii] = img_arr[::-1,:,ii]
    brainmask_arr[:,:,ii] = brainmask_arr[::-1,:,ii]     
    caudate[:,:,ii] = caudate[::-1,:,ii] 
    putamen[:,:,ii] = putamen[::-1,:,ii]
    pallidum[:,:,ii] = pallidum[::-1,:,ii]
    V3[:,:,ii] = V3[::-1,:,ii] 
    midbrain[:,:,ii] = midbrain[::-1,:,ii] 
    pons[:,:,ii] = pons[::-1,:,ii] 


# In[5]:


#crop T1(rotated z-axis, x-axis, rotation, flip) - img_arr_crp, biomarker_crp
temp=np.where(img_arr>threshold)

ax1=np.min(temp[0])
ax2=np.max(temp[0])
ax3=np.min(temp[1])
ax4=np.max(temp[1])
ax5=np.min(temp[2])
ax6=np.max(temp[2]) 

img_arr_crp = img_arr[ax1:ax2,ax3:ax4,ax5:ax6]
brainmask_arr_crp = brainmask_arr[ax1:ax2,ax3:ax4,ax5:ax6] 
caudate_crp = caudate[ax1:ax2,ax3:ax4,ax5:ax6]
putamen_crp = putamen[ax1:ax2,ax3:ax4,ax5:ax6]
pallidum_crp = pallidum[ax1:ax2,ax3:ax4,ax5:ax6]
V3_crp = V3[ax1:ax2,ax3:ax4,ax5:ax6]
midbrain_crp = midbrain[ax1:ax2,ax3:ax4,ax5:ax6]
pons_crp = pons[ax1:ax2,ax3:ax4,ax5:ax6]



# In[6]:


#original data crop - img_org_crp
threshold_org = 40000
nii_org_path = final_output_DIR + 'orig_nii/' + label + '.nii' #DICOM->NII
img_org = nib.load(nii_org_path).get_fdata() 
temp = np.where(img_org>threshold_org)
img_org_crp = img_org[np.min(temp[0]):np.max(temp[0]),np.min(temp[1]):np.max(temp[1]),np.min(temp[2]):np.max(temp[2])]


# In[7]:


#reize to (256x256x128)
def ResampleLinear1D(original, targetLen):
    original = np.array(original, dtype=np.float)
    index_arr = np.linspace(0, len(original)-1, num=targetLen, dtype=np.float)
    index_floor = np.array(index_arr, dtype=np.int) #Round down
    index_ceil = index_floor + 1
    index_rem = index_arr - index_floor #Remain

    val1 = original[index_floor]
    val2 = original[index_ceil % len(original)]
    interp = val1 * (1.0-index_rem) + val2 * index_rem
    assert(len(interp) == targetLen)
    return interp


# In[8]:


#img_org_crp resize & save
a,b,c = img_org_crp.shape

img_stack = np.zeros((t1,t2,c))
img_stack2 = np.zeros((t1,t2,t3)) 
for idx in range(c):
    img = img_stack[:, :, idx]
    img_sm = cv2.resize(img_org_crp[:, :, idx], (t1, t2), interpolation=cv2.INTER_CUBIC)
    img_stack[:, :, idx] = img_sm
for idx1 in range(t1):
    for idx2 in range(t2):
        img_stack2[idx1,idx2,:] = ResampleLinear1D(img_stack[idx1,idx2, :], t3)       

#brainmask_arr_crp resize & save
a,b,c = img_org_crp.shape

img_stack = np.zeros((t1,t2,c))
img_stack2 = np.zeros((t1,t2,t3)) 
for idx in range(c):
    img = img_stack[:, :, idx]
    img_sm = cv2.resize(brainmask_arr_crp[:, :, idx], (t1, t2), interpolation=cv2.INTER_CUBIC)
    img_stack[:, :, idx] = img_sm
for idx1 in range(t1):
    for idx2 in range(t2):
        img_stack2[idx1,idx2,:] = ResampleLinear1D(img_stack[idx1,idx2, :], t3)       



# In[9]:


#others(T1, biomarkers) resize & save
a,b,c = img_arr_crp.shape
   
#img_arr_crp resize
img_stack = np.zeros((t1,t2,c)) #t1,t2,t3 = (480,480,160)
img_stack2 = np.zeros((t1,t2,t3)) 
for idx in range(c):
    img = img_stack[:, :, idx]
    img_sm = cv2.resize(img_arr_crp[:, :, idx], (t1, t2), interpolation=cv2.INTER_CUBIC)
    img_stack[:, :, idx] = img_sm
for idx1 in range(t1):
    for idx2 in range(t2):
        img_stack2[idx1,idx2,:] = ResampleLinear1D(img_stack[idx1,idx2, :], t3)
temp = nib.Nifti1Image(img_stack2, affine=np.eye(4))
nib.save(temp, FIN_DIR + label + '_Fs_T1.nii.gz') 

#pons_crp
img_stack = np.zeros((t1,t2,c))
img_stack2 = np.zeros((t1,t2,t3)) 
for idx in range(c):
    img = img_stack[:, :, idx]
    img_sm = cv2.resize(pons_crp[:, :, idx], (t1, t2), interpolation=cv2.INTER_CUBIC)
    img_stack[:, :, idx] = img_sm
for idx1 in range(t1):
    for idx2 in range(t2):
        img_stack2[idx1,idx2,:] = ResampleLinear1D(img_stack[idx1,idx2, :], t3)
temp = nib.Nifti1Image(img_stack2, affine=np.eye(4))
nib.save(temp, FIN_DIR + label + '_pons_'+str(len(np.where(img_stack2>0.1)[0]))+'.nii.gz') 

#midbrain_crp
img_stack = np.zeros((t1,t2,c))
img_stack2 = np.zeros((t1,t2,t3)) 
for idx in range(c):
    img = img_stack[:, :, idx]
    img_sm = cv2.resize(midbrain_crp[:, :, idx], (t1, t2), interpolation=cv2.INTER_CUBIC)
    img_stack[:, :, idx] = img_sm
for idx1 in range(t1):
    for idx2 in range(t2):
        img_stack2[idx1,idx2,:] = ResampleLinear1D(img_stack[idx1,idx2, :], t3)
temp = nib.Nifti1Image(img_stack2, affine=np.eye(4))
nib.save(temp, FIN_DIR + label + '_midbrain_'+str(len(np.where(img_stack2>0.1)[0]))+'.nii.gz') 

#caudate_crp
img_stack = np.zeros((t1,t2,c))
img_stack2 = np.zeros((t1,t2,t3)) 
for idx in range(c):
    img = img_stack[:, :, idx]
    img_sm = cv2.resize(caudate_crp[:, :, idx], (t1, t2), interpolation=cv2.INTER_CUBIC)
    img_stack[:, :, idx] = img_sm
for idx1 in range(t1):
    for idx2 in range(t2):
        img_stack2[idx1,idx2,:] = ResampleLinear1D(img_stack[idx1,idx2, :], t3)
temp = nib.Nifti1Image(img_stack2, affine=np.eye(4))
nib.save(temp, FIN_DIR + label + '_caudate_'+str(len(np.where(img_stack2>0.1)[0]))+'.nii.gz') 

#putamen_crp
img_stack = np.zeros((t1,t2,c))
img_stack2 = np.zeros((t1,t2,t3)) 
for idx in range(c):
    img = img_stack[:, :, idx]
    img_sm = cv2.resize(putamen_crp[:, :, idx], (t1, t2), interpolation=cv2.INTER_CUBIC)
    img_stack[:, :, idx] = img_sm
for idx1 in range(t1):
    for idx2 in range(t2):
        img_stack2[idx1,idx2,:] = ResampleLinear1D(img_stack[idx1,idx2, :], t3)
temp = nib.Nifti1Image(img_stack2, affine=np.eye(4))
nib.save(temp, FIN_DIR + label + '_putamen_'+str(len(np.where(img_stack2>0.1)[0]))+'.nii.gz') 

#pallidum_crp
img_stack = np.zeros((t1,t2,c))
img_stack2 = np.zeros((t1,t2,t3)) 
for idx in range(c):
    img = img_stack[:, :, idx]
    img_sm = cv2.resize(pallidum_crp[:, :, idx], (t1, t2), interpolation=cv2.INTER_CUBIC)
    img_stack[:, :, idx] = img_sm
for idx1 in range(t1):
    for idx2 in range(t2):
        img_stack2[idx1,idx2,:] = ResampleLinear1D(img_stack[idx1,idx2, :], t3)
temp = nib.Nifti1Image(img_stack2, affine=np.eye(4))
nib.save(temp, FIN_DIR + label + '_pallidum_'+str(len(np.where(img_stack2>0.1)[0]))+'.nii.gz') 

#V3_crp
img_stack = np.zeros((t1,t2,c))
img_stack2 = np.zeros((t1,t2,t3)) 
for idx in range(c):
    img = img_stack[:, :, idx]
    img_sm = cv2.resize(V3_crp[:, :, idx], (t1, t2), interpolation=cv2.INTER_CUBIC)
    img_stack[:, :, idx] = img_sm
for idx1 in range(t1):
    for idx2 in range(t2):
        img_stack2[idx1,idx2,:] = ResampleLinear1D(img_stack[idx1,idx2, :], t3)
temp = nib.Nifti1Image(img_stack2, affine=np.eye(4))
nib.save(temp, FIN_DIR + label + '_V3_'+str(len(np.where(img_stack2>0.1)[0]))+'.nii.gz') 






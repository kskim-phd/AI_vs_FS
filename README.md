# AI_vs_FS
# Comparative Validation of AI against non-AI in MRI Volumetry for Diagnosis of Parkinsonian Syndrome

We released preprocessing FS, V-Net, and UNETR codes.

Collaborators: Juyoung Hahm, Kyungsu Kim

Detailed instructions for testing the 3D images are as follows.

## Implementation
TesnorFlow and PyTorch implementation are based on original V-Net & UNETR code.

The original code of V-Net, UNETR are the following:

V-net: https://github.com/faustomilletari/VNet]

UNETR: https://github.com/Project-MONAI/research-contributions/tree/main/UNETR/BTCV

## Main reference Package

matplotlib

monai

tqdm

shutil

pdb

pandas

numpy

time

pytorch

tensorflow

## Multi view dataset
Please request an email (kskim.doc@gmail.com) for the inference sample data (as this work is under review, it is open to reviewers only).

## Environment and data preparation
Instructions on how to run FreeSurfer are uploaded in FS_readme.txt.

Set segmentation file into 'folder_number'_seg.

## V-Net
Please read VNET_readme.txt to run the V-Net code.

Using skull-striped volume from FreeSurfer as input for V-Net, we obtained the labels and GT in pickle format through FreeSurfer.py, nii_to_pkl.py.

With the command 'python main_vnet.py,' the deep learning (DL) models' training and evaluation described in the research paper are conducted.

V-Net performs the evaluation using GPU and CPU and presents the processing time for each frameworks.


## UNETR
Please read UNETR_readme.txt to run the UNETR code.

Create 'Image_Tr' and 'Image_Ts' folder and obtain the labels and GT.

Train the model with the command 'python main_unetr.py' and evaluate with the command 'python eval_unetr.py'.

UNETR performs the evaluation using GPU and CPU and presents the processing time for each frameworks.

## Segmentation result

![KakaoTalk_20230116_213508472](https://user-images.githubusercontent.com/70966997/212679697-6022602b-0dcf-4365-bf5c-b831db225132.jpg)

a): V-Net b): UNETR
Segmentation results of a) CNN-based V-Net and b) ViT-based UNETR (left 3D images in first column and red-highlighted areas in second column) and FS (right 3D images in first column and blue-highlighted areas in second column) for each brain structure. 

## Acknowledgement

V-Net [https://github.com/faustomilletari/VNet] (Thanks to Fausto Milletari and Sagar Hukkire)

UNETR [https://github.com/Project-MONAI/research-contributions/tree/main/UNETR/BTCV] (Thanks to Ali Hatamizadeh and other contributors)

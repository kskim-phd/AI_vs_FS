# AI_vs_FS
# Comparative Validation of AI against non-AI in MRI Volumetry for Diagnosis of Parkinsonian Syndrome

We released preprocessing FS, V-Net, and UNETR codes.

Collaborators: Juyoung Hahm, Kyungsu Kim

Detailed instructions for testing the 3D images are as follows.

## Implementation
TesnorFlow and PyTorch implementation are based on original V-Net & UNETR code.

V-Net [https://github.com/faustomilletari/VNet] (Thanks to Fausto Milletari and Sagar Hukkire)

UNETR [https://github.com/Project-MONAI/research-contributions/tree/main/UNETR/BTCV] (Thanks to Ali Hatamizadeh and other contributors)

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

Using skull-striped volume from FreeSurfer as input for V-Net, we obtained the label and GT in pickle format through FreeSurfer.py, nii_to_pkl.py.

With the command 'python main_vnet.py,' the deep learning (DL) models' training and evaluation described in the research paper are conducted.

V-Net perform the evaluation using GPU and CPU and present the processing time for each frameworks.


## UNETR

UNETR_readme.txt 를 읽어주세요.

UNETR 은 Image_Tr 및 Image_Ts 폴더를 만들어 label, GT 를 불러옵니다.

python main_unetr 를 실행하여 학습을 진행한뒤

python eval_unetr 를 실행하여 평가를 진행합니다.

UNETR은 GPU 및 CPU 의 평가를 진행하며 각각의 평가 시간을 출력한다.



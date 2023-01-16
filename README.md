# AI_vs_FS
# Comparative Validation of AI against non-AI in MRI Volumetry for Diagnosis of Parkinsonian Syndrome

We release preprocessing FS, V-Net, UNETR code.

Collaborators: Juyoung Hahm, Kyungsu Kim

Detailed instructions for testing the 3D image are as follows.

## Implementation
A Tesnorflow & Pytorch implementation of segmentation based on original V-Net & UNETR code.

V-Net [https://github.com/faustomilletari/VNet] (Thanks for Fausto Milletari and Sagar Hukkire.)

UNETR [https://github.com/Project-MONAI/research-contributions/tree/main/UNETR/BTCV] (Thank you for Ali Hatamizadeh and other contributors)

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
Please send me a request email (kskim.doc@gmail.com) for that inference sample data (As this work is under review, so it is open to reviewers only)

## Prepare environment and data:
Instructions to how to run FreeSurfer is upload in FS_readme.txt

Set segmentation file into 'folder_number'_seg

## V-Net
VNET_readme.txt 를 읽어주세요.

VNET 은 피클로되어있는 Freesurfer skull striped 볼륨을 인풋으로 이용하여 Freesurfer.py를통해 label을 획득합니다.

python main_vnet.py 를 통해 논문에 기술한 실험 학습 및 평가를 실시한다.

V-Net은 GPU 및 CPU 의 평가를 진행하며 각각의 평가 시간을 출력한다.





## UNETR
After running V-Net, read UNETR_readme.txt


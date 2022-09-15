# AI_vs_FS
# Comparative Validation of AI against non-AI in MRI Volumetry for Diagnosis of Parkinsonian Syndrome

We release preprocessing FS, V-Net, UNETR code.

Collaborators: Juyoung Hahm, Kyungsu Kim

Detailed instructions for testing the 3D image are as follows.

## Implementation
A PyTorch implementation of segmentatino based on original V-Net & UNETR code.

FreeSurfer [https://surfer.nmr.mgh.harvard.edu/fswiki]

V-Net [https://github.com/faustomilletari/VNet] (Thanks to Fausto Milletari and Sagar Hukkire.)

UNETR [https://github.com/Project-MONAI/research-contributions/tree/main/UNETR/BTCV] (Thanks to Ali Hatamizadeh and other contributors)

## Prepare environment and data:
Instructions to how to run FreeSurfer is upload in FS_readme.txt

## V-Net
After running FreeSurfer, read VNET_readme.txt

## UNETR
After running V-Net, read UNETR_readme.txt

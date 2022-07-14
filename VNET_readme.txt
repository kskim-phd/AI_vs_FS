readme.txt

Prepare environment and data:
*read FS_readme.txt for running the FS.
**set segmentation file into 'folder_number'_seg

1. After FreeSurfer process, make nii.gz file for:  
	- brainstemSsLabels.v12.FSvoxelSpace.mgz
	- aparc+aseg.mgz
	- T1.mgz
	- brainmask.mgz
in the FS segmentation folder, 'folder_number'_seg/mri
and save to same folder('folder_name'_seg/mri/)

How to convert .mgz to .nii.gz:
##Use 'mri_convert' in bash: (example below)
#mri_convert brainmask.mgz brainmask.nii.gz	 
#mri_convert T1.mgz T1.nii.gz
#mri_convert aparc+aseg.mgz aparc+aseg.nii.gz
#mri_convert brainstemSsLabels.v12.FSvoxelSpace.mgz brainstemSsLabels.v12.FSvoxelSpace.nii.gz

2. run FreeSurfer.py
python FreeSurfer.py --label "$i"

#this file corrects the tilted/fliped/rotated mri images into default
#Also changes the mri size to 256x256x128
#output: all 6 brain parts individual nii.gz files

3. Change nii.gz to .pkl
#separate the brain parts into individual folder before running: BRAINMASK, CAUDATE, MIDBRAIN, PALLIDUM, PONS, PUTAMEN, V3
python nii_to_pkl.py 

4. Run V-Net model
python main.py



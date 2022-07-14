# ===== Instruction for FreeSurfer Installation & Usage =====
# Last update: June 28, 2021
 
# ===== Installation =====
# FreeSurfer version: 7 (May 2020)
# ref: https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall

#####################################################
sudo apt-get update

sudo apt-get install ./freesurfer_7-dev_amd64.deb

ls /usr/local/freesurfer/7-dev/

echo "export XDG_RUNTIME_DIR=$HOME/.xdg" >> $HOME/.bashrc

echo "export FREESURFER_HOME=/usr/local/freesurfer/7-dev" >> $HOME/.bashrc

echo "export FS_LICENSE=$HOME/license.txt" >> $HOME/.bashrc

echo "source /usr/local/freesurfer/7-dev/SetUpFreeSurfer.sh" >> $HOME/.bashrc

# end and restart terminal 
 
#(IGNORE!)# export DISPLAY=:1.0
#####################################################

# Brainstem matlab installation
# ref: https://surfer.nmr.mgh.harvard.edu/fswiki/MatlabRuntime
sudo apt-get update
sudo apt-get upgrade
fs_install_mcr R2014b

# ===== Usage =====
# 1. convert a series of dcm files to a nii file
# ref: https://www.nitrc.org/plugins/mwiki/index.php/dcm2nii:MainPage
# (1) open MRIcroGL program: MRIcroGL_dcm2nii/MRIcroGL/MRIcroGL.exe
# (2) go to Import >> Convert DICOM to NIfTI
# (3) set up variables as below:
# - Output Filename: %f
# - Output Directory: Select the folder where your output (nii) will be saved (e.g. js_workspace/neuro)
# - Output Format: Uncompressed NIfTI (.nii)
# - Create BIDS Sidercar: Yes, Anonymized
# - Advanced: Precise Philips Scaling
# (4) click 'Select Folder To Convert': select the folder where your input (a series of dcm) is located (e.g. js_workspace/neuro/dicom data/279844)
# (5) once the conversion is done, 'Conversion required xx seconds ~' would be shown in the interface
# (6) check that your output is successfully created (e.g. js_workspace/neuro/279844.nii)
# DONE

# 2. set up FreeSurfer environment 
# (1) open terminal and enter the command below to specify SUBJECTS_DIR 
# export SUBJECTS_DIR='the directory location where the nii file is located and the segmented output folder will be saved'
export SUBJECTS_DIR=/home/smc/workspace/js_workspace/neuro
# (2) in terminal, enter the command below to move to SUBJECTS_DIR
cd $SUBJECTS_DIR

# 3. perform biomarker segmentation
# (1) in terminal, enter the command below to perform cortical / subcortical reconstruction process
# ref: https://freesurfer.net/fswiki/recon-all?highlight=%28recon%5C-all%29
# recon-all -all -i 'nii_file_name_to_segment' -s 'output_folder_name'
recon-all -all -i 279844.nii -s 279844_seg
# NOTE: the output folder will be automatically created; if the folder is already existed, it will cause an error
# NOTE: this would take about 4-5 hours per nii file
# (2) in terminal, enter the command below to continously perform brainstem substructure segmentation
# ref: https://freesurfer.net/fswiki/BrainstemSubstructures?highlight=%28brainstem%29
# segmentBS.sh 'output_folder_name' $SUBJECTS_DIR
segmentBS.sh 279844_seg $SUBJECTS_DIR
# NOTE: the output folder name here is same as the one from (1)
# NOTE: this would take about 1 hours per nii file

# 4. view the segmentatio result 
# (1) in terminal, move to mri folder in the output folder
# cd 'output_folder_name/mri'
cd 279844_seg/mri
# (2) in terminal, view mri and segmented result in FreeSurfer
# freeview -v 'mri_file_name' 'cortical_segmentation_file':colormap=lut 'brainstem_segmentation_file':colormap=lut  
# NOTE: output file name could be varied but similar as an example below
freeview -v T1.mgz aparc+aseg.mgz:colormap=lut brainstemSsLabels.v12.mgz:colormap=lut  

# 5. check the volumetric analysis
# ref: https://freesurfer.net/fswiki/FsTutorial/VolumetricGroupAnalysis?highlight=%28volumetric%29%7C%28analysis%29
# in Files, go to 279844_seg/stats and check aseg.stats and brainstem.v12.stats



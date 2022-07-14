readme.txt

Prepare environment and data:
set folder for: 
'imagesTr', 'imagesTs' (for brainmask)
'labelsTr_caudate', 'labelsTr_midbrain', 'labelsTr_pallidum', 'labelsTr_pons', 'labelsTr_putamen', 'labelsTr_V3' (for training)
'labelsTs_caudate', 'labelsTs_midbrain', 'labelsTs_pallidum', 'labelsTs_pons', 'labelsTs_putamen', 'labelsTs_V3' (for testing)
and put 'VNET_readme.txt' - step 2 output

1. run UNETR model
python main_unetr.py

2. segmentation evaluation & save predicted segmentation
python eval_unetr.py
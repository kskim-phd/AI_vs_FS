import os
import shutil
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)


import torch
import pdb
from pickle_load import *
import time
import glob
import nibabel as nib


root_dir = '/home/dgxadmin/workspace/hjy_workspace/unetr/workspace/hjy_workspace/unetr/' # unetr 이 존재하는 경로를 작성하시오.
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        ToTensord(keys=["image", "label"]),
    ]
)

data_dir = "/media/dgxadmin/Seagate/hjy/unetr_data/A/"# 데이터가 존재하는 root dir 를 작성하세요.
split_JSON = "brain_V3.json"  # 데이터 정보가 담긴 json 혹은 txt 경로를 작성하시오.
datasets = data_dir + split_JSON
datalist = load_decathlon_datalist(datasets, True, "training")
val_files = load_decathlon_datalist(datasets, True, "validation")


val_ds = CacheDataset(
    data=val_files, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=4
)
val_loader = DataLoader(
    val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print_config()


model = UNETR( # model load
    in_channels=1,
    out_channels=2, #14
    img_size=(96, 96, 96),
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    pos_embed="perceptron",
    norm_name="instance",
    res_block=True,
    dropout_rate=0.0,
).to(device)




model.load_state_dict(torch.load('/home/dgxadmin/workspace/hjy_workspace/unetr/workspace/hjy_workspace/unetr/V3_A.pth')) # weight load
model.eval()
result = []


start = time.time()

# GPU use evalutation
result = []
with torch.no_grad(): # Use GPU
  for i in range(136):
    img_name = os.path.split(val_ds[i]["image_meta_dict"]["filename_or_obj"])[1]
    img = val_ds[i]["image"]
    val_inputs = torch.unsqueeze(img, 1).cuda()
    val_outputs = sliding_window_inference(
        val_inputs, (96, 96, 96), 4, model, overlap=0.8
    )
    stop = time.time()
    print(f"GPU time: {stop - start}s")
    result.append(torch.argmax(val_outputs, dim=1).detach().cpu().numpy()[0, :, :, :])
    
# CPU use evaluation
with torch.no_grad(): # Use CPU
  for i in range(136):
    img_name = os.path.split(val_ds[i]["image_meta_dict"]["filename_or_obj"])[1]
    img = val_ds[i]["image"]
    val_inputs = torch.unsqueeze(img, 1)
    val_outputs = sliding_window_inference(
        val_inputs, (96, 96, 96), 4, model, overlap=0.8
    )
    stop = time.time()
    print(f"CPU time: {stop - start}s")
    result.append(torch.argmax(val_outputs, dim=1).detach().cpu().numpy()[0, :, :, :])


result_path = '/media/dgxadmin/Seagate/hjy/unetr_data/A/pickle_A/'
result_file = 'V3_A.pkl' # result pickle load
result = np.array(result)
save_pickle(result_path+ result_file, result)   





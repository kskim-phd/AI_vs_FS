import pandas as pd
from setting import * # setting 에 따른 data 
from pickle_load import *
import os
import pdb
import numpy as np
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" # Use GPU 그래픽카드 번호에 따른 GPU 번호 입력
# Load data, DCM_AUG_vol.shape = (#,128,256,256,1)

[DCM_vol, GT_vol] = load_pickle(DATA_PATH_A)
train_data, train_gt = DCM_vol, GT_vol

[DCM_vol, GT_vol] = load_pickle(DATA_PATH_B)
train_data = np.append(train_data, DCM_vol, axis = 0)
train_gt = np.append(train_gt, DCM_vol, axis = 0)


[DCM_vol, GT_vol] = load_pickle(DATA_PATH_C)
test_data, test_gt = DCM_vol, GT_vol

# 본 스크립트는 V-net 학습 및 평가가 진행됩니다. 학습은 GPU로 진행되며 평가는 GPU 와 CPU로 진행됩니다.
# Train and evaluate the model
model = V_NET(data_shape=DATA_SHAPE, output=OUTPUT, dim=DIM, kernel_size=KERNEL_SIZE, attention=ATTENTION) # V-net 모델 호출
model.compile(optimizer=Adam(lr=LEARNING_RATE), loss=LOSS, metrics=METRICS)
model.summary()

model.fit(train_data, train_gt, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(test_data, test_gt), verbose=2)
model.evaluate(test_data, test_gt)

# save model at 'model_save'
model.save('/tmp/update_code/model_save/github/brainmask_putamen_AB_test_result.h5') # 모델의 가중치를 저장할 경로를 입력하시오. (가중치는 weight로 적으시면 됩니다.)



# save test pickle at 'test_result
test_result = model.predict(test_data)
result_path = '/tmp/update_code/test_result/github' # 결과를 저장할 경로를 입력하시오.
result_file = '/brainmask_caudate_C.pkl' # 결과 저장할 피클 이름을 작성하시오.
save_pickle(result_path + result_file, [test_result, test_data]) # prediction result, brainmask

#GPU use evalutation
result = []
time_data = []
for i in range(len(test_data)):
    start = time.time()
    result.append([i, ] + model.evaluate(np.expand_dims(test_data[i], axis=0), np.expand_dims(test_gt[i], axis=0)))
    end = time.time()
    total_time = np.array('%.4f' % (end-start),  dtype=np.float64)
    print(f"GPU time:{totaltime}")
    time_data.append(total_time.tolist())

#CPU use evalutation
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Use CPU
result = []
time_data = []
for i in range(len(test_data)):
    start = time.time()
    result.append([i, ] + model.evaluate(np.expand_dims(test_data[i], axis=0), np.expand_dims(test_gt[i], axis=0)))
    end = time.time()
    total_time = np.array('%.4f' % (end-start),  dtype=np.float64)
    print(f"CPU time:{totaltime}")
    time_data.append(total_time.tolist())
    
# save test result as excel
result_df = pd.DataFrame(result)  
time_df = pd.DataFrame(time_data)
df = pd.concat([result_df, time_df], axis=1)
excel_path = '/tmp/update_code/csv_files/github/' # test 결과를 저장할 엑셀 경로를 입력하시오.
df.to_excel(excel_path + 'brainmask_putamen_C.xlsx') # test 결과를 저장할 엑셀 이름을 입력하시오.

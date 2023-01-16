import pandas as pd
from setting import *
from pickle_load import *
import os
import pdb
import numpy as np
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" # Use GPU 
# Load data, DCM_AUG_vol.shape = (#,128,256,256,1)

[DCM_vol, GT_vol] = load_pickle(DATA_PATH_A)
train_data, train_gt = DCM_vol, GT_vol

[DCM_vol, GT_vol] = load_pickle(DATA_PATH_B)
train_data = np.append(train_data, DCM_vol, axis = 0)
train_gt = np.append(train_gt, DCM_vol, axis = 0)


[DCM_vol, GT_vol] = load_pickle(DATA_PATH_C)
test_data, test_gt = DCM_vol, GT_vol


# Train and evaluate the model
model = V_NET(data_shape=DATA_SHAPE, output=OUTPUT, dim=DIM, kernel_size=KERNEL_SIZE, attention=ATTENTION) # load model
model.compile(optimizer=Adam(lr=LEARNING_RATE), loss=LOSS, metrics=METRICS)
model.summary()

model.fit(train_data, train_gt, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(test_data, test_gt), verbose=2)
model.evaluate(test_data, test_gt)

# save model at 'model_save'
model.save('/tmp/update_code/model_save/github/brainmask_putamen_AB_test_result.h5') # model save



# save test pickle at 'test_result
test_result = model.predict(test_data)
result_path = '/tmp/update_code/test_result/github'
result_file = '/brainmask_caudate_C.pkl'
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
excel_path = '/tmp/update_code/csv_files/github/' 
df.to_excel(excel_path + 'brainmask_putamen_C.xlsx') #change

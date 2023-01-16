# Import necessary libraries
from metrics import *
from module import *
import os
from tensorflow.keras.optimizers import *
#from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model



'''
gpu = tf.config.experimental.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(gpu[0], True)
except RuntimeError as e:
    print(e)  # Error
'''
# Parameters
# About dataset
DATA_SHAPE = (128,256,256, 1) #(128, 256, 256, 1)  # (128, 128, 128, 1)
AUGMENTATION = False  # load augmented dataset if true

data_dir = '/tmp/output/input/' # 데이터 경로
dataset_1 = 'BRAINMASK_A_CAUDATE_3_128_256_256.pkl'  # 3fold 학습을 위한 fold1 데이터
dataset_2 = 'BRAINMASK_B_CAUDATE_3_128_256_256.pkl'  # 3fold 학습을 위한 fold2 데이터
dataset_3 = 'BRAINMASK_C_CAUDATE_3_128_256_256.pkl'  # 3fold 학습을 위한 fold3 데이터


#three-fold    
DATA_PATH_A = data_dir + dataset_1
DATA_PATH_B = data_dir + dataset_2
DATA_PATH_C = data_dir + dataset_3



# About model setup
CNN_MODEL = V_NET  # select one of: V_NET CNN 모델 선택
ATTENTION = False  # Attention hyper parameter 선택
OUTPUT = 1
DIM = 4 # or 8
KERNEL_SIZE = 5
BATCH_SIZE = 1  # or 4
EPOCHS = 100
VERBOSE = 1


# About model compile
LEARNING_RATE = 1e-4
LOSS = dice_loss  # tf.keras.losses.CategoricalCrossentropy()
METRICS = dice_score  # tf.keras.metrics.Accuracy()
TRAINING = True  # train the model if true, otherwise proceed prediction

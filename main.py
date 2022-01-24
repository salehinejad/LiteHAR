
import argparse
import numpy as np
import sys
sys.path.insert(0, "utils")
sys.path.insert(0, "models")
import dataloader
import rocket_rigid

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_path", required = True)
parser.add_argument("-cv", "--num_runs", type = int, default = 1)
parser.add_argument("-m", "--model", required = True)
parser.add_argument("-k", "--num_kernels", type = int, default = 10000)
parser.add_argument("-e", "--num_epochs", type = int, default = 100)
parser.add_argument("-g", "--gpu", type = int, default = 2)
args = parser.parse_args()

## Parameters Setup
gpu_id = ['0','1','2']
auto_save = False
N_epochs = args.num_epochs
N_cv = args.num_runs
num_kernels = args.num_kernels
classes = ['run','pickup','bed','fall','sitdown','standup','walk'] 
N_classes = len(classes)
partial_flag = False
rebuild_data = False
val_per = 0
tst_per = 0.2

batch_size = 8
lr = 0.001
lr_adaptive = True
decay_rate = 0.5
decay_step = 20 
pooling = 2
reinitialize_rocket = False # If True, will reinitialize rocket kernels for each CV

## Sampling
if pooling ==1:
    fval = '1k'
else:
    fval = str(int(1000/pooling))
frequency = fval+'hz'
print('Sampling Frequency is:',frequency)

## Prep
if lr_adaptive==False:
    decay_step = N_epochs+1 # decay_step more than number of epochs

model_name = args.model

X,Y = dataloader.preparedataRigRocket(args.input_path,classes,partial_flag,rebuild_data)
print('Data size from blob:',X.shape,Y.shape)


accuracy_collection = np.zeros((1,N_cv))
cm_collection = np.zeros((N_classes,N_classes,N_cv))
inf_time_collection = np.zeros((1,N_cv))
tr_time_collection = np.zeros((1,N_cv))
for cv_indx in range(N_cv):
    X_tr,X_val,X_ts,Y_tr,Y_val,Y_ts = dataloader.splitter(X,Y,val_per,tst_per)
    
    if model_name=='rigRocket':
        acc,cm, inf_time, tr_tim = rocket_rigid.main(X_tr,X_ts,Y_tr,Y_ts,num_kernels,N_classes,batch_size,N_epochs,gpu_id,partial_flag,lr,decay_rate,decay_step,pooling,frequency, N_cv,reinitialize_rocket,'rigRocket')           

    accuracy_collection[0,cv_indx] = acc
    cm_collection[:,:,cv_indx] = cm
    inf_time_collection[0,cv_indx] = inf_time
    tr_time_collection[0,cv_indx] = tr_tim    

    accuracy_collection = np.asarray(accuracy_collection)
print(model_name)
print(accuracy_collection)
print('Average Accuracy:',np.mean(accuracy_collection))
print(np.mean(cm_collection,axis=2))
print('Average CV Inference Time:',np.mean(inf_time_collection))
print('Average CV Training Time:',np.mean(tr_time_collection))
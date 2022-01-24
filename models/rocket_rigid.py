import os, pickle, time
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import RidgeClassifierCV
import matplotlib.pyplot as plt
from rocket_functions import generate_kernels, apply_kernels
from joblib import Parallel, delayed

def ridigd_training(X,Y):
    model = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True)
    model.fit(X, Y)
    return model

def scoring(model,X):
    prediction = model.predict(X)
    return prediction


def main(X_tr,X_ts,Y_tr,Y_ts,num_kernels,num_motions,batch_size,n_epochs,gpu_id,partial_flag,lr,decay_rate,decay_step,pooling,frequency,N_cv, reinitialize_rocket,model_):
    #### Sampling along time
    print('Sampling Frequency is:',frequency)
    if pooling>1:
        print('Sampling along time at window size of ',str(pooling), ' ...')
        X_tr = X_tr[:,::pooling,:]
        X_ts = X_ts[:,::pooling,:]
        T_Max = X_tr.shape[1]
    T_Max = X_tr.shape[1]
    print(T_Max)
    print(X_tr.shape)
    np.savetxt('sampleInput.txt',X_tr[0,:,:])
    st = time.time()

    X_tr,X_ts,Y_tr,Y_ts = rocketize(T_Max,num_kernels,X_tr,X_ts,frequency,N_cv,Y_tr,Y_ts,reinitialize_rocket)
    print(X_tr.shape,X_ts.shape) #  N,2xKernel, 90
    np.savetxt('sampleKernel.txt',X_tr[0,:,:])
    
    print('Parallel Training ...')
    Nsubc = X_tr.shape[2]
    models = Parallel(n_jobs=-2,backend="threading")(delayed(ridigd_training)(X_tr[:,:,m_],Y_tr) for m_ in tqdm(range(Nsubc)))
    tr_time = time.time() - st

    # Testing
    print('Parallel Testing ...')
    top_collection = []
    disagrees_subcarries_collect = []
    disagrees_histogram = np.zeros((1,Nsubc))
    time_collect = 0
    for s_indx in range(X_ts.shape[0]): # for each test sample
        st = time.time()
        predictions = Parallel(n_jobs=1,backend="threading")(delayed(scoring)(models[m_],np.expand_dims(X_ts[s_indx,:,m_],axis=0)) for m_ in range(Nsubc))
        time_collect+=(time.time()-st)
        (unique, counts) = np.unique(predictions, return_counts=True)
        top_collection.append([unique[np.argmax(counts)],Y_ts[s_indx]]) # prediction Target 
        disagrees_binary = predictions!=Y_ts[s_indx]
        disagrees_subcarries = np.where(disagrees_binary==True)[0]
        disagrees_subcarries_collect.append(disagrees_subcarries)
        for i in disagrees_subcarries: # histogram of disagrees update
            disagrees_histogram [0,i]+=1 

    print('Prediction vs. Target:', top_collection)
    print('Disagreed subcarriers histogram:',disagrees_histogram/X_ts.shape[0])
    top_collection = np.asarray(top_collection)
    acc = (np.sum(top_collection[:,0]==top_collection[:,1]))/X_ts.shape[0]
    print('Accuracy is:', acc)
    print('Avg. Inferene Time (full,per sample):',time_collect,time_collect/X_ts.shape[0])
    print('Training Time (full,per sample):',tr_time,tr_time/X_tr.shape[0])
    cm = confusion_matrix(top_collection[:,1], top_collection[:,0]) # Target prediction

    return acc,cm,time_collect/X_ts.shape[0],tr_time/X_tr.shape[0]



def rocketize(T_Max,num_kernels,X_tr,X_ts,frequency,N_cv,Y_tr,Y_ts,reinitialize_rocket):
    if os.path.isfile('blob/'+frequency+'rocket'+'/X_tr_RockOnly.pkl') and reinitialize_rocket==False:
        print('Loading pickled data...') 
        with open('blob/'+frequency+'rocket'+'/X_tr_RockOnly.pkl', 'rb') as f:
            X_tr = pickle.load(f)
        with open('blob/'+frequency+'rocket'+'/X_tst_RockOnly.pkl', 'rb') as f:
            X_ts = pickle.load(f)           
        with open('blob/'+frequency+'rocket'+'/T_MAX_RockOnly.pkl', 'rb') as f:
            T_Max = pickle.load(f)  
    else:
        print("Building the rocket  ...")
        print('Computing Rocket of training samples...')

        input_length = T_Max
        kernels = generate_kernels(input_length, num_kernels)

        print('Rocketizing trianing data ...')
        X_tr_rock = np.zeros((X_tr.shape[0],X_tr.shape[2],2*num_kernels)) 
        for sample_indx in tqdm(range(X_tr.shape[0])): # for each sample
            input_sample = np.swapaxes(X_tr[sample_indx,:,:],0,1)
            X_tr_rock[sample_indx,:,:] = apply_kernels(input_sample, kernels) # out: (N, 180, 2*N_Kernels)

        print('Rocketizing testing data ...')
        X_ts_rock = np.zeros((X_ts.shape[0],X_ts.shape[2],2*num_kernels)) 
        for sample_indx in tqdm(range(X_ts.shape[0])): # for each sample
            input_sample = np.swapaxes(X_ts[sample_indx,:,:],0,1)
            X_ts_rock[sample_indx,:,:] = apply_kernels(input_sample, kernels) # out: (N, 180, 2*N_Kernels)

        X_tr = np.swapaxes(X_tr_rock,1,2)
        X_ts = np.swapaxes(X_ts_rock,1,2)

        # Makedir for frequency
        if not os.path.exists('blob/'+frequency+'rocket'):
            os.makedirs('blob/'+frequency+'rocket')
        
        print('Saving the files in the blob ...')
        with open('blob/'+frequency+'rocket'+'/X_tr_RockOnly.pkl', 'wb') as f:
            pickle.dump(X_tr, f,protocol=4)
        with open('blob/'+frequency+'rocket'+'/X_tst_RockOnly.pkl', 'wb') as f:
            pickle.dump(X_ts, f,protocol=4)  
        with open('blob/'+frequency+'rocket'+'/T_MAX_RockOnly.pkl', 'wb') as f:
            pickle.dump(T_Max, f,protocol=4)


    ## Shuffling for CV
    all_data = np.vstack((X_tr,X_ts))
    all_labels = np.vstack((Y_tr,Y_ts))
    all_data = all_data[:,:,:30]
    ## Remove certain classes : pick up: index 1
    # class_1_keep_indx = [indx for indx in range(all_labels.shape[0]) if all_labels[indx,1]!=1] 
    # all_data = all_data[class_1_keep_indx,:,:]
    # all_labels = all_labels[class_1_keep_indx,:]


    N_samples = all_data.shape[0]
    indx_ = np.arange(N_samples)
    np.random.shuffle(indx_)
    N_TS = int(np.ceil(0.2*N_samples))
    ts_range = indx_[:N_TS]
    tr_range = indx_[N_TS:]
    X_tr = all_data[tr_range,:,:]
    X_ts = all_data[ts_range,:,:]
    Y_tr = all_labels[tr_range,:]
    Y_ts = all_labels[ts_range,:]
    Y_tr = [np.where(y==1)[0][0] for y in Y_tr] # numeric labels
    Y_ts = [np.where(y==1)[0][0] for y in Y_ts]
    return X_tr,X_ts,Y_tr,Y_ts    

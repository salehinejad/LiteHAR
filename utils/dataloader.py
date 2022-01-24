
import numpy as np
import scipy.io as scio
import os, pickle,sys
from tqdm import tqdm
from joblib import Parallel, delayed
import math, pickle
from joblib import Parallel, delayed

def read_files(input_path,classes):   
    print('Reading files...') 
    input_file_names = [i for i in os.listdir(input_path) if i.endswith('.csv') and i.startswith('input')]
    annotation_file_names = [i for i in os.listdir(input_path) if i.endswith('.csv') and i.startswith('annotation')]
    
    clean_annotation_file_names = ['_'.join(i.split('_')[1:]) for i in annotation_file_names]
    
    files_matching = []
    # Match files with non-name
    for i in input_file_names:
        cl_i = '_'.join(i.split('_')[1:])
        if cl_i in clean_annotation_file_names:
            files_matching.append([i,'annotation_'+cl_i])
    # Match files with names
    for i in input_file_names:
        cl_i = '_'.join(i.split('_')[2:])
        if 'siamak' in cl_i or 'sankalp' in cl_i:
            if cl_i in clean_annotation_file_names:
                files_matching.append([i,'annotation_'+cl_i])
    # listing inputs and annotation and class
    classes_stat_dict = {el:[] for el in classes} # dictionary of classes and index of each sample
    files_matching_wclasses = []
    for indx, i in enumerate(files_matching):
        for k in classes:
            if k in i[0].split('_'):                
                one_hot_vect = one_hot(k,classes) # one-hot encoding the class
                files_matching_wclasses.append([i[0],i[1],k,one_hot_vect]) # input_name, annotation name, class name, onehot class vector
                classes_stat_dict[k].append(indx)
                break
    print('Number of samples:',len(files_matching_wclasses))
    print('Number of samples per class:')
    for key in classes_stat_dict.keys():
        print(key,len(classes_stat_dict[key]))

    return files_matching_wclasses, classes_stat_dict

def one_hot(k, classes):
    indx = classes.index(k)
    one_hot_vect = len(classes)*[0]
    one_hot_vect[indx] = 1
    return one_hot_vect

def zero_padding(X, T, Y, N_classes):
    print('Zero-padding...')
    T_Max = np.max(T)
    N_subcarriers = X[0].shape[1]
    N_samples = X.shape[0]
    print('T_Max:',T_Max,'N Subcarriers:',N_subcarriers,'Number of samples:',N_samples)   
    X_padded = np.zeros((N_samples,T_Max,N_subcarriers))
    Y_ = np.zeros((N_samples,N_classes))
    for i in tqdm(range(N_samples)):
        X_padded[i,:X[i].shape[0],:] = X[i]
        Y_[i,:] = Y[i]
    return X_padded, Y_, T_Max

def normalize_data(X):
    N_samples = X.shape[0]
    T_Max = X.shape[1]
    
    min_vec = np.min(X,axis=(0,1))
    min_vec = np.expand_dims(min_vec,axis=(0,1))
    tiled_min = np.tile(min_vec,(N_samples,T_Max,1))
    X_ = X - tiled_min
    max_vec = np.max(X_,axis=(0,1))
    max_vec = np.expand_dims(max_vec,axis=(0,1))    
    tiled_max = np.tile(max_vec,(N_samples,T_Max,1))
    X = X_/tiled_max
    return  X


def preparedata(input_path,classes,partial_flag,rebuild_data):
    blob_type = 'all'
    if partial_flag:
        blob_type = 'toy'

    if rebuild_data==False:
        if os.path.isfile('blob/X_'+blob_type+'.pkl') and os.path.isfile('blob/Y_'+blob_type+'.pkl'):
            print('Loading from blob...')
            with open('blob/X_'+blob_type+'.pkl', 'rb') as f:
                X = pickle.load(f)
            with open('blob/Y_'+blob_type+'.pkl', 'rb') as f:
                Y = pickle.load(f)
        else:
            print('Pickle files do not exist.')
            sys.exit()
    elif rebuild_data==True:
        N_classes = len(classes)
        ## Read files and match inputs and annotations and classes
        files_matching_wclasses, classes_stat_dict = read_files(input_path, classes)
        ## load csvs
        X, Y, T = load_csv(input_path,files_matching_wclasses,partial_flag) # X.one-hot labels, length of each sample
        ## zero-padding
        X, Y, T_Max = zero_padding(X,T,Y,N_classes) # zero-padded X, max length of signal
        ## Normalization
        X = normalize_data(X)
        ## Saving data
        with open('blob/X_'+blob_type+'.pkl', 'wb') as f:
            pickle.dump(X, f, protocol=4)
        with open('blob/Y_'+blob_type+'.pkl', 'wb') as f:
            pickle.dump(Y, f, protocol=4)

    return X,Y


def preparedataRigRocket(input_path,classes,partial_flag,rebuild_data):
    blob_type = 'all'
    if partial_flag:
        blob_type = 'toy'

    if os.path.isfile('blob/1khzrocket/X_'+blob_type+'RigRocket.pkl') and os.path.isfile('blob/1khzrocket/Y_'+blob_type+'RigRocket.pkl'):
        print('Loading from blob...')
        with open('blob/1khzrocket/X_'+blob_type+'RigRocket.pkl', 'rb') as f:
            X = pickle.load(f)
        with open('blob/1khzrocket/Y_'+blob_type+'RigRocket.pkl', 'rb') as f:
            Y = pickle.load(f)
    else:
        print('Pickle files do not exist. Building it')

        N_classes = len(classes)
        ## Read files and match inputs and annotations and classes
        files_matching_wclasses, classes_stat_dict = read_files(input_path, classes)
        ## load csvs
        X, Y, T = load_csv(input_path,files_matching_wclasses,partial_flag) # X.one-hot labels, length of each sample
        ## zero-padding
        X, Y, T_Max = zero_padding(X,T,Y,N_classes) # zero-padded X, max length of signal
        ## Normalization
        # X = normalize_data(X)
        ## Saving data
        with open('blob/1khzrocket/X_'+blob_type+'RigRocket.pkl', 'wb') as f:
            pickle.dump(X, f, protocol=4)
        with open('blob/1khzrocket/Y_'+blob_type+'RigRocket.pkl', 'wb') as f:
            pickle.dump(Y, f, protocol=4)

    return X,Y



def max_pooling(X):
    T = X.shape[0] # T x 90 
    # print(T,X.shape)
    Xx = np.expand_dims(X,axis=1)
    # print(Xx.shape)

    if T%2==1:
        T_temp = T-1
        Xx = np.reshape(Xx[:T_temp,:],(int(T_temp/2),2,X.shape[1]))
        new_X = np.zeros((int(T_temp/2)+1,X.shape[1]))
        dd = np.max(Xx,axis=1)
        new_X[:-1,:] = dd
        new_X[-1,:] = X[-1,:]
    else: 
        T_temp = T
        Xx = np.reshape(Xx,(int(T_temp/2),2,X.shape[1]))
        new_X = np.zeros((int(T_temp/2),X.shape[1]))
        dd = np.max(Xx,axis=1)
        new_X = dd

    return new_X



def parallel_read(input_path,i,files_matching_wclasses):
    file_path_x = input_path+files_matching_wclasses[i][0]
    file_path_y = input_path+files_matching_wclasses[i][1]
    Y = np.asarray(files_matching_wclasses[i][3])
    x = np.loadtxt(file_path_x,delimiter=',',dtype='float')
    y = np.loadtxt(file_path_y,delimiter=',',dtype='str')
    yy = np.where(y==files_matching_wclasses[i][2])
    start_ = int(yy[0][0])
    finish_ = int(yy[0][-1])
    
    X = x[start_:finish_+1,1:91] # 90: amplitiude only; for all replace with :

    signal_len = X.shape[0]
    return X,Y,signal_len

def load_csv(input_path,files_matching_wclasses, partial_flag):
    n_samples = len(files_matching_wclasses)
    if partial_flag:
        n_samples = 40
    results = Parallel(n_jobs=-2,backend="threading")(delayed(parallel_read)(input_path,i,files_matching_wclasses) for i in tqdm(range(n_samples)))
    results = np.asarray(results)
    X = results[:,0]
    Y = results[:,1]
    signal_lengths = results[:,2]
    return X, Y, signal_lengths


def splitter(X,Y,val_per,tst_per):
    print('Splitting train and test data ...')
    N_samples = X.shape[0]
    indxes = np.arange(N_samples)
    np.random.shuffle(indxes)
    val_range = int(np.ceil(val_per*N_samples))
    ts_range = int(np.ceil(tst_per*N_samples))
    tr_range = N_samples - (val_range+ts_range)

    X_tr = X[0:tr_range,:,:]
    X_val = X[tr_range:tr_range+val_range,:,:]
    X_ts = X[tr_range+val_range:,:,:]
    
    Y_tr = Y[0:tr_range,:]
    Y_val = Y[tr_range:tr_range+val_range,:]
    Y_ts = Y[tr_range+val_range:,:]
    print('Number of Training samples:',X_tr.shape[0])
    print('Number of Validation samples:',X_val.shape[0])
    print('Number of Test samples:',X_ts.shape[0])

    return X_tr,X_val,X_ts,Y_tr,Y_val,Y_ts






import numpy as np
import pandas as pd
from scipy.io import loadmat
from matplotlib import pyplot as plt
from itertools import permutations 
from time import time
from subspace_alignment import subspace, knn
from kl_divergence import KLdivergence

def load_data(dataset_name):
    #caffenet
    if dataset_name == 'caffe':
        webcam = loadmat("data/CaffeNet4096/webcam.mat")
        dslr = loadmat("data/CaffeNet4096/dslr.mat")
        amazon = loadmat("data/CaffeNet4096/amazon.mat")

    #googlenet
    elif dataset_name == 'googlenet':
        webcam = loadmat("data/GoogleNet1024/webcam.mat")
        dslr = loadmat("data/GoogleNet1024/dslr.mat")
        amazon = loadmat("data/GoogleNet1024/amazon.mat")

    #surf
    elif dataset_name == 'surf':
        webcam = loadmat("data/surf/webcam.mat")
        dslr = loadmat("data/surf/dslr.mat")
        amazon = loadmat("data/surf/amazon.mat")

    return webcam, dslr, amazon


def mean_cov(df, display_cov = False):
    # Calculate mean vector for source dataset
    print('Mean Vector for Source data: \n', df.mean(), '\n')
    
    if display_cov == True:
        # Calculate covariance matrix for source dataset
        print('Covariance matrix for Source data: \n')
        print(df.cov())

def scatter(X_s,y_s):
    X_s ,y_s = np.asarray(X_s), np.asarray(y_s)
    for class_value in range(10):
        # get row indexes for samples with this class
        row_ix = np.where(y_s == class_value)
        # create scatter of these samples
        plt.scatter(X_s[row_ix, 0], X_s[row_ix, 1])
    # show the plot
    plt.title('Source Data')
    labels = [('Class {}').format(i) for i in range(1,11)]
    plt.legend(labels)
    plt.show()

if __name__ == "__main__":
    dataset = input('Enter the name of the dataset (caffe/googlenet/surf): ')
    print('Dataset: ',dataset)
    webcam, dslr, amazon = load_data(dataset)

    if dataset == 'surf':
        df_webcam = pd.DataFrame(np.hstack((webcam['fts'], webcam['labels'])))
        df_dslr = pd.DataFrame(np.hstack((dslr['fts'], dslr['labels'])))
        df_amazon = pd.DataFrame(np.hstack((amazon['fts'], amazon['labels'])))
    else:
        df_webcam = pd.DataFrame(np.hstack((webcam['fts'], webcam['labels'].T)))
        df_dslr = pd.DataFrame(np.hstack((dslr['fts'], dslr['labels'].T)))
        df_amazon = pd.DataFrame(np.hstack((amazon['fts'], amazon['labels'].T)))
    
    n = len(df_webcam.keys()) - 1

    mean_cov(df_webcam.iloc[:,:n], display_cov=False)
    # scatter(df_webcam.iloc[:,:n], df_webcam.iloc[:,-1])

    mean_cov(df_dslr.iloc[:,:n])
    # scatter(df_dslr.iloc[:,:n], df_dslr.iloc[:,-1])

    mean_cov(df_amazon.iloc[:,:n])
    # scatter(df_webcam.iloc[:,:n], df_webcam.iloc[:,-1])

    df_list = [df_webcam,df_dslr,df_amazon]
    df_iter = list(permutations(df_list,2))

    # list1 = ["webcam","dslr","amazon"]
    list1 = ["webcam","dslr"]
    index = list(permutations(list1,2))
    methods = ["subspace","without subspace"]
    caffe_timedf = pd.DataFrame(index=index,columns=methods)
    caffe_accdf = pd.DataFrame(index=index,columns=methods)

    # index -> 2 list[[webcam,dslr],[dslr, webcam]]
    # seeds = [20,40,42]

    seeds = [20]

    for s in seeds:
        print('Random Seed: ', s,'\n')
        for i in range(len(index)):
        # Just to check for source: webcam, target: dslr case do as below
        # for i in range(2):
            t0 = time()
            source = (df_iter[i][0]).sample(frac = 0.5, random_state = s)
    #         source = df_iter[i][0]
            print(('Source: {} Target: {}').format(index[i][0],index[i][1]))
            sub = subspace("caffe",source,df_iter[i][1],13,index[i], class_kl= False, plot = True)
            caffe_accdf.iloc[i,0] = sub.fit_predict(plot = False)
            t1 = time()
            caffe_timedf.iloc[i,0] = t1-t0
            t2 = time()
            caffe_accdf.iloc[i,1] = knn("caffe",source,df_iter[i][1], index[i], seed = s, plot = True)
            t3 = time()
            caffe_timedf.iloc[i,1] = t3-t2
            
        print(caffe_accdf)
        print(('Accuracy for (webcam,dslr) \n with SA = {} \n w/o SA = {} \n').format(
        caffe_accdf['subspace'][('webcam', 'dslr')],
        caffe_accdf['without subspace'][('webcam', 'dslr')]))
        
        print(('Accuracy for (dslr,webcam) \n with SA = {} \n w/o SA = {} \n').format(
        caffe_accdf['subspace'][('dslr', 'webcam')],
        caffe_accdf['without subspace'][('dslr', 'webcam')]))
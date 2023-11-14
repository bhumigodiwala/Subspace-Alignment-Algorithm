import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions
from kl_divergence import KLdivergence

class subspace:
    def __init__(self,dataset,S,T,d,seed, plot = False):
        self.dataset = dataset
        self.S = S
        self.T = T
        self.d = d
        self.seed = seed
        
    def pca(self,x, n_components):
        cov = np.cov(x , rowvar = False)
        eigen_values , eigen_vectors = np.linalg.eigh(cov)
        sorted_index = np.argsort(eigen_values)[::-1]
        sorted_eigenvalue = eigen_values[sorted_index]
        sorted_eigenvectors = eigen_vectors[:,sorted_index]
        return sorted_eigenvectors[:,0:n_components]
    
    def fit_predict(self,plot):
#         KL Divergence b/w Source and Target data SA
        print(('KL Divergence b/w orig S and T data (SA) = {} \n').format(KLdivergence(self.S, self.T)))
        
        #normalising the data
        scale = StandardScaler()
        n = len(self.S.keys()) - 1
#         print('SA random Seed:', self.seed)
#         S = self.S.sample(random_state = self.seed, frac = 0.5)
        x_S = self.S.iloc[:,:n]
#         print(x_S)
        x_T = self.T.iloc[:,:n]
        x_T = scale.fit_transform(x_T)
        x_S = scale.fit_transform(x_S)
        
        y_S = self.S.iloc[:,-1]
        y_T = self.T.iloc[:,-1]
        
        print('Class wise KL Divergence b/w orig S & T data before SA: \n')
        for class_value in range(1,11):
                # get row indexes for samples with this class
                
                S_row_ix = np.where(y_S == class_value)
                T_row_ix = np.where(y_T == class_value)
#                 print(('Class value = {} and S_row_ix = {} and T_row_ix = {}').format(class_value, S_row_ix, T_row_ix))
#                 print(x_S[row_ix])
                
                class_kl = KLdivergence(x_S[S_row_ix], x_T[T_row_ix])
                print(('KL divergence for Class {} = {} \n').format(class_value, class_kl))
        
        
        #pca and picking the top eigen vectors
        if plot == True:
            xs = self.pca(x_S,2)
            xt = self.pca(x_T,2)
        else:
            xs = self.pca(x_S,self.d)
            xt = self.pca(x_T,self.d)
            
                    
        xa = np.dot(np.dot(xs,xs.T),xt)
        sa_pca = np.dot(x_S,xs)
        
        
#         print(self.S.shape, sa_pca.shape)

        # KL Divergence b/w Original Source and Source data SA after PCA
#         print(('KL Divergence b/w Original Source and Source data (SA) after PCA = {} \n').format(KLdivergence(self.S, sa_pca)))

        sa = np.dot(x_S,xa)
        
        st = np.dot(x_T,xt)
        
        test_s = np.dot(x_S,xs)
#         KL Divergence b/w Original Source and Target data SA after PCA
        print(('KL Divergence b/w Original Source and target data (SA) after PCA = {} \n').format(KLdivergence(test_s, st)))
        
        
        # KL Divergence b/w Source Aligned data and target data SA after PCA
        print(('KL Divergence b/w Source Aligned data (SA) and target data after PCA = {} \n').format(KLdivergence(sa, st)))
        
        
        
        print('Class wise KL Divergence b/w orig S & T data after SA & PCA: \n')
        for class_value in range(1,11):
            # get row indexes for samples with this class

            S_row_ix = np.where(y_S == class_value)
            T_row_ix = np.where(y_T == class_value)
#                 print(('Class value = {} and S_row_ix = {} and T_row_ix = {}').format(class_value, S_row_ix, T_row_ix))
#                 print(x_S[row_ix])

            class_kl_sa = KLdivergence(sa[S_row_ix], st[T_row_ix])
            print(('KL divergence for Class {} = {} \n').format(class_value, class_kl_sa))

        
        if plot == True:
            # create scatter plot for Source dataset after PCA before SA
            for class_value in range(1,11):
                # get row indexes for samples with this class
                row_ix = np.where(y_S == class_value)
                # create scatter of these samples
                plt.scatter(sa_pca[row_ix, 0], sa_pca[row_ix, 1])
            # show the plot
            plt.title('Source dataset after PCA before SA')
            plt.legend()
            plt.show()

            # create scatter plot for Subspace aligned dataset
            for class_value in range(1,11):
                # get row indexes for samples with this class
                row_ix = np.where(y_S == class_value)
                # create scatter of these samples
                plt.scatter(sa[row_ix, 0], sa[row_ix, 1])
            # show the plot
            plt.title('Subspace Aligned Data')
            plt.legend(['Class 0', 'Class 1'])
            plt.show()
        
        #knn classifier
        knn = KNeighborsClassifier(4)
        
        accuracy = []
        for j in range(5):
            knn.fit(sa,y_S)
            labels = knn.predict(st)
            acc = accuracy_score(labels,y_T)
            accuracy.append(acc)
        
        if plot == True:
            fig, ax = plt.subplots()
            title = ('Decision surface with SA')


            fig = plot_decision_regions(sa, np.asarray(y_S).astype(np.int_),knn,legend=2)
            plt.title('Source data with Source ground truth labels WITH SUBSPACE')
            plt.show()

            fig = plot_decision_regions(st, np.asarray(y_T).astype(np.int_),knn,legend=2)
            plt.title('Target data with Target ground truth labels WITH SUBSPACE')
            plt.show()

            ax.set_ylabel('y label')
            ax.set_xlabel('x label')
            ax.set_xticks(())
            ax.set_yticks(())
            plt.show()
#         print('SA Acc:',accuracy)
        
        acc_mean = np.mean(accuracy)
        acc_std = np.std(accuracy)
        acc_min = min(accuracy)
        acc_max = max(accuracy)
        
        return [acc_mean, acc_std, acc_min, acc_max]

def knn(dataset,S,T,seed, plot = False):
#         KL Divergence b/w Source and Target data
#     print(('KL Divergence b/w Source and Target data (w/o SA) = {}').format(KLdivergence(S, T)))
    dataset = dataset
    n = len(S.keys()) - 1
#     print('W/O SA random Seed:', seed)
#     S = S.sample(random_state = seed,frac = 0.5)
    x_S = S.iloc[:,:n]
#     print(x_S)
    x_T = T.iloc[:,:n]
    y_S = S.iloc[:,-1]
    y_T = T.iloc[:,-1]
    scale = StandardScaler()
    x_S = scale.fit_transform(x_S)
    x_T = scale.fit_transform(x_T)
    
    if plot == True:
        pca = PCA(n_components=2)
        x_S_pca = pca.fit_transform(x_S)
        x_T_pca = pca.fit_transform(x_T)
    #     print('pca_x_s',x_S_pca.shape)

         # create scatter plot for Source dataset after PCA supervised learning case
        for class_value in range(1,11):
            # get row indexes for samples with this class
            row_ix = np.where(y_S == class_value)
            # create scatter of these samples
            plt.scatter(x_S_pca[row_ix, 0], x_S_pca[row_ix, 1])
        # show the plot
        plt.title('Source dataset after PCA in SL')
        plt.legend(['Class 0', 'Class 1'])
        plt.show()
    
        knn = KNeighborsClassifier(1)
        
        accuracy = []
        for j in range(5):
            knn.fit(x_S_pca,y_S)
            labels = knn.predict(x_T_pca)
            acc = accuracy_score(labels,y_T)
            accuracy.append(acc)
    
        fig, ax = plt.subplots()
        title = ('Decision surface without SA')

        fig = plot_decision_regions(x_S_pca, np.asarray(y_S).astype(np.int_),knn,legend=2)
        plt.title('Source data with Source ground truth labels WITHOUT SUBSPACE')
        plt.show()

        fig = plot_decision_regions(x_T_pca, np.asarray(y_T).astype(np.int_),knn,legend=2)
        plt.title('Target data with Target ground truth labels WITHOUT SUBSPACE')
        plt.show()

        ax.set_ylabel('y label')
        ax.set_xlabel('x label')
        ax.set_xticks(())
        ax.set_yticks(())
        plt.show()
    
    else:
        print('Note: We do not apply PCA in classification w/o SA \n')
        knn = KNeighborsClassifier(4)
        accuracy = []
        for j in range(5):
            knn.fit(x_S,y_S)
            labels = knn.predict(x_T)
            acc = accuracy_score(labels,y_T)
            accuracy.append(acc)
#     print('Acc:',accuracy)
    acc_mean = np.mean(accuracy)
    acc_std = np.std(accuracy)
    acc_min = min(accuracy)
    acc_max = max(accuracy)
#     print(('acc mean std min max = {}, {}, {}, {}').format(acc_mean, acc_std, acc_min, acc_max))
    return [acc_mean, acc_std, acc_min, acc_max]

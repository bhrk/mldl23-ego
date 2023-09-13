import pickle
from utils.logger import logger
import torch.nn.parallel
import torch.optim
import torch
from utils.loaders import EpicKitchensDataset
from utils.args import args
from utils.utils import pformat_dict
import utils
import numpy as np
import os
import models as model_list
import tasks
import time
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kmeans_pytorch import kmeans # import kmeans-pytorch using pip install kmeans-pytorch on env; then install tqdm 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
import seaborn as sns

modalities = None
np.random.seed(13696641)
torch.manual_seed(13696641)
X=[]
X_torch=[]
true_labels=[]
def init():
    global X,X_torch,true_labels
    saved_feat_path="./saved_features/saved_feat_D3_train.pkl"
    annotations_path="./train_val/D3_train.pkl" 
    file=open(saved_feat_path,'rb')
    feat=pickle.load(file)
    annotations=pickle.load(open(annotations_path,'rb'))
    true_labels=annotations['verb_class']
    x=[]
    i=0
    for record in feat['features']:
        x.append(record['features_RGB'][2])
    X=np.array(x)
    X_torch=torch.from_numpy(X)
    print(f"size of X features: {X.shape}")
def main():
    # global variables among training functions
   init()
   performKmeans(X,X_torch,8,true_labels)

def performKmeans(X,X_torch,num_clusters,true_labels):

    # kmeans with pytorch
    start=time.time()
    #num_clusters=8
    cluster_ids_x, cluster_centers = kmeans(
        X=X_torch, num_clusters=num_clusters, distance='euclidean', device=torch.device('cpu')
    )
    end=time.time()
    print(f"time required for pytorch kmeans: {end-start}")

    # kmeans with scikitlearn
    start=time.time()
    kmeans_SL = KMeans(n_clusters=num_clusters, random_state=0, n_init='auto').fit(X)
    end=time.time()
    print(f"time required for scikit_learn kmeans: {end-start}")
    
    # silhouette calculation
    silh=silhouette_score(X,kmeans_SL.labels_)
    silh_torch=silhouette_score(X,cluster_ids_x)
    print(f"silhouette for kmeans_sklearn: {silh}")
    print(f"silhouette for kmeans_torch: {silh_torch}")

    ############ PCA & PLOT POINTS #################
    random_state=0
    # with PCA
    model=make_pipeline(StandardScaler(),PCA(2,random_state=random_state))
    X_2feat=model.fit_transform(X)
    kmeans_SL = KMeans(n_clusters=num_clusters, random_state=0, n_init='auto').fit(X_2feat)
    s=silhouette_score(X_2feat,kmeans_SL.labels_)
    print(f"silhouette for kmeans_sklearn with PCA=2: {s} ")

    fig, ax = plt.subplots(2)
    sns.scatterplot(x=X_2feat[:,0],y=X_2feat[:,1],c=kmeans_SL.labels_,ax=ax[0])
    # plt.scatter(x=X_2feat[:,0],y=X_2feat[:,1])
    ax[0].set_title(f"Features distributions - PCA 2 components - silh: {s}")
    ax[0].plot()
    ax[0].scatter(
    kmeans_SL.cluster_centers_[:, 0],
    kmeans_SL.cluster_centers_[:, 1],
    marker="x",
    s=169,
    linewidths=3,
    zorder=10,
    color='r'
    )
    sns.scatterplot(x=X_2feat[:,0],y=X_2feat[:,1],c=true_labels,ax=ax[1])
    # plt.scatter(x=X_2feat[:,0],y=X_2feat[:,1])
    ax[1].set_title(f"Features distributions - true labels (focus on groups)")
    ax[1].plot()
    print()

#try with average clip
#try with DB Scan & GMM mixtures
def performDB_Scan():
    return
def performGMM():
    return

if __name__ == '__main__':
    main()
#
import argparse
import copy
import decimal
import pickle
import random
import math

import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.decomposition import TruncatedSVD
import sklearn.linear_model as sklearn_linear_model
import sklearn.preprocessing as sklearn_preprocessing
import sklearn.svm as sklearn_svm
from sklearn.svm import SVR
from scipy import spatial

from dataset_container import *
from sampling_methods import *
import utils

import torch
import torch.nn as nn

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from sklearn.cluster import KMeans, BisectingKMeans
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from sklearn.cluster import DBSCAN

import hdbscan

import torch.nn.functional as F

from statistics import mean

def mean_abs_dev(arr):
    mea = np.mean(arr)
    abs_dev = [abs(x - mea) for x in arr]
    return np.mean(abs_dev)

def assign_weight(cluster_idx, k, x, y):
    dicts = [[] for i in range(k)]
    for i in range(len(x)):
        dicts[cluster_idx[i]].append(y[i])
    sum = [np.sum(dicts[i]) for i in range(k)]
    average = []
    for i in range(k):
        if len(dicts[i]):
            average.append(sum[i] / len(dicts[i]))
        else:
            average.append(0)
    return dicts, average

def loss_function_vae(x_hat, x, mu, log_var, criterion):
    MSE = criterion(x_hat, x)
    KLD = 0.5 * torch.sum(torch.exp(log_var) + torch.pow(mu, 2) - 1. - log_var)
    loss = MSE + KLD
    return loss, MSE, KLD

class VAE(nn.Module):
    def __init__(self, input_dim=784, h_dim=400, z_dim=20):
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.fc1 = nn.Linear(input_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)

        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, input_dim)

    def forward(self, x):
        mu, log_var = self.encode(x)
        sampled_z = self.reparameterization(mu, log_var)
        x_hat = self.decode(sampled_z)
        return x_hat, mu, log_var

    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc2(h)
        log_var = self.fc3(h)
        return mu, log_var

    def reparameterization(self, mu, log_var):
        sigma = torch.exp(log_var * 0.5)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

    def decode(self, z):
        h = F.relu(self.fc4(z))
        x_hat = torch.sigmoid(self.fc5(h))
        return x_hat

class FNN(nn.Module):
  def __init__(self, input_size=36, hidden_size=18, output_size=1, prob_dropout=0):
    super(FNN, self).__init__()
    self.predict = nn.Sequential(
        nn.Linear(input_size, hidden_size), nn.PReLU(), nn.Dropout(prob_dropout),
        nn.Linear(hidden_size, hidden_size), nn.PReLU(), nn.Dropout(prob_dropout),
        nn.Linear(hidden_size, hidden_size), nn.PReLU(), nn.Dropout(prob_dropout),
        nn.Linear(hidden_size, output_size)
    )

  def forward(self, x):
    x = self.predict(x)
    return x

def parse_input():
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset", type=str, choices={"real_1", "real_2", "real_3"},
                        help="Name of the dataset.")
    parser.add_argument("model", type=str, choices={"svm"},
                        help="Name of the model")
    parser.add_argument("active_learning", type=str, choices={"dist_to_boundary", "margin", "qbc", "uniform"},
                        help="Name of the active learning algorithm")

    parser.add_argument("--dataset_path", type=str, default="./datasets/synthetic",
                        help="Path where datasets will be loaded/saved.")

    parser.add_argument("--init_num_labeled", type=float, default=0.05,
                        help="Initial number of labeled data in active learning. Float values less than 1.0 indicate a"
                             " percentage.")
    parser.add_argument("--budget_round", type=float, default=0.05,
                        help="Budget in each active learning round. Float values less than 1.0 indicate a percentage.")
    parser.add_argument("--active_epoch", type=int, default=9,
                        help="Number of rounds to run the active learning algorithm.")

    parser.add_argument("--boundaries", nargs='*',
                        help="The sequence of values of decision boundaries that a model is expected to learn.")

    parser.add_argument("--random_seed", type=int, default=0)

    # arguments for QBC
    parser.add_argument("--num_models", type=int, default=10)
    parser.add_argument("--num_samples", type=float, default=0.8)
    
    # arguments for retrain
    parser.add_argument("--retrain",type=bool,default=0)

    args = parser.parse_args()
    return args

class Arguments:
    def __init__(self):
        self.random_seed = 500

        self.dataset = "real_4"
        self.split_mode = 0
        self.dataset_path = "./datasets/real"

        self.model = "logistic_regression"  # {"logistic_regression", "svm"}

        self.if_hybrid = 1  # {1,0} 1 refers to two types of descriptors 0 otherwise
        self.representationA = "pka_bde01"
        self.representationB = "morgan_fp"  # {"morgan_fp", "one_hot", "morgan_pka","ohe_pka"}

        self.representation = "Mordred"  # {"morgan_fp", "one_hot", "Mordred", "morgan_pka", "ohe_pka"}
        self.representation_dim = 2048  # used for morgan fingerprint or morgan_pka
        self.reduce_dim = 'pca' # {pca, vae}

        self.representations = ['morgan_fp', 'pka_bde01', 'rxnfp']
        self.reduce_method = ['pca', 'pca', 'pca']
        self.pca_components = [128, 6, 100]

        self.vae_hide_dim = 40
        self.vae_components = 10
        self.vae_lr = 0.01
        self.vae_epoch = 10

        self.active_learning = "qbc"  # {"dist_to_boundary", "least_confidence", "margin", "qbc", "uniform"}
        self.qbc_strategy = "margin"  # {"dist_to_boundary", "least_confidence", "margin"}
        self.active_epoch = 10
        self.budget_round = 5  # budget in each active learning round, real values indicate a percentage
        self.init_num_labeled = 5  # number of initial labeled data, real values indicate a percentage

        self.num_models = 25
        self.num_samples = 0.7
        
        self.retrain = False
        self.no_label_pred = True
        self.random_test = True

def PCA_reduce(xs, n_components = 72, random_state = 500):
    trunc_svd = TruncatedSVD(n_components = n_components, random_state = random_state)
    xs_reduced = trunc_svd.fit_transform(xs)
    pca_explained_variance_ratio = trunc_svd.explained_variance_ratio_.sum()

    print("xs_reduced.shape: {}".format(xs_reduced.shape))
    print("explained_variance_ratio_: {}".format(pca_explained_variance_ratio))
    return xs_reduced, trunc_svd

def VAE_reduce(xs, h_dim, z_dim, device, lr = 0.01, vae_epoch = 10):
    vae_model = VAE(input_dim = xs.shape[1], h_dim = h_dim, z_dim = z_dim).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(vae_model.parameters(), lr = lr)
    
    for epoch in range(vae_epoch):
        vae_model = vae_model.double()
        if epoch % 10 == 0:
            print("epoch:" + str(epoch))
        data_id = 0
        for x in xs:
            data_id += 1
            x = torch.tensor(x)
            x = x.to(dtype=torch.double)
            x = x.to(device)

            x_hat, mu, log_var = vae_model(x)

            loss, MSE, KLD = loss_function_vae(x_hat, x, mu, log_var, criterion)
            if data_id % 100 == 0:
                print("data_id:" + str(data_id))
                print("MSE:",MSE)
                print("KLD:",KLD)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    xs_reduced = []
    for x in xs:
        x = torch.tensor(x)
        x = x.to(dtype=torch.double)
        x = x.to(device)
        mu, log_var = vae_model.encode(x)
        sampled_z = vae_model.reparameterization(mu, log_var)
        xs_reduced.append(sampled_z.cpu().detach().numpy())
    xs_reduced = np.array(xs_reduced)
    print(xs_reduced.shape)

    return xs_reduced, vae_model

def main():
    cuda = torch.cuda.is_available()
    if cuda:
        device = torch.device("cuda")
        print("train on gpu")
    else:
        device = torch.device("cpu")
        print("train on cpu")

    args = Arguments()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    dataset_kwargs = dict()
    dataset_kwargs["pred"] = False
    dataset_kwargs["split_mode"] = args.split_mode

    origx_trains = []
    origx_tests = []
    for i in range(len(args.representations)):

        representation = args.representations[i]
        reduce_method = args.reduce_method[i]
        pca_component = args.pca_components[i]
        print(representation + ',' + reduce_method)
        if representation == "morgan_fp" or representation == "morgan_pka" or representation == "morgan_pka01":
            dataset_kwargs["representation_dim"] = args.representation_dim
        else:
            dataset_kwargs["representation_dim"] = None
        
        original_dataset = get_dataset(args.dataset, args.dataset_path, representation, **dataset_kwargs)
        (orig_x_train, orig_y_train_unnormalized), (orig_x_test, orig_y_test_unnormalized) = original_dataset

        num_orig_train = orig_x_train.shape[0]
        if reduce_method is not None:
            xs = np.concatenate((orig_x_train, orig_x_test), axis=0)
            xs_reduced = xs.copy()
            print("xs.shape: {}".format(xs.shape))
            if reduce_method == "pca":
                xs_reduced, trunc_svd = PCA_reduce(xs, n_components = pca_component, random_state=args.random_seed)
            elif reduce_method == "vae":
                xs, trunc_svd = PCA_reduce(xs=xs, n_components = pca_component, random_state=args.random_seed)
                xs_reduced, vae_model = VAE_reduce(xs, args.vae_hide_dim, args.vae_components, device, args.vae_lr, args.vae_epoch)
            orig_x_train = xs_reduced[:num_orig_train, :]
            orig_x_test = xs_reduced[num_orig_train:, :]
        print(orig_x_train.shape)
        origx_trains.append(orig_x_train)
        origx_tests.append(orig_x_test)

    orig_x_train = np.concatenate(origx_trains, axis=1)
    orig_x_test = np.concatenate(origx_tests, axis=1)


    orig_y_train = (orig_y_train_unnormalized - orig_y_train_unnormalized.min()) / \
                   (orig_y_train_unnormalized.max() - orig_y_train_unnormalized.min())
    orig_y_test = (orig_y_test_unnormalized - orig_y_train_unnormalized.min()) / \
                  (orig_y_train_unnormalized.max() - orig_y_train_unnormalized.min())
    print("orig_x_train.shape: {}".format(orig_x_train.shape))
    print("orig_x_test.shape: {}".format(orig_x_test.shape))

    old_x_test = orig_x_test.copy()
    old_x_train = orig_x_train.copy()

    low_id = []
    for i in range(len(orig_x_train)):
        low_id.append(i)
    
    low_id = np.array(low_id)
    
    low_x = old_x_train[low_id, ...]
    low_y = orig_y_train_unnormalized[low_id, ...]
    
    all_low = np.concatenate((low_x, old_x_test), axis=0)

    def kmeans_cluster():
        n_clusters = 60
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(all_low)
        clusters_train = kmeans.labels_
        
        # 以上聚类结束，后面进行权重统计
        dicts, average = assign_weight(clusters_train, n_clusters, low_x, low_y)
                
        vars = []
        svars = []
        ddd = []
        for it in dicts:
            arr = np.array(it)
            variance = np.var(arr)
            dd = mean_abs_dev(arr)
            vars.append(variance)
            ddd.append(dd)
            svars.append(variance**0.5)
        #print(average)
        #print(vars)
        print(mean(vars))
        #print(ddd)
        #print(svars)
        
        with open('put.csv', 'w') as f:
            for it in clusters_train:
                f.write(str(it) + '\n')
    kmeans_cluster()

    #db_pre = DBSCAN(eps = 10, min_samples = 5)
    def cluster_dbscan():
        db_pre = hdbscan.HDBSCAN(min_cluster_size=5)
        db_pre.fit(all_low)
        n_clusters = max(db_pre.labels_) + 1

        print(db_pre.labels_[-len(orig_x_test):])
        sums = [0] * n_clusters
        num = [0] * n_clusters
        for i in range(len(low_x)):
            ccc=db_pre.labels_[i]
            num[ccc] += 1
            sums[ccc] += low_y[i]

        average = []
        for i in range(len(sums)):
            if num[i]:
                average.append(sums[i]/num[i])
            else:
                average.append(0)

        ans = []
        for i in range(len(orig_x_test)):
            ans.append(average[db_pre.labels_[i + len(low_x)]])

        with open("./results/cluster.csv","w")  as f:
            f.write("cluster\n")
            for an in ans:
                f.write(str(an))
                f.write('\n')
    # cluster_dbscan()
    
    #TODO:聚类方法
    def cluster_kcenter():
        k = 60
        centers = np.zeros([k, all_low.shape[1]])
        centers[0, : ] = all_low[random.randint(0, all_low.shape[0]-1) , : ]
        cluster_idx = np.array([0 for i in range(all_low.shape[0])])
        num_centers = 1
        dist = spatial.distance.cdist(all_low, centers[range(num_centers), : ], 'euclidean').min(1)
        
        while num_centers < k:
            idx = dist.argmax()
            centers[num_centers, : ] = all_low[idx, : ]
            tmp_dist = spatial.distance.cdist(all_low, [all_low[idx, : ]], 'euclidean').min(1)
            concate_dist = np.vstack((dist, tmp_dist))
            tmp_dist_smaller = concate_dist.argmin(0)
            dist = concate_dist.min(0)
            cluster_idx[tmp_dist_smaller == 1] = num_centers
            num_centers += 1
        
        dicts, average = assign_weight(cluster_idx, k, low_x, low_y)

        vars = []
        svars = []
        ddd = []
        for it in dicts:
            arr = np.array(it)
            variance = np.var(arr)
            dd = mean_abs_dev(arr)
            vars.append(variance)
            ddd.append(dd)
            svars.append(variance**0.5)
        #print(average)
        #print(vars)
        print(mean(vars))
        #print(ddd)
        #print(svars)
        
        with open('put.csv', 'w') as f:
            for it in cluster_idx:
                f.write(str(it) + '\n')
    # cluster_kcenter()
    
    def cluster_kmedian():
        k = 60
        medians = np.zeros((k, all_low.shape[1]))
        medians = all_low[np.random.choice(all_low.shape[0], k, replace=False), : ]
        dist = spatial.distance.cdist(all_low, medians, 'euclidean')
        cluster_idx = dist.argmin(1)
        for i in range(k):
            medians[i] = np.median(all_low[cluster_idx == i], axis=0)
        
        iter_max = 100
        for i in range(iter_max):
            idx_old = cluster_idx
            metric = 'cityblock'
            # metric = 'euclidean'
            dist = spatial.distance.cdist(all_low, medians, metric=metric)
            cluster_idx = dist.argmin(1)
            if np.sum(cluster_idx != idx_old) == 0:
                break
            for i in range(k):
                medians[i] = np.median(all_low[cluster_idx == i], axis=0)
        
        dicts, average = assign_weight(cluster_idx, k, low_x, low_y)
        
        vars = []
        svars = []
        ddd = []
        for it in dicts:
            arr = np.array(it)
            variance = np.var(arr)
            dd = mean_abs_dev(arr)
            vars.append(variance)
            ddd.append(dd)
            svars.append(variance**0.5)
        #print(average)
        #print(vars)
        print(mean(vars))
        #print(ddd)
        #print(svars)
        
        with open('put.csv', 'w') as f:
            for it in cluster_idx:
                f.write(str(it) + '\n')
    # cluster_kmedian()
    
    def cluster_bisectingKMeans():
        k = 60
        bisect_means = BisectingKMeans(n_clusters=k, random_state=0).fit(all_low)
        clusters_train = bisect_means.labels_
        
        dicts, average = assign_weight(clusters_train, k, low_x, low_y)
                
        vars = []
        svars = []
        ddd = []
        for it in dicts:
            arr = np.array(it)
            variance = np.var(arr)
            dd = mean_abs_dev(arr)
            vars.append(variance)
            ddd.append(dd)
            svars.append(variance**0.5)
        #print(average)
        #print(vars)
        print(mean(vars))
        #print(ddd)
        #print(svars)
        
        with open('put.csv', 'w') as f:
            for it in clusters_train:
                f.write(str(it) + '\n')
    # cluster_bisectingKMeans()

if __name__ == "__main__":
    main()

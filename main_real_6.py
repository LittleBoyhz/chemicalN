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

from dataset_container import *
from sampling_methods import *
import utils

import torch
import torch.nn as nn

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

class FNN(nn.Module):
  def __init__(self, input_size=36, hidden_size=18, output_size=1, prob_dropout=0.1):
    super(FNN, self).__init__()
    self.predict = nn.Sequential(
        nn.Linear(input_size, hidden_size), nn.PReLU(), nn.Dropout(prob_dropout),
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

def train(x_train, y_train, model, preprocessor, boundary_value):
    """Trains a model.
    """
    x = preprocessor.fit_transform(x_train)
    y = utils.real_to_binary(y_train, boundary_value)
    if np.unique(y).shape[0] == 1:
        return -1
    model.fit(x, y)
    return model.score(x, y)

def test(x_test, y_test, model, preprocessor, boundary_value):
    """Tests a model.
    """
    x = preprocessor.transform(x_test)
    y = utils.real_to_binary(y_test, boundary_value)
    return model.score(x, y)

def predict(x, model, preprocessor):
    """Predicts class labels.
    """
    x = preprocessor.transform(x)
    return model.predict(x)

class Arguments:
    def __init__(self):
        self.random_seed = 500

        self.dataset = "real_6"
        self.split_mode = 0
        self.dataset_path = "./datasets/real"

        self.model = "logistic_regression"  # {"logistic_regression", "svm"}

        self.if_hybrid = 1  # {1,0} 1 refers to two types of descriptors 0 otherwise
        self.representationA = "pka_bde01"
        self.representationB = "morgan_fp"  # {"morgan_fp", "one_hot", "morgan_pka","ohe_pka"}

        self.representation = "morgan_fp"  # {"morgan_fp", "one_hot", "Mordred", "morgan_pka", "ohe_pka"}
        self.representation_dim = 2048  # used for morgan fingerprint or morgan_pka
        self.reduce_dim = 'pca' # {pca}
        self.pca_components = 72

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

def main():
    # args = parse_input()
    args = Arguments()

    # set random seeds
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    # load dataset
    dataset_kwargs = dict()
    dataset_kwargs["pred"] = False
    dataset_kwargs["split_mode"] = args.split_mode
    if args.if_hybrid == 0:
        if args.representation == "morgan_fp" or args.representation == "morgan_pka" or args.representation == "morgan_pka01":
            dataset_kwargs["representation_dim"] = args.representation_dim
        original_dataset = get_dataset(args.dataset, args.dataset_path, args.representation, **dataset_kwargs)
        (orig_x_train, orig_y_train_unnormalized), (orig_x_test, orig_y_test_unnormalized) = original_dataset

    else:  # if the representation consists of two types of descriptors
        original_dataseta = get_dataset(args.dataset, args.dataset_path, args.representationA, **dataset_kwargs)
        (orig_x_traina, orig_y_train_unnormalized), (orig_x_testa, orig_y_test_unnormalized) = original_dataseta
        
        if args.representationB == "morgan_fp" or args.representationB == "morgan_pka" or args.representationB == "morgan_pka01":
            dataset_kwargs["representation_dim"] = args.representation_dim
        original_datasetb = get_dataset(args.dataset, args.dataset_path, args.representationB, **dataset_kwargs)
        (orig_x_trainb, orig_y_train_unnormalized), (orig_x_testb, orig_y_test_unnormalized) = original_datasetb
        orig_x_train = np.concatenate([orig_x_traina, orig_x_trainb], axis=1)
        orig_x_test = np.concatenate([orig_x_testa, orig_x_testb], axis=1)

    orig_y_train = (orig_y_train_unnormalized - orig_y_train_unnormalized.min()) / \
                   (orig_y_train_unnormalized.max() - orig_y_train_unnormalized.min())
    orig_y_test = (orig_y_test_unnormalized - orig_y_train_unnormalized.min()) / \
                  (orig_y_train_unnormalized.max() - orig_y_train_unnormalized.min())
    print("orig_x_train.shape: {}".format(orig_x_train.shape))
    print("orig_x_test.shape: {}".format(orig_x_test.shape))
    num_orig_train = orig_x_train.shape[0]

    if args.reduce_dim is not None:
        xs = np.concatenate((orig_x_train, orig_x_test), axis=0)
        print("xs.shape: {}".format(xs.shape))

        if args.reduce_dim == "pca":
            trunc_svd = TruncatedSVD(n_components=args.pca_components, random_state=args.random_seed)
            xs_reduced = trunc_svd.fit_transform(xs)
            pca_explained_variance_ratio = trunc_svd.explained_variance_ratio_.sum()

            print("xs_reduced.shape: {}".format(xs_reduced.shape))
            print("explained_variance_ratio_: {}".format(pca_explained_variance_ratio))

            orig_x_train = xs_reduced[:num_orig_train, :]
            orig_x_test = xs_reduced[num_orig_train:, :]

    # shuffle the dataset
    random_permutation = np.random.default_rng().permutation(num_orig_train)
    orig_x_train = orig_x_train[random_permutation, :]
    orig_y_train = orig_y_train[random_permutation]

    al_init_kwargs = dict()  # arguments for initializing an active learning algorithm
    if args.active_learning == "dist_to_boundary":
        al_class = DistToBoundary
    elif args.active_learning == "least_confidence":
        al_class = LeastConfidence
    elif args.active_learning == "margin":
        al_class = Margin
    elif args.active_learning == "qbc":
        al_class = QBC
        al_init_kwargs["num_models"] = args.num_models
        al_init_kwargs["num_samples"] = args.num_samples
        al_init_kwargs["strategy"] = args.qbc_strategy
    elif args.active_learning == "uniform":
        al_class = UniformSampling
    else:
        raise ValueError("Unsupported active learning algorithm: {}".format(args.active_learning))

    model_init_kwargs = dict()
    if args.model == "logistic_regression":
        model_class = sklearn_linear_model.LogisticRegression
        model_init_kwargs["max_iter"] = 10000
    elif args.model == "svm":
        model_class = sklearn_svm.SVC
        model_init_kwargs["probability"] = True
    else:
        raise ValueError("Unsupported model name: {}".format(args.model))

    preprocessor_class = sklearn_preprocessing.StandardScaler
    
    new_models,new_preprocessors = utils.getNewModel2(model_class,model_init_kwargs,preprocessor_class,[0.5,0.7,0.6,0.85,0.4,0.3,0.15],train,orig_x_train,orig_y_train)
    
    svr_model = SVR(kernel='poly', degree=5)
    svr_model.fit(orig_x_train,orig_y_train)
    
    input_size = orig_x_train.shape[1]
    hidden_size = input_size // 2
    output_size = 1
    fnn_model = FNN(input_size, hidden_size, output_size)
    
    # Define the loss function and optimization algorithm
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(fnn_model.parameters(), lr=0.01)
    
    # Train the model
    for epoch in range(100):
        # Forward pass
        fnn_model = fnn_model.double()
        for i in range(len(orig_x_train)):
            input_fnn = torch.tensor(orig_x_train[i])
            input_fnn = input_fnn.to(dtype=torch.double)
            target_fnn = torch.tensor(orig_y_train[i])
            output = fnn_model(input_fnn)
            loss = criterion(output, target_fnn)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    svr_pred = svr_model.predict(orig_x_test)
    svr_pred = (orig_y_train_unnormalized.max() - orig_y_train_unnormalized.min()) * svr_pred + orig_y_train_unnormalized.min()
    '''
    for i in range(len(svr_pred)):
        if svr_pred[i]<0:
            svr_pred[i]=0
        elif svr_pred[i]>1:
            svr_pred[i]=100
        else:
            svr_pred[i]*=100
    '''
    with open("./results/svr.csv",'w') as f:
        f.write("svr\n")
        for i in range(len(svr_pred)):
            f.write(str(svr_pred[i]))
            f.write("\n")

    fnn_pred = []
    for i in range(len(orig_x_test)):
        input_fnn = torch.tensor(orig_x_test[i])
        input_fnn = input_fnn.to(dtype=torch.double)
        output = fnn_model(input_fnn)
        fnn_pred.append(output.item())
    fnn_pred = np.array(fnn_pred)
    fnn_pred = (orig_y_train_unnormalized.max() - orig_y_train_unnormalized.min()) * fnn_pred + orig_y_train_unnormalized.min()
    with open("./results/fnn.csv",'w') as f:
        f.write("fnn\n")
        for i in range(len(fnn_pred)):
            f.write(str(fnn_pred[i]))
            f.write("\n")
            
    n_clusters = 8
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(orig_x_train)
    clusters_train = kmeans.predict(orig_x_train)
    clusters_test = kmeans.predict(orig_x_test)
    
    sum = [0] * n_clusters
    num = [0] * n_clusters
    for i in range(len(orig_x_train)):
        num[clusters_train[i]] += 1
        sum[clusters_train[i]] += orig_y_train_unnormalized[i]
        
    average = []
    for i in range(len(sum)):
        if num[i]:
            average.append(sum[i]/num[i])
        else:
            average.append(0)

    ans = []
    for i in range(len(orig_x_test)):
        ans.append(average[clusters_test[i]])
    print(ans)
    print(orig_x_test[0])
  
if __name__ == "__main__":
    main()

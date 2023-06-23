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
    parser.add_argument("--retrain",type=int,default=0)

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

def one_iteration(boundary, model_class, model_init_kwargs, preprocessor_class, active_dataset, al_class,
                  al_init_kwargs, init_num_labeled, budget_round, active_epoch):
    fail = 0
    fail_max = 5
    for active_iter in range(active_epoch - 1):
        proportion = decimal.Decimal(str(active_dataset.num_labeled)) / \
                     decimal.Decimal(str(active_dataset.num_labeled + active_dataset.num_unlabeled))
        print("    Number of labeled samples: {}/{}, {:.1f}%".format(active_dataset.num_labeled,
                                                                     active_dataset.num_labeled + active_dataset.num_unlabeled,
                                                                     proportion * decimal.Decimal("100.0")))

        model_50 = model_class(**model_init_kwargs)
        preprocessor_50 = preprocessor_class()

        labeled_idx = active_dataset.index["labeled"]  # indices of labeled data
        # train the model using labeled data
        accuracy_train = train(active_dataset.orig_x_train[labeled_idx, ...],
                               active_dataset.orig_y_train[labeled_idx, ...], model_50, preprocessor_50, boundary)
        print("        Training accuracy: {}%".format(accuracy_train * 100.0))
        while accuracy_train == -1:  # active_iter must be equal to 0
            fail += 1
            if fail == fail_max or active_dataset.num_unlabeled == 0:
                fail = fail_max
                break

            active_dataset.uniformly_convert_unlabeled_to_labeled(init_num_labeled)
            model_50 = model_class(**model_init_kwargs)
            preprocessor_50 = preprocessor_class()

            labeled_idx = active_dataset.index["labeled"]  # indices of labeled data
            # train the model using labeled data
            accuracy_train = train(active_dataset.orig_x_train[labeled_idx, ...],
                                   active_dataset.orig_y_train[labeled_idx, ...], model_50, preprocessor_50, boundary)

        if fail == fail_max:
            break

        if active_dataset.num_unlabeled == 0 or active_iter == active_epoch:
            break

        al_alg = al_class(active_dataset, **al_init_kwargs)

        query_kwargs = dict()  # arguments for data query
        query_kwargs["model"] = model_50
        query_kwargs["preprocessor"] = preprocessor_50
        query_kwargs["model_class"] = model_class
        query_kwargs["model_init_args"] = model_init_kwargs
        query_kwargs["preprocessor_class"] = preprocessor_class
        query_kwargs["boundary_value"] = boundary

        indices, is_returned_indices_global = al_alg.query(min(budget_round, active_dataset.num_unlabeled),
                                                           **query_kwargs)
        active_dataset.move_from_unlabeled_to_labeled(indices, is_returned_indices_global)

    if fail == fail_max:
        print("    Training failed beacuse all training data belong to a single class")
        return -1.0, -1.0

    return model_50, preprocessor_50


def filter_data(boundary, model, preprocessor, active_dataset, init_num_labeled, filter_type):
    assert filter_type in ["gt", "lt"]

    labeled_idx = active_dataset.index["labeled"]
    unlabeled_idx = active_dataset.index["unlabeled"]

    labeled_x_train = active_dataset.orig_x_train[labeled_idx, ...]
    labeled_y_train = active_dataset.orig_y_train[labeled_idx, ...]
    condition_labeled = (labeled_y_train >= boundary) if filter_type == "gt" else (labeled_y_train <= boundary)
    labeled_x_train = labeled_x_train[condition_labeled, ...]
    labeled_y_train = labeled_y_train[condition_labeled, ...]
    num_existing_labeled = labeled_x_train.shape[0]

    unlabeled_x_train = active_dataset.orig_x_train[unlabeled_idx, ...]
    unlabeled_y_train = active_dataset.orig_y_train[unlabeled_idx, ...]
    predictions = predict(unlabeled_x_train, model, preprocessor)
    condition_unlabeled = (predictions == 1) if filter_type == "gt" else (predictions == 0)
    if condition_unlabeled.sum() == 0:  # the previously trained model is not well trained
        unlabeled_x_train = unlabeled_x_train
        unlabeled_y_train = unlabeled_y_train
    else:
        unlabeled_x_train = unlabeled_x_train[condition_unlabeled, :]
        unlabeled_y_train = unlabeled_y_train[condition_unlabeled]

    if num_existing_labeled and active_dataset.num_unlabeled:
        active_dataset = ActiveDataset(np.concatenate((labeled_x_train, unlabeled_x_train), axis=0),
                                       np.concatenate((labeled_y_train, unlabeled_y_train), axis=0),
                                       init_num_labeled, np.arange(num_existing_labeled))
    elif num_existing_labeled:
        active_dataset = ActiveDataset(labeled_x_train, labeled_y_train, init_num_labeled,
                                       np.arange(num_existing_labeled))
    elif active_dataset.num_unlabeled:
        active_dataset = ActiveDataset(unlabeled_x_train, unlabeled_y_train, init_num_labeled)
    else:
        print("    Training failed for lack of training data")
        active_dataset = -1.0

    return active_dataset


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

        self.representation = "morgan_fp"  # {"morgan_fp", "one_hot", "Mordred", "morgan_pka", "ohe_pka"}
        self.representation_dim = 2048  # used for morgan fingerprint or morgan_pka
        self.reduce_dim = "pca" # {pca}
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
    
    all_labeled_idx = []
    
    boundary = 0.5
    print("Boundary: {}".format(boundary))
    active_dataset = ActiveDataset(orig_x_train, orig_y_train, args.init_num_labeled)
    model_50, preprocessor_50 = \
        one_iteration(boundary, model_class, model_init_kwargs, preprocessor_class, active_dataset, al_class,
                      al_init_kwargs, args.init_num_labeled, args.budget_round, args.active_epoch)
    active_dataset_50 = copy.deepcopy(active_dataset)
    all_labeled_idx += active_dataset.index["labeled"].tolist()

    boundary = 0.7
    print("Boundary: {}".format(boundary))
    active_dataset = filter_data(0.5, model_50, preprocessor_50, active_dataset_50, args.init_num_labeled,
                                 filter_type="gt")  # greater than lt less than
    model_70, preprocessor_70 = \
        one_iteration(boundary, model_class, model_init_kwargs, preprocessor_class, active_dataset, al_class,
                      al_init_kwargs, args.init_num_labeled, args.budget_round, args.active_epoch)
    active_dataset_70 = copy.deepcopy(active_dataset)
    all_labeled_idx += active_dataset.index["labeled"].tolist()

    boundary = 0.85
    print("Boundary: {}".format(boundary))
    active_dataset = filter_data(0.7, model_70, preprocessor_70, active_dataset_70, args.init_num_labeled,
                                 filter_type="gt")
    model_85, preprocessor_85 = \
        one_iteration(boundary, model_class, model_init_kwargs, preprocessor_class, active_dataset, al_class,
                      al_init_kwargs, args.init_num_labeled, args.budget_round, args.active_epoch)
    all_labeled_idx += active_dataset.index["labeled"].tolist()

    boundary = 0.6
    print("Boundary: {}".format(boundary))
    active_dataset = filter_data(0.7, model_70, preprocessor_70, active_dataset_70, args.init_num_labeled,
                                 filter_type="lt")
    model_60, preprocessor_60 = \
        one_iteration(boundary, model_class, model_init_kwargs, preprocessor_class, active_dataset, al_class,
                      al_init_kwargs, args.init_num_labeled, args.budget_round, args.active_epoch)
    all_labeled_idx += active_dataset.index["labeled"].tolist()

    boundary = 0.3
    print("Boundary: {}".format(boundary))
    active_dataset = filter_data(0.5, model_50, preprocessor_50, active_dataset_50, args.init_num_labeled,
                                 filter_type="lt")
    model_30, preprocessor_30 = \
        one_iteration(boundary, model_class, model_init_kwargs, preprocessor_class, active_dataset, al_class,
                      al_init_kwargs, args.init_num_labeled, args.budget_round, args.active_epoch)
    active_dataset_30 = copy.deepcopy(active_dataset)
    all_labeled_idx += active_dataset.index["labeled"].tolist()

    boundary = 0.4
    print("Boundary: {}".format(boundary))
    active_dataset = filter_data(0.3, model_30, preprocessor_30, active_dataset_30, args.init_num_labeled,
                                 filter_type="gt")
    model_40, preprocessor_40 = \
        one_iteration(boundary, model_class, model_init_kwargs, preprocessor_class, active_dataset, al_class,
                      al_init_kwargs, args.init_num_labeled, args.budget_round, args.active_epoch)
    all_labeled_idx += active_dataset.index["labeled"].tolist()

    boundary = 0.15
    print("Boundary: {}".format(boundary))
    active_dataset = filter_data(0.3, model_30, preprocessor_30, active_dataset_30, args.init_num_labeled,
                                 filter_type="lt")
    model_15, preprocessor_15 = \
        one_iteration(boundary, model_class, model_init_kwargs, preprocessor_class, active_dataset, al_class,
                      al_init_kwargs, args.init_num_labeled, args.budget_round, args.active_epoch)
    all_labeled_idx += active_dataset.index["labeled"].tolist()
    
    all_labeled_idx=list(set(all_labeled_idx))
    all_unlabeled_idx = []
    for i in range(num_orig_train):
        if i not in all_labeled_idx:
            all_unlabeled_idx.append(i)
    
    all_unlabeled_idx = np.array(all_unlabeled_idx)
    all_labeled_idx=np.array(all_labeled_idx)

    x_test = orig_x_test
    y_test = orig_y_test_unnormalized
    idx_test = np.arange(x_test.shape[0])

    # 0.5
    predictions = predict(x_test, model_50, preprocessor_50)
    condition = (predictions == 1)
    idx_gt_50 = idx_test[condition, ...]
    idx_lt_50 = idx_test[~condition, ...]

    # 0.7
    if idx_gt_50.shape[0]:
        predictions = predict(x_test[idx_gt_50, ...], model_70, preprocessor_70)
        condition = (predictions == 1)
        idx_gt_70 = idx_gt_50[condition, ...]
        idx_gt_50_lt_70 = idx_gt_50[~condition, ...]
    else:
        idx_gt_70 = np.empty(shape=(0,))
        idx_gt_50_lt_70 = np.empty(shape=(0,))

    # 0.85
    if idx_gt_70.shape[0]:
        predictions = predict(x_test[idx_gt_70, ...], model_85, preprocessor_85)
        condition = (predictions == 1)
        idx_gt_85 = idx_gt_70[condition, ...]
        idx_gt_70_lt_85 = idx_gt_70[~condition, ...]
    else:
        idx_gt_85 = np.empty(shape=(0,))
        idx_gt_70_lt_85 = np.empty(shape=(0,))

    # 0.6
    if idx_gt_50_lt_70.shape[0]:
        predictions = predict(x_test[idx_gt_50_lt_70, ...], model_60, preprocessor_60)
        condition = (predictions == 1)
        idx_gt_60_lt_70 = idx_gt_50_lt_70[condition, ...]
        idx_gt_50_lt_60 = idx_gt_50_lt_70[~condition, ...]
    else:
        idx_gt_60_lt_70 = np.empty(shape=(0,))
        idx_gt_50_lt_60 = np.empty(shape=(0,))

    # 0.3
    if idx_lt_50.shape[0]:
        predictions = predict(x_test[idx_lt_50, ...], model_30, preprocessor_30)
        condition = (predictions == 1)
        idx_gt_30_lt_50 = idx_lt_50[condition, ...]
        idx_lt_30 = idx_lt_50[~condition, ...]
    else:
        idx_gt_30_lt_50 = np.empty(shape=(0,))
        idx_lt_30 = np.empty(shape=(0,))

    # 0.4
    if idx_gt_30_lt_50.shape[0]:
        predictions = predict(x_test[idx_gt_30_lt_50, ...], model_40, preprocessor_40)
        condition = (predictions == 1)
        idx_gt_40_lt_50 = idx_gt_30_lt_50[condition, ...]
        idx_gt_30_lt_40 = idx_gt_30_lt_50[~condition, ...]
    else:
        idx_gt_40_lt_50 = np.empty(shape=(0,))
        idx_gt_30_lt_40 = np.empty(shape=(0,))

    # 0.15
    if idx_lt_30.shape[0]:
        predictions = predict(x_test[idx_lt_30, ...], model_15, preprocessor_15)
        condition = (predictions == 1)
        idx_gt_15_lt_30 = idx_lt_30[condition, ...]
        idx_lt_15 = idx_lt_30[~condition, ...]
    else:
        idx_gt_15_lt_30 = np.empty(shape=(0,))
        idx_lt_15 = np.empty(shape=(0,))

    results = [""] * x_test.shape[0]
    indices = [idx_lt_15, idx_gt_15_lt_30, idx_gt_30_lt_40, idx_gt_40_lt_50, idx_gt_50_lt_60, idx_gt_60_lt_70,
               idx_gt_70_lt_85, idx_gt_85]
    after_norm = [0.15, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1]
    before_norm = list()
    for x in after_norm:
        before_norm.append(x * (orig_y_train_unnormalized.max() - orig_y_train_unnormalized.min()) +
                           orig_y_train_unnormalized.min())
    print(before_norm)
    to_str = {
        0: "0.0-{:.1f}".format(before_norm[0]),
        1: "{:.1f}-{:.1f}".format(before_norm[0], before_norm[1]),
        2: "{:.1f}-{:.1f}".format(before_norm[1], before_norm[2]),
        3: "{:.1f}-{:.1f}".format(before_norm[2], before_norm[3]),
        4: "{:.1f}-{:.1f}".format(before_norm[3], before_norm[4]),
        5: "{:.1f}-{:.1f}".format(before_norm[4], before_norm[5]),
        6: "{:.1f}-{:.1f}".format(before_norm[5], before_norm[6]),
        7: "{:.1f}-{:.1f}".format(before_norm[6], before_norm[7])
    }
    for i, idx in enumerate(indices):
        if idx.shape[0] == 0:
            continue

        for j in range(idx.shape[0]):
            results[idx[j]] = to_str[i]

    if args.init_num_labeled == -1:
        init_num_labeled = num_orig_train
    elif args.init_num_labeled < 1.0:
        init_num_labeled = round(args.init_num_labeled * num_orig_train)
    else:
        init_num_labeled = round(args.init_num_labeled)
    if args.budget_round < 1.0:  # convert to int
        budget = round(args.budget_round * num_orig_train)
    else:
        budget = round(args.budget_round)

    if args.model == "svm":
        wandb_group = "svm"
    elif args.model == "logistic_regression":
        wandb_group = "lore"
    al = "qbc_" + args.qbc_strategy if args.active_learning == "qbc" else args.active_learning
    if args.init_num_labeled == -1:
        wandb_group += "-whole data"
    else:
        wandb_group += (
                "-" + al + "_" + str(init_num_labeled) + "_" + str(budget) + "_" + str(args.active_epoch))
    if args.if_hybrid == 0:
        if args.representation == "morgan_fp" or args.representation == "morgan_pka":
            wandb_rep = args.representation + "_" + str(args.representation_dim)
        else:
            wandb_rep = args.representation
    else:
        wandb_rep = args.representationA
        if args.representationB == "morgan_fp" or args.representationB == "morgan_pka":
            wandb_rep += "_" + args.representationB + "_" + str(args.representation_dim)
        else:
            wandb_rep += "_" + args.representationB

    wandb_group += ("-" + wandb_rep)
    if args.reduce_dim is not None:
        if args.reduce_dim == "pca":
            wandb_group = wandb_group + "-pca_{}_{:.2f}".format(args.pca_components, pca_explained_variance_ratio)

    print(wandb_group)

    save_path = os.path.join("./results/real_5/split_" + str(args.split_mode))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    result_csv = os.path.join(save_path, wandb_group + ".csv")
    if os.path.exists(result_csv):
        df = pd.read_csv(result_csv)
    else:
        df = pd.DataFrame(data=y_test, columns=["True Yield"])
    col_name = "Prediction_Ours_" + str(args.random_seed)
    df[col_name] = pd.DataFrame(data=results, columns=[col_name])
    df.to_csv(result_csv, index=False)
    
    if args.retrain:
    
        #重新利用所有标签训练
        new_models,new_preprocessors = utils.getNewModel(model_class,model_init_kwargs,preprocessor_class,[0.5,0.7,0.6,0.85,0.4,0.3,0.15],train,all_labeled_idx,orig_x_train,orig_y_train)
        model_50_new,model_70_new,model_60_new,model_85_new,model_40_new,model_30_new,model_15_new=new_models
        preprocessor_50_new,preprocessor_70_new,preprocessor_60_new,preprocessor_85_new,preprocessor_40_new,preprocessor_30_new,preprocessor_15_new=new_preprocessors
        
        # 统计每个svm在全样本空间准确性
        test_boundarys=[0.5,0.7,0.6,0.85,0.4,0.3,0.15]
        old_models=[model_50,model_70,model_60,model_85,model_40,model_30,model_15]
        old_preprocessors=[preprocessor_50,preprocessor_70,preprocessor_60,preprocessor_85,preprocessor_40,preprocessor_30,preprocessor_15]
        
        for i in range(len(test_boundarys)):
            test_boundary=test_boundarys[i]
            print("测试边界：",test_boundary)
            true_result = [_y_test > test_boundary*100 for _y_test in y_test]
            predictions = predict(x_test, old_models[i], old_preprocessors[i])
            hits=sum([predictions[j]==true_result[j] for j in range(len(x_test))])
            acc=hits/len(x_test)
            Phits=sum([predictions[j]==true_result[j] and true_result[j]==1 for j in range(len(x_test))])
            if sum(true_result)!= 0:
                Pacc=Phits/sum(true_result)
            else:
                Pacc=1
            print("原svm：",acc)
            predictions = predict(x_test, new_models[i], new_preprocessors[i])
            hits=sum([predictions[j]==true_result[j] for j in range(len(x_test))])
            acc=hits/len(x_test)
            Phits=sum([predictions[j]==true_result[j] and true_result[j]==1 for j in range(len(x_test))])
            if sum(true_result)!= 0:
                Pacc=Phits/sum(true_result)
            else:
                Pacc=1
            print("新svm：",acc)
            print("")
        
        x_test = orig_x_test
        y_test = orig_y_test_unnormalized
        idx_test = np.arange(x_test.shape[0])
        predictions = predict(x_test, model_50_new, preprocessor_50_new)
        condition = (predictions == 1)
        idx_gt_50 = idx_test[condition, ...]
        idx_lt_50 = idx_test[~condition, ...]
        if idx_gt_50.shape[0]:
            predictions = predict(x_test[idx_gt_50, ...], model_70_new, preprocessor_70_new)
            condition = (predictions == 1)
            idx_gt_70 = idx_gt_50[condition, ...]
            idx_gt_50_lt_70 = idx_gt_50[~condition, ...]
        else:
            idx_gt_70 = np.empty(shape=(0,))
            idx_gt_50_lt_70 = np.empty(shape=(0,))
        if idx_gt_70.shape[0]:
            predictions = predict(x_test[idx_gt_70, ...], model_85_new, preprocessor_85_new)
            condition = (predictions == 1)
            idx_gt_85 = idx_gt_70[condition, ...]
            idx_gt_70_lt_85 = idx_gt_70[~condition, ...]
        else:
            idx_gt_85 = np.empty(shape=(0,))
            idx_gt_70_lt_85 = np.empty(shape=(0,))
        if idx_gt_50_lt_70.shape[0]:
            predictions = predict(x_test[idx_gt_50_lt_70, ...], model_60_new, preprocessor_60_new)
            condition = (predictions == 1)
            idx_gt_60_lt_70 = idx_gt_50_lt_70[condition, ...]
            idx_gt_50_lt_60 = idx_gt_50_lt_70[~condition, ...]
        else:
            idx_gt_60_lt_70 = np.empty(shape=(0,))
            idx_gt_50_lt_60 = np.empty(shape=(0,))
        if idx_lt_50.shape[0]:
            predictions = predict(x_test[idx_lt_50, ...], model_30_new, preprocessor_30_new)
            condition = (predictions == 1)
            idx_gt_30_lt_50 = idx_lt_50[condition, ...]
            idx_lt_30 = idx_lt_50[~condition, ...]
        else:
            idx_gt_30_lt_50 = np.empty(shape=(0,))
            idx_lt_30 = np.empty(shape=(0,))
        if idx_gt_30_lt_50.shape[0]:
            predictions = predict(x_test[idx_gt_30_lt_50, ...], model_40_new, preprocessor_40_new)
            condition = (predictions == 1)
            idx_gt_40_lt_50 = idx_gt_30_lt_50[condition, ...]
            idx_gt_30_lt_40 = idx_gt_30_lt_50[~condition, ...]
        else:
            idx_gt_40_lt_50 = np.empty(shape=(0,))
            idx_gt_30_lt_40 = np.empty(shape=(0,))
        if idx_lt_30.shape[0]:
            predictions = predict(x_test[idx_lt_30, ...], model_15_new, preprocessor_15_new)
            condition = (predictions == 1)
            idx_gt_15_lt_30 = idx_lt_30[condition, ...]
            idx_lt_15 = idx_lt_30[~condition, ...]
        else:
            idx_gt_15_lt_30 = np.empty(shape=(0,))
            idx_lt_15 = np.empty(shape=(0,))
        results = [""] * x_test.shape[0]
        indices = [idx_lt_15, idx_gt_15_lt_30, idx_gt_30_lt_40, idx_gt_40_lt_50, idx_gt_50_lt_60, idx_gt_60_lt_70,
                idx_gt_70_lt_85, idx_gt_85]
        after_norm = [0.15, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1]
        before_norm = list()
        for x in after_norm:
            before_norm.append(x * (orig_y_train_unnormalized.max() - orig_y_train_unnormalized.min()) +
                            orig_y_train_unnormalized.min())
        print(before_norm)
        to_str = {
            0: "0.0-{:.2f}".format(before_norm[0]),
            1: "{:.2f}-{:.2f}".format(before_norm[0], before_norm[1]),
            2: "{:.2f}-{:.2f}".format(before_norm[1], before_norm[2]),
            3: "{:.2f}-{:.2f}".format(before_norm[2], before_norm[3]),
            4: "{:.2f}-{:.2f}".format(before_norm[3], before_norm[4]),
            5: "{:.2f}-{:.2f}".format(before_norm[4], before_norm[5]),
            6: "{:.2f}-{:.2f}".format(before_norm[5], before_norm[6]),
            7: "{:.2f}-{:.2f}".format(before_norm[6], before_norm[7])
        }
        for i, idx in enumerate(indices):
            if idx.shape[0] == 0:
                continue
            for j in range(idx.shape[0]):
                results[idx[j]] = to_str[i]
        if args.init_num_labeled == -1:
            init_num_labeled = num_orig_train
        elif args.init_num_labeled < 1.0:
            init_num_labeled = round(args.init_num_labeled * num_orig_train)
        else:
            init_num_labeled = round(args.init_num_labeled)
        if args.budget_round < 1.0:  # convert to int
            budget = round(args.budget_round * num_orig_train)
        else:
            budget = round(args.budget_round)
        if args.model == "svm":
            wandb_group = "svm"
        elif args.model == "logistic_regression":
            wandb_group = "lore"
        al = "qbc_" + args.qbc_strategy if args.active_learning == "qbc" else args.active_learning
        if args.init_num_labeled == -1:
            wandb_group += "-whole data"
        else:
            wandb_group += (
                    "-" + al + "_" + str(init_num_labeled) + "_" + str(budget) + "_" + str(args.active_epoch))
        if args.if_hybrid == 0:
            if args.representation == "morgan_fp" or args.representation == "morgan_pka":
                wandb_rep = args.representation + "_" + str(args.representation_dim)
            else:
                wandb_rep = args.representation
        else:
            wandb_rep = args.representationA
            if args.representationB == "morgan_fp" or args.representationB == "morgan_pka":
                wandb_rep += "_" + args.representationB + "_" + str(args.representation_dim)
            else:
                wandb_rep += "_" + args.representationB
        wandb_group += ("-" + wandb_rep)
        if args.reduce_dim is not None:
            if args.reduce_dim == "pca":
                wandb_group = wandb_group + "-pca_{}_{:.2f}".format(args.pca_components, pca_explained_variance_ratio)
        print(wandb_group)
        save_path = os.path.join("./results/real_5/split_" + str(args.split_mode))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        result_csv = os.path.join(save_path, 'new-' + wandb_group + ".csv")
        if os.path.exists(result_csv):
            df = pd.read_csv(result_csv)
        else:
            df = pd.DataFrame(data=y_test, columns=["True Yield"])
        col_name = "Prediction_Ours_" + str(args.random_seed)
        df[col_name] = pd.DataFrame(data=results, columns=[col_name])
        df.to_csv(result_csv, index=False)
    random_idx = random.sample(range(0,orig_x_train.shape[0]), all_labeled_idx.shape[0])
    unrandom_idx = []
    for i in range(num_orig_train):
        if i not in random_idx:
            unrandom_idx.append(i)
    unrandom_idx = np.array(unrandom_idx)
    random_idx = np.array(random_idx)
    
    labeled_data=orig_x_train[all_labeled_idx, ...]
    labeled_yield=orig_y_train[all_labeled_idx, ...]
    unlabeled_data = orig_x_train[all_unlabeled_idx, ...]
    unlabeled_yield = orig_y_train[all_unlabeled_idx, ...] * 100
    random_data = orig_x_train[random_idx, ...]
    random_yield = orig_y_train[random_idx, ...]
    unrandom_data = orig_x_train[unrandom_idx, ...]
    unrandom_yield = orig_y_train[unrandom_idx, ...] * 100
    
    svr_model = SVR(kernel='poly', degree=5)
    svr_model.fit(labeled_data,labeled_yield)
    
    print(labeled_data.shape)
    input_size = labeled_data.shape[1]
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
        for i in range(len(labeled_data)):
            input_fnn = torch.tensor(labeled_data[i])
            input_fnn = input_fnn.to(dtype=torch.double)
            target_fnn = torch.tensor(labeled_yield[i])
            output = fnn_model(input_fnn)
            loss = criterion(output, target_fnn)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    if not args.no_label_pred:
        final_x = x_test
        final_y = y_test
        output_file = "./results/record.csv"
    else:
        final_x = unlabeled_data
        final_y = unlabeled_yield
        output_file = "./results/record_test.csv"
        
    svr_pred = svr_model.predict(final_x)
    for i in range(len(svr_pred)):
        if svr_pred[i]<0:
            svr_pred[i]=0
        elif svr_pred[i]>1:
            svr_pred[i]=100
        else:
            svr_pred[i]*=100
    r2_svr = r2_score(final_y,svr_pred)
    mse_svr = mean_squared_error(final_y,svr_pred)
    mae_svr = mean_absolute_error(final_y,svr_pred)
    print(r2_svr,mse_svr,mae_svr)
    with open("./results/svr.csv",'w') as f:
        f.write("pred,real\n")
        for i in range(len(svr_pred)):
            f.write(str(svr_pred[i]))
            f.write(",")
            f.write(str(final_y[i]))
            f.write("\n")
            
    fnn_pred = []
    for i in range(len(final_x)):
        input_fnn = torch.tensor(final_x[i])
        input_fnn = input_fnn.to(dtype=torch.double)
        output = fnn_model(input_fnn)
        fnn_pred.append(output.item())
    for i in range(len(fnn_pred)):
        if math.isnan(fnn_pred[i]):
            fnn_pred[i]=0
        if fnn_pred[i]<0:
            fnn_pred[i]=0
        elif fnn_pred[i]>1:
            fnn_pred[i]=100
        else:
            fnn_pred[i]*=100
    fnn_pred = np.array(fnn_pred)
    with open("./results/fnn.csv",'w') as f:
        f.write("pred,real\n")
        for i in range(len(fnn_pred)):
            f.write(str(fnn_pred[i]))
            f.write(",")
            f.write(str(final_y[i]))
            f.write("\n")

    r2_fnn = r2_score(final_y,fnn_pred)
    mse_fnn = mean_squared_error(final_y,fnn_pred)
    mae_fnn = mean_absolute_error(final_y,fnn_pred)
    print(r2_fnn,mse_fnn,mae_fnn)
    
    with open(output_file, "a+") as f:
        f.write(str(args.dataset) + ',')
        if args.if_hybrid:
            f.write(args.representationA + ',')
            f.write(args.representationB + ',')
        else:
            f.write(args.representation + ',' + ',')
        f.write(str(r2_svr) + ',')
        f.write(str(mse_svr) + ',')
        f.write(str(mae_svr) + ',')
        f.write(str(r2_fnn) + ',')
        f.write(str(mse_fnn) + ',')
        f.write(str(mae_fnn) + '\n')
    
    if args.random_test:
        random_svr = SVR(kernel='poly', degree=5)
        random_svr.fit(random_data,random_yield)
        
        input_size = random_data.shape[1]
        hidden_size = input_size // 2
        output_size = 1
        random_fnn = FNN(input_size, hidden_size, output_size)
        
        # Define the loss function and optimization algorithm
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(random_fnn.parameters(), lr=0.01)
        
        # Train the model
        for epoch in range(100):
            # Forward pass
            random_fnn = random_fnn.double()
            for i in range(len(random_data)):
                input_fnn = torch.tensor(random_data[i])
                input_fnn = input_fnn.to(dtype=torch.double)
                target_fnn = torch.tensor(random_yield[i])
                output = random_fnn(input_fnn)
                loss = criterion(output, target_fnn)

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        svr_pred = random_svr.predict(unrandom_data)
        for i in range(len(svr_pred)):
            if svr_pred[i]<0:
                svr_pred[i]=0
            elif svr_pred[i]>1:
                svr_pred[i]=100
            else:
                svr_pred[i]*=100
        r2_svr_n = r2_score(unrandom_yield,svr_pred)
                
        fnn_pred = []
        for i in range(len(unrandom_data)):
            input_fnn = torch.tensor(unrandom_data[i])
            input_fnn = input_fnn.to(dtype=torch.double)
            output = random_fnn(input_fnn)
            fnn_pred.append(output.item())
        for i in range(len(fnn_pred)):
            if math.isnan(fnn_pred[i]):
                fnn_pred[i]=0
            if fnn_pred[i]<0:
                fnn_pred[i]=0
            elif fnn_pred[i]>1:
                fnn_pred[i]=100
            else:
                fnn_pred[i]*=100
        fnn_pred = np.array(fnn_pred)

        r2_fnn_n = r2_score(unrandom_yield,fnn_pred)
        
        with open("./results/random.csv", "a+") as f:
            f.write(str(args.dataset) + ',')
            if args.if_hybrid:
                f.write(args.representationA + ',')
                f.write(args.representationB + ',')
            else:
                f.write(args.representation + ',' + ',')
            f.write(str(r2_svr) + ',')
            f.write(str(r2_fnn) + ',')
            f.write(str(r2_svr_n) + ',')
            f.write(str(r2_fnn_n) + '\n')

if __name__ == "__main__":
    main()

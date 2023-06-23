import argparse
import copy
import decimal
import pickle
import random

import numpy as np
from scipy.spatial import distance
from sklearn.decomposition import TruncatedSVD
import sklearn.linear_model as sklearn_linear_model
import sklearn.preprocessing as sklearn_preprocessing
import sklearn.svm as sklearn_svm
import wandb

from dataset_container import *
from sampling_methods import *
import utils


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

    args = parser.parse_args()
    return args


class Arguments:
    def __init__(self):
        self.random_seed = 400

        self.dataset = "rxnfp"  # {"real_1", "real_2", "real_3"}
        self.dataset_path = "./datasets/real"

        self.model = "logistic_regression"  # {"logistic_regression", "svm"}
        self.representation = "one_hot"  # {"morgan_fp", "one_hot"}
        self.representation_dim = 1024  # used for morgan fingerprint
        self.reduce_dim = None
        self.pca_components = 36

        self.active_learning = "uniform"  # {"dist_to_boundary", "least_confidence", "margin", "qbc", "uniform"}
        self.qbc_strategy = "margin"  # {"dist_to_boundary", "least_confidence", "margin"}
        self.active_epoch = 10
        self.budget_round = 10  # budget in each active learning round, real values indicate a percentage
        self.init_num_labeled = 10  # number of initial labeled data, real values indicate a percentage

        self.num_models = 25
        self.num_samples = 0.7

        # self.boundaries = [0.5, 0.75, 0.875, 0.9375, 0.96875, 0.984375, 0.9921875]
        self.boundaries = [0.5, 0.75, 0.875, 0.9375, 0.96875]


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


def main():
    # args = parse_input()
    args = Arguments()

    # set random seeds
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    # load dataset
    dataset_kwargs = dict()
    dataset_kwargs["pred"] = False
    if args.representation == "morgan_fp":
        dataset_kwargs["representation_dim"] = args.representation_dim
    original_dataset = get_dataset(args.dataset, args.dataset_path, args.representation, **dataset_kwargs)
    (orig_x_train, orig_y_train_unnormalized), (orig_x_test, orig_y_test_unnormalized) = original_dataset
    orig_y_train = (orig_y_train_unnormalized - orig_y_train_unnormalized.min()) / \
                   (orig_y_train_unnormalized.max() - orig_y_train_unnormalized.min())
    orig_y_test = (orig_y_test_unnormalized - orig_y_test_unnormalized.min()) / \
                  (orig_y_test_unnormalized.max() - orig_y_test_unnormalized.min())
    print("orig_x_train.shape: {}".format(orig_x_train.shape))
    print("orig_x_test.shape: {}".format(orig_x_test.shape))
    num_orig_train = orig_x_train.shape[0]
    num_orig_test = orig_x_test.shape[0]

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

    model_series = []  # a list that stores the trained model
    preprocessor_series = []  # a list that stores the data pre-processing tools

    boundaries = [float(x) for x in args.boundaries]
    num_labeled_required_train = []
    for boundary_idx, boundary_value in enumerate(boundaries):
        print("Boundary: {}".format(boundary_value))

        if boundary_idx > 0:  # use the previously trained model to filter training data
            if args.init_num_labeled == -1:
                condition = (orig_y_train > boundaries[boundary_idx - 1])
                active_dataset = ActiveDataset(orig_x_train[condition, ...], orig_y_train[condition, ...],
                                               args.init_num_labeled)
            else:
                labeled_idx = active_dataset.index["labeled"]
                labeled_x_train = active_dataset.orig_x_train[labeled_idx, ...]
                labeled_y_train = active_dataset.orig_y_train[labeled_idx, ...]
                condition_labeled = (labeled_y_train > boundaries[boundary_idx - 1])
                labeled_x_train = labeled_x_train[condition_labeled, ...]
                labeled_y_train = labeled_y_train[condition_labeled, ...]
                num_existing_labeled = labeled_x_train.shape[0]

                if active_dataset.num_unlabeled:
                    unlabeled_idx = active_dataset.index["unlabeled"]
                    unlabeled_x_train = active_dataset.orig_x_train[unlabeled_idx, ...]
                    unlabeled_y_train = active_dataset.orig_y_train[unlabeled_idx, ...]
                    predictions = predict(unlabeled_x_train, model_series[boundary_idx - 1],
                                          preprocessor_series[boundary_idx - 1])
                    condition_unlabeled = (predictions == 1)
                    if condition_unlabeled.sum() == 0:  # the previously trained model is not well trained
                        unlabeled_x_train = unlabeled_x_train
                        unlabeled_y_train = unlabeled_y_train
                    else:
                        unlabeled_x_train = unlabeled_x_train[condition_unlabeled, :]
                        unlabeled_y_train = unlabeled_y_train[condition_unlabeled]

                if num_existing_labeled and active_dataset.num_unlabeled:
                    active_dataset = ActiveDataset(np.concatenate((labeled_x_train, unlabeled_x_train), axis=0),
                                                   np.concatenate((labeled_y_train, unlabeled_y_train), axis=0),
                                                   np.arange(num_existing_labeled))
                elif num_existing_labeled:
                    active_dataset = ActiveDataset(labeled_x_train, labeled_y_train, np.arange(num_existing_labeled))
                elif active_dataset.num_unlabeled:
                    active_dataset = ActiveDataset(unlabeled_x_train, unlabeled_y_train, args.init_num_labeled)
                else:
                    print("    Training failed for lack of training data")
                    break

            condition_test = (orig_y_test > boundaries[boundary_idx - 1])  # filter test data
            x_test = orig_x_test[condition_test, :]
            y_test = orig_y_test[condition_test]
        else:
            active_dataset = ActiveDataset(orig_x_train, orig_y_train, args.init_num_labeled)
            x_test = orig_x_test
            y_test = orig_y_test

        fail = 0
        fail_max = 5
        for active_iter in range(args.active_epoch + 1):
            proportion = decimal.Decimal(str(active_dataset.num_labeled)) / \
                         decimal.Decimal(str(active_dataset.num_labeled + active_dataset.num_unlabeled))
            print("    Number of labeled samples: {}/{}, {:.1f}%".format(active_dataset.num_labeled,
                                                                         active_dataset.num_labeled + active_dataset.num_unlabeled,
                                                                         proportion * decimal.Decimal("100.0")))

            model = model_class(**model_init_kwargs)
            preprocessor = preprocessor_class()

            labeled_idx = active_dataset.index["labeled"]  # indices of labeled data
            # train the model using labeled data
            accuracy_train = train(active_dataset.orig_x_train[labeled_idx, ...],
                                   active_dataset.orig_y_train[labeled_idx, ...], model, preprocessor, boundary_value)
            print("        Training accuracy: {}%".format(accuracy_train * 100.0))
            while accuracy_train == -1:  # active_iter must be equal to 0
                fail += 1
                if fail == fail_max or active_dataset.num_unlabeled == 0:
                    fail = fail_max
                    break

                active_dataset.uniformly_convert_unlabeled_to_labeled(args.init_num_labeled)
                model = model_class(**model_init_kwargs)
                preprocessor = preprocessor_class()

                labeled_idx = active_dataset.index["labeled"]  # indices of labeled data
                # train the model using labeled data
                accuracy_train = train(active_dataset.orig_x_train[labeled_idx, ...],
                                       active_dataset.orig_y_train[labeled_idx, ...], model, preprocessor,
                                       boundary_value)

            if fail == fail_max:
                break

            # test the trained model
            accuracy_test = test(x_test, y_test, model, preprocessor, boundary_value)
            print("        Test accuracy: {}%".format(accuracy_test * 100.0))

            if active_dataset.num_unlabeled == 0 or active_iter == args.active_epoch:
                break

            al_alg = al_class(active_dataset, **al_init_kwargs)

            query_kwargs = dict()  # arguments for data query
            query_kwargs["model"] = model
            query_kwargs["preprocessor"] = preprocessor
            query_kwargs["model_class"] = model_class
            query_kwargs["model_init_args"] = model_init_kwargs
            query_kwargs["preprocessor_class"] = preprocessor_class
            query_kwargs["boundary_value"] = boundary_value

            indices, is_returned_indices_global = al_alg.query(min(args.budget_round, active_dataset.num_unlabeled),
                                                               **query_kwargs)
            active_dataset.move_from_unlabeled_to_labeled(indices, is_returned_indices_global)

        if fail == fail_max:
            print("    Training failed beacuse all training data belong to a single class")
            break
        else:
            model_series.append(copy.deepcopy(model))
            preprocessor_series.append(copy.deepcopy(preprocessor))
            if boundary_idx == 0:
                num_labeled_required_train.append(active_dataset.num_labeled - active_dataset.num_existing_labeled)
            else:
                num_labeled_required_train.append(active_dataset.num_labeled - active_dataset.num_existing_labeled +
                                                  num_labeled_required_train[boundary_idx - 1])

    if args.init_num_labeled == -1:
        for i in range(len(num_labeled_required_train)):
            num_labeled_required_train[i] = orig_x_train.shape[0]

    print("num_labeled_required_train: {}".format(num_labeled_required_train))
    print("")

    if len(model_series) > 1:
        final_trained_boundary = boundaries[len(model_series) - 1]
        final_helpful_boundary = final_trained_boundary
        num_points_remained = []
        true_positive_remained = []
        true_positive_remained_ratio = []
        yield_result = []
        num_labeled_required = []

        to_pred_x = orig_x_test
        to_pred_y = orig_y_test_unnormalized
        for i in range(len(model_series)):
            print("Doing calculation for {}".format(args.boundaries[i]))
            model = model_series[i]
            preprocessor = preprocessor_series[i]
            predictions = predict(to_pred_x, model, preprocessor)
            condition = (predictions == 1)
            to_pred_x = to_pred_x[condition, :]
            to_pred_y = to_pred_y[condition]
            print("    {}".format(to_pred_y.shape[0]))
            print("    {}".format(to_pred_y))

            if to_pred_y.shape[0] == 0:
                all_are_helpful = 0
                final_helpful_boundary = boundaries[i-1]
                break

            num_points_remained.append(to_pred_x.shape[0])
            num_true_positive = (to_pred_y >= args.boundaries[i]).sum()
            true_positive_remained.append(num_true_positive)
            true_positive_remained_ratio.append(num_true_positive / (orig_y_test_unnormalized >= args.boundaries[i]).sum())
            yield_result.append(to_pred_y.max().item())

        for i in range(len(num_labeled_required_train), len(boundaries)):
            num_labeled_required_train.append(0)
        for i in range(len(yield_result), len(boundaries)):
            yield_result.append(0)
        for i in range(len(num_points_remained), len(boundaries)):
            num_points_remained.append(0)
        for i in range(len(true_positive_remained), len(boundaries)):
            true_positive_remained.append(0)
            true_positive_remained_ratio.append(0.0)
        for xx, yy in zip(num_labeled_required_train, num_points_remained):
            num_labeled_required.append(xx + yy)

        print("final_trained_boundary: {}".format(final_trained_boundary))
        print("final_helpful_boundary: {}".format(final_helpful_boundary))
        print("boundaries: {}".format(boundaries))
        print("yield: {}".format(yield_result))
        print("num_labeled_required_train: {}".format(num_labeled_required_train))
        print("num_labeled_required: {}".format(num_labeled_required))
        print("num_points_left: {}".format(num_points_remained))
        print("true_positive_remained: {}".format(true_positive_remained))
        print("true_positive_remained_ratio: {}".format(true_positive_remained_ratio))

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
        # prepare to use Weights & Bias
        wandb_job_type = args.dataset
        if args.model == "svm":
            wandb_group = "svm"
        elif args.model == "logistic_regression":
            wandb_group = "lore"
        al = "qbc_" + args.qbc_strategy if args.active_learning == "qbc" else args.active_learning
        if args.init_num_labeled == -1:
            wandb_group += "-whole data"
        else:
            wandb_group += ("-" + al + "_" + str(init_num_labeled) + "_" + str(budget) + "_" + str(args.active_epoch))
        if args.representation == "morgan_fp":
            wandb_rep = args.representation + "_" + str(args.representation_dim)
        else:
            wandb_rep = args.representation
        wandb_group += ("-" + wandb_rep)
        if args.reduce_dim is not None:
            if args.reduce_dim == "pca":
                wandb_group = wandb_group + "-pca_{}_{:.2f}".format(args.pca_components, pca_explained_variance_ratio)
        wandb_run = wandb.init(entity="opstreadstone", project="chemical", job_type=wandb_job_type, group=wandb_group,
                               name=wandb_group + "-seed_" + str(args.random_seed), reinit=True)
        wandb.define_metric("Boundary")
        wandb.define_metric("Yields", step_metric="Boundary")
        wandb.define_metric("Number of Labels Required for Training", step_metric="Boundary")
        wandb.define_metric("Number of Labels Required for Training (Ratio)", step_metric="Boundary")
        wandb.define_metric("Number of Labels Required", step_metric="Boundary")
        wandb.define_metric("Number of Labels Required (Ratio)", step_metric="Boundary")
        wandb.define_metric("Number of Remaining Data", step_metric="Boundary")
        wandb.define_metric("Number of Remaining Data (Ratio)", step_metric="Boundary")
        wandb.define_metric("Number of True Positives in Remaining Data", step_metric="Boundary")
        wandb.define_metric("Number of True Positives in Remaining Data (Ratio)", step_metric="Boundary")

        log_data = [[a, b, c, d, e, f, g, h, i, j] for a, b, c, d, e, f, g, h, i, j in
                    zip(boundaries, yield_result,
                        num_labeled_required_train, [x / num_orig_train for x in num_labeled_required_train],
                        num_labeled_required, [x / (num_orig_train + num_orig_test) for x in num_labeled_required],
                        num_points_remained, [x / num_orig_test for x in num_points_remained],
                        true_positive_remained, true_positive_remained_ratio)]
        for lis in log_data:
            log_dict = {"Boundary": lis[0], "Yields": lis[1],
                        "Number of Labels Required for Training": lis[2],
                        "Number of Labels Required for Training (Ratio)": lis[3],
                        "Number of Labels Required": lis[4], "Number of Labels Required (Ratio)": lis[5],
                        "Number of Remaining Data": lis[6], "Number of Remaining Data (Ratio)": lis[7],
                        "Number of True Positives in Remaining Data": lis[8],
                        "Number of True Positives in Remaining Data (Ratio)": lis[9]}
            wandb.log(log_dict)

        # log_data = [[final_boundary]]
        # columns = ["Final boundary"]
        # table = wandb.Table(data=log_data, columns=columns)
        # wandb.log({"final-boundary": table})
        #
        # if len(model_series) >= boi_start:
        #     wandb.define_metric("Boundary")
        #
        #     log_data = [[x, y] for x, y in zip(boundaries[boi_start - 1:boi_end], yield_result[boi_start - 1:boi_end])]
        #     table = wandb.Table(data=log_data, columns=["Boundary", "Yields"])
        #     wandb.log({"yields-bar": wandb.plot.bar(table, "Boundary", "Yields", title="Yields")})
        #     wandb.define_metric("Yields", step_metric="Boundary")
        #     for lis in log_data:
        #         log_dict = {"Boundary": lis[0], "Yields": lis[1]}
        #         wandb.log(log_dict)
        #
        #     log_data = [[x, y] for x, y in zip(boundaries[boi_start - 1:boi_end],
        #                                        num_labeled_required[boi_start - 1:boi_end])]
        #     table = wandb.Table(data=log_data, columns=["Boundary", "Number of labels required"])
        #     wandb.log({"num-labels-bar": wandb.plot.bar(table, "Boundary", "Number of labels required",
        #                                                 title="Number of Labels Required")})
        #     wandb.define_metric("Number of Labels Required", step_metric="Boundary")
        #     for lis in log_data:
        #         log_dict = {"Boundary": lis[0], "Number of Labels Required": lis[1]}
        #         wandb.log(log_dict)
        #
        #     log_data = [[x, y] for x, y in zip(boundaries[boi_start - 1:boi_end], num_points_left[boi_start - 1:boi_end])]
        #     table = wandb.Table(data=log_data, columns=["Boundary", "Number of remaining data"])
        #     wandb.log({"num-data-left-bar": wandb.plot.bar(table, "Boundary", "Number of Remaining Data",
        #                                                    title="Number of Remaining Data")})
        #     wandb.define_metric("Number of Remaining Data", step_metric="Boundary")
        #     for lis in log_data:
        #         log_dict = {"Boundary": lis[0], "Number of Remaining Data": lis[1]}
        #         wandb.log(log_dict)

        wandb_run.finish()


if __name__ == "__main__":
    main()

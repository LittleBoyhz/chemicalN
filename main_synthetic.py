import argparse
import copy
import decimal
import random

import numpy as np
from scipy.spatial import distance
import sklearn.linear_model as sklearn_linear_model
import sklearn.preprocessing as sklearn_preprocessing
import sklearn.svm as sklearn_svm
import wandb

from dataset_container import *
from sampling_methods import *
import utils


def parse_input():
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset", type=str, choices={"gaussian_1", "gaussian_2"},
                        help="Name of the dataset.")
    parser.add_argument("model", type=str, choices={"svm"},
                        help="Name of the model")
    parser.add_argument("active_learning", type=str, choices={"dist_to_boundary", "qbc", "uniform"},
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
        self.random_seed = 100

        self.dataset = "gaussian_1"
        self.dataset_path = "./datasets/synthetic"

        self.model = "svm"

        self.active_learning = "dist_to_boundary"
        self.active_epoch = 50
        self.budget_round = 5  # budget in each active learning round, real values indicate a percentage
        self.init_num_labeled = 5  # number of initial labeled data, real values indicate a percentage

        self.num_models = 25
        self.num_samples = 0.7

        self.boundaries = [0.5, 0.75, 0.875, 0.9375, 0.96875, 0.984375, 0.9921875]


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

    boi = 3
    num_final_sample = 200

    # set random seeds
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    # load dataset
    original_dataset = get_dataset(args.dataset, args.dataset_path)
    (orig_x_train, orig_y_train), (orig_x_test, orig_y_test), (orig_x_pred, orig_y_pred) = original_dataset

    # shuffle the dataset
    random_permutation = np.random.default_rng().permutation(orig_x_train.shape[0])
    orig_x_train = orig_x_train[random_permutation, :]
    orig_y_train = orig_y_train[random_permutation]

    al_init_kwargs = dict()  # arguments for initializing an active learning algorithm
    if args.active_learning == "dist_to_boundary":
        al_class = DistToBoundary
    elif args.active_learning == "margin":
        al_class = Margin
    elif args.active_learning == "qbc":
        al_class = QBC
        al_init_kwargs["num_models"] = args.num_models
        al_init_kwargs["num_samples"] = args.num_samples
    elif args.active_learning == "uniform":
        al_class = UniformSampling
    else:
        raise ValueError("Unsupported active learning algorithm: {}".format(args.active_learning))

    model_init_kwargs = dict()
    if args.model == "logistic_regression":
        model_class = sklearn_linear_model.LogisticRegression
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
            predictions = predict(previous_x_train, model_series[boundary_idx - 1], preprocessor_series[boundary_idx - 1])
            condition = (predictions == 1)
            if condition.sum() == 0:  # the previously trained model was not well trained
                unlabeled_x_train = previous_x_train
                unlabeled_y_train = previous_y_train
            else:
                unlabeled_x_train = previous_x_train[condition, :]
                unlabeled_y_train = previous_y_train[condition]

            condition = (orig_y_test > boundaries[boundary_idx - 1])  # filter test data
            x_test = orig_x_test[condition, :]
            y_test = orig_y_test[condition]
        else:
            unlabeled_x_train = orig_x_train
            unlabeled_y_train = orig_y_train
            x_test = orig_x_test
            y_test = orig_y_test

        # wandb_run = wandb.init(entity="opstreadstone", project="chemical",
        #                        job_type=args.dataset + "-" + str(boundary_value) + "-" + args.model,
        #                        group=args.active_learning, name=args.active_learning + "-seed_" + str(args.random_seed),
        #                        reinit=True)
        # wandb.define_metric("Number of labeled data")
        # wandb.define_metric("Test accuracy", step_metric="Number of labeled data")

        fail = 0
        active_dataset = ActiveDataset(unlabeled_x_train, unlabeled_y_train, args.init_num_labeled)
        num_init = 0
        for active_iter in range(args.active_epoch + 1):
            proportion = decimal.Decimal(str(active_dataset.num_labeled)) / \
                         decimal.Decimal(str(unlabeled_x_train.shape[0]))
            print("    Number of labeled samples: {}/{}, {:.1f}%".format(active_dataset.num_labeled,
                                                                         unlabeled_x_train.shape[0],
                                                                         proportion * decimal.Decimal("100.0")))

            model = model_class(**model_init_kwargs)
            preprocessor = preprocessor_class()

            labeled_idx = active_dataset.index["labeled"]  # indices of labeled data
            # train the model using labeled data
            accuracy_train = train(active_dataset.orig_x_train[labeled_idx, ...],
                                   active_dataset.orig_y_train[labeled_idx, ...], model, preprocessor, boundary_value)
            # print("        Training accuracy: {}%".format(accuracy_train * 100.0))
            while accuracy_train == -1:  # active_iter must be equal to 0
                fail += 1
                num_init += active_dataset.num_labeled
                if fail == 10:
                    break

                active_dataset = ActiveDataset(unlabeled_x_train, unlabeled_y_train, args.init_num_labeled)
                model = model_class(**model_init_kwargs)
                preprocessor = preprocessor_class()

                labeled_idx = active_dataset.index["labeled"]  # indices of labeled data
                # train the model using labeled data
                accuracy_train = train(active_dataset.orig_x_train[labeled_idx, ...],
                                       active_dataset.orig_y_train[labeled_idx, ...], model, preprocessor,
                                       boundary_value)

            if fail == 10:
                break

            num_labeled = active_dataset.num_labeled

            # test the trained model
            accuracy_test = test(x_test, y_test, model, preprocessor, boundary_value)
            print("        Test accuracy: {}%".format(accuracy_test * 100.0))

            # log_dict = {
            #     "Number of labeled data": labeled_idx.shape[0],
            #     "Test accuracy": accuracy_test
            # }
            # wandb.log(log_dict)

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

        if fail == 10:
            print("    Training failed")
            break
        else:
            model_series.append(copy.deepcopy(model))
            preprocessor_series.append(copy.deepcopy(preprocessor))
            if boundary_idx == 0:
                num_labeled_required_train.append(num_init + num_labeled)
            else:
                num_labeled_required_train.append(num_init + num_labeled + num_labeled_required_train[boundary_idx - 1])
            previous_x_train = unlabeled_x_train
            previous_y_train = unlabeled_y_train
    print("num_labeled_required_train: {}".format(num_labeled_required_train))
    print("")

    if len(model_series) > 1:
        final_boundary = boundaries[len(model_series) - 1]
        num_points_left = []
        dist_to_max_all = []
        dist_to_max_sample = []
        true_value_all = []
        true_value_sample = []
        num_labeled_required_all = []
        num_labeled_required_sample = []

        to_pred_x = orig_x_pred
        to_pred_y = orig_y_pred
        for i in range(len(model_series)):
            print("Doing calculation for {}".format(args.boundaries[i]))
            model = model_series[i]
            preprocessor = preprocessor_series[i]
            predictions = model.predict(preprocessor.transform(to_pred_x))
            condition = (predictions == 1)
            to_pred_x = to_pred_x[condition, :]
            to_pred_y = to_pred_y[condition]

            num_points_left.append(to_pred_x.shape[0])

            dist = distance.cdist(to_pred_x, np.array([[1, 2, 3, 4, 5]]))
            idx = np.argmin(dist, axis=0)
            dist_to_max_all.append(dist[idx, 0].item())
            true_value_all.append(to_pred_y[idx].item())
            num_labeled_required_all.append(num_labeled_required_train[i] + to_pred_x.shape[0])

            if to_pred_x.shape[0] <= num_final_sample:
                dist_to_max_sample.append(dist_to_max_all[i])
                true_value_sample.append(true_value_all[i])
                num_labeled_required_sample.append(num_labeled_required_all[i])
            else:
                sampled_idx = np.random.permutation(to_pred_x.shape[0])[:num_final_sample]
                dist = distance.cdist(to_pred_x[sampled_idx, ...], np.array([[1, 2, 3, 4, 5]]))
                idx = np.argmin(dist, axis=0)
                dist_to_max_sample.append(dist[idx, 0].item())
                true_value_sample.append(to_pred_y[sampled_idx][idx].item())
                num_labeled_required_sample.append(num_labeled_required_train[i] + num_final_sample)

        print("final_boundary: {}".format(final_boundary))
        print("num_points_left: {}".format(num_points_left))
        print("dist_to_max_all: {}".format(dist_to_max_all))
        print("dist_to_max_sample: {}".format(dist_to_max_sample))
        print("true_value_all: {}".format(true_value_all))
        print("true_value_sample: {}".format(true_value_sample))
        print("num_labeled_required_all: {}".format(num_labeled_required_all))
        print("num_labeled_required_sample: {}".format(num_labeled_required_sample))

        if args.init_num_labeled < 1.0:
            init_num_labeled = round(args.init_num_labeled * orig_x_train.shape[0])
        else:
            init_num_labeled = round(args.init_num_labeled)
        if args.budget_round < 1.0:  # convert to int
            budget = round(args.budget_round * orig_x_train.shape[0])
        else:
            budget = round(args.budget_round)
        # prepare to use Weights & Bias
        wandb_run = wandb.init(entity="opstreadstone", project="chemical", job_type=args.dataset + "-" + args.model,
                               group=args.active_learning + "-" + str(init_num_labeled) + "-" + str(budget) + "-" +
                                     str(args.active_epoch),
                               name=args.active_learning + "-" + str(init_num_labeled) + "-" + str(budget) + "-" +
                                    str(args.active_epoch) + "-seed_" + str(args.random_seed),
                               reinit=True)

        log_data = [[final_boundary]]
        columns = ["Final boundary"]
        table = wandb.Table(data=log_data, columns=columns)
        wandb.log({"final-boundary": table})

        if len(model_series) >= boi:
            wandb.define_metric("Boundary")

            log_data = [[x, y] for x, y in zip(boundaries[boi - 1:], dist_to_max_all[boi - 1:])]
            table = wandb.Table(data=log_data, columns=["Boundary", "Distance to maximum"])
            wandb.log({"distance-all-bar": wandb.plot.bar(table, "Boundary", "Distance to maximum", title="Distance (all)")})
            wandb.define_metric("Distance to maximum (all)", step_metric="Boundary")
            for lis in log_data:
                log_dict = {"Boundary": lis[0], "Distance to maximum (all)": lis[1]}
                wandb.log(log_dict)

            log_data = [[x, y] for x, y in zip(boundaries[boi - 1:], dist_to_max_sample[boi - 1:])]
            table = wandb.Table(data=log_data, columns=["Boundary", "Distance to maximum"])
            wandb.log({"distance-sample-bar": wandb.plot.bar(table, "Boundary", "Distance to maximum", title="Distance (sample)")})
            wandb.define_metric("Distance to maximum (sample)", step_metric="Boundary")
            for lis in log_data:
                log_dict = {"Boundary": lis[0], "Distance to maximum (sample)": lis[1]}
                wandb.log(log_dict)
            # wandb.log({"distance-sample-line": wandb.plot.line(table, "Boundary", "Distance to maximum", title="Distance (sample)")})

            log_data = [[x, y] for x, y in zip(boundaries[boi - 1:], true_value_all[boi - 1:])]
            table = wandb.Table(data=log_data, columns=["Boundary", "Value"])
            wandb.log({"true-value-all-bar": wandb.plot.bar(table, "Boundary", "Value", title="True Value (all)")})
            wandb.define_metric("True Value (all)", step_metric="Boundary")
            for lis in log_data:
                log_dict = {"Boundary": lis[0], "True Value (all)": lis[1]}
                wandb.log(log_dict)
            # wandb.log({"true-value-all-line": wandb.plot.line(table, "Boundary", "Value", title="True Value (all)")})

            log_data = [[x, y] for x, y in zip(boundaries[boi - 1:], true_value_sample[boi - 1:])]
            table = wandb.Table(data=log_data, columns=["Boundary", "Value"])
            wandb.log({"true-value-sample-bar": wandb.plot.bar(table, "Boundary", "Value", title="True Value (sample)")})
            wandb.define_metric("True Value (sample)", step_metric="Boundary")
            for lis in log_data:
                log_dict = {"Boundary": lis[0], "True Value (sample)": lis[1]}
                wandb.log(log_dict)
            # wandb.log({"true-value-sample-line": wandb.plot.line(table, "Boundary", "Value", title="True Value (sample)")})

            log_data = [[x, y] for x, y in zip(boundaries[boi - 1:], num_labeled_required_all[boi - 1:])]
            table = wandb.Table(data=log_data, columns=["Boundary", "Number of labels required"])
            wandb.log({"num-labels-all-bar": wandb.plot.bar(table, "Boundary", "Number of labels required",
                                                            title="Number of Labels (all)")})
            wandb.define_metric("Number of Labels (all)", step_metric="Boundary")
            for lis in log_data:
                log_dict = {"Boundary": lis[0], "Number of Labels (all)": lis[1]}
                wandb.log(log_dict)
            # wandb.log({"num-labels-all-line": wandb.plot.line(table, "Boundary", "Number of labels required",
            #                                                   title="Number of Labels (all)")})

            log_data = [[x, y] for x, y in zip(boundaries[boi - 1:], num_labeled_required_sample[boi - 1:])]
            table = wandb.Table(data=log_data, columns=["Boundary", "Number of labels required"])
            wandb.log({"num-labels-sample-bar": wandb.plot.bar(table, "Boundary", "Number of labels required",
                                                               title="Number of Labels (sample)")})
            wandb.define_metric("Number of Labels (sample)", step_metric="Boundary")
            for lis in log_data:
                log_dict = {"Boundary": lis[0], "Number of Labels (sample)": lis[1]}
                wandb.log(log_dict)
            # wandb.log({"num-labels-sample-line": wandb.plot.line(table, "Boundary", "Number of labels required",
            #                                                      title="Number of Labels (sample)")})

        wandb_run.finish()

    # x = np.array([[1, 2, 3, 4, 5]])
    # for i in range(len(model_series)):
    #     model = model_series[i]
    #     preprocessor = preprocessor_series[i]
    #     print(model.predict(preprocessor.transform(x)), end=", ")
    # print(" ")
    #
    # x = np.array([[-1, 2, -3, 4, 5]])
    # for i in range(len(model_series)):
    #     model = model_series[i]
    #     preprocessor = preprocessor_series[i]
    #     print(model.predict(preprocessor.transform(x)), end=", ")
    # print(" ")
    #
    # x = np.array([[1, -2, 3, -4, -5]])
    # for i in range(len(model_series)):
    #     model = model_series[i]
    #     preprocessor = preprocessor_series[i]
    #     print(model.predict(preprocessor.transform(x)), end=", ")
    # print(" ")


if __name__ == "__main__":
    main()

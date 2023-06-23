import os

import numpy as np
from scipy.spatial import distance
import scipy.stats as scipy_stats


def get_gaussian(mean, cov, num_data, data_range, pred):
    dim_data = mean.shape[0]  # dimension of data
    if pred:
        p = []
        for i in range(dim_data):
            a = np.linspace(data_range[i, 0], data_range[i, 1], num=num_data)
            p.append(a)
        x = np.array(np.meshgrid(*p)).T.reshape(-1, dim_data)
    else:
        x = np.zeros((num_data, dim_data))
        for i in range(dim_data):
            # the values of the i-th dimension of x are uniformly drawn over the interval [data_range[i,0], data_range[i,1])
            x[:, i] = np.random.default_rng().uniform(low=data_range[i, 0], high=data_range[i, 1], size=(num_data,))
    # compute the values using a Gaussian pdf and normalize them to (0, 1]
    y = scipy_stats.multivariate_normal.pdf(x, mean=mean, cov=cov) / \
        scipy_stats.multivariate_normal.pdf(mean, mean=mean, cov=cov)
    return x, y


def generate_syn_data(mean_, cov_, num_data, boundary, nums, data_range, num_test_data_per_class):
    dim_data = mean_.shape[0]

    train_data = np.zeros((num_data, dim_data))
    train_labels = np.zeros((num_data,))
    test_data = np.empty(shape=(0, dim_data))
    test_labels = np.empty(shape=(0,))

    right = 0
    for i in range(len(boundary) - 1):  # generate data points with labels within the half interval (boundary[i], boundary[i+1]]
        x, y = get_gaussian(mean_, cov_, num_data, data_range[i], False)

        condition = ((y > boundary[i]) & (y <= boundary[i + 1]))
        n = nums[i] - nums[i + 1]
        print("Number of points in ({}, {}]: {}".format(boundary[i], boundary[i + 1], condition.sum()))
        assert (condition.sum() >= n + num_test_data_per_class)

        left = right
        right += n

        train_data[left:right, :] = x[condition, :][:n, :]
        train_labels[left:right] = y[condition][:n]
        test_data = np.concatenate((test_data, x[condition, :][n:n + num_test_data_per_class, :]), axis=0)
        test_labels = np.concatenate((test_labels, y[condition][n:n + num_test_data_per_class]), axis=0)
        print("    Training data: {}, labels: {}, unique: {}".format(x[condition, :][:n, :].shape,
                                                                     y[condition][:n].shape,
                                                                     np.unique(y[condition][:n]).shape))
        print("    Test data: {}, labels: {}, unique: {}".format(x[condition, :][n:n + num_test_data_per_class, :].shape,
                                                                 y[condition][n:n + num_test_data_per_class].shape,
                                                                 np.unique(y[condition][n:n + num_test_data_per_class]).shape))

    print("Is train data unique: {}".format(np.unique(train_labels).shape[0] == train_data.shape[0]))
    print("Is test data unique: {}".format(np.unique(test_labels).shape[0] == test_data.shape[0]))

    print("Final verification:")
    for i in range(len(boundary) - 1):
        print("({}, {}]:".format(boundary[i], boundary[i + 1]))

        condition = ((train_labels > boundary[i]) & (train_labels <= boundary[i + 1]))
        print("    Training data: {}, is_unique: {}".format(condition.sum(),
                                                            np.unique(train_labels[condition]).shape[
                                                                0] == condition.sum()))

        condition = ((test_labels > boundary[i]) & (test_labels <= boundary[i + 1]))
        print("    Test data: {}, is_unique: {}".format(condition.sum(),
                                                        np.unique(test_labels[condition]).shape[0] == condition.sum()))

    pred_data, pred_labels = get_gaussian(mean_, cov_, 10, data_range[0], True)

    return (train_data, train_labels), (test_data, test_labels), (pred_data, pred_labels)


def generate_gaussian_1():
    """Generates synthetic data that has only one peak value.
    """
    dataset_name = "gaussian_1"
    data_path = "../datasets/synthetic/" + dataset_name
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    mean_ = np.array([1, 2, 3, 4, 5])
    cov_ = np.array([[1, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0],
                     [0, 0, 1, 0, 0],
                     [0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 1]])
    num_data = 100000  # number of data points to generate

    # the number of data, which has labels in the half interval (boundary[i], boundary[i+1]], is nums[i]-nums[i+1]
    boundary = [0, 0.5, 0.75, 0.875, 0.9375, 0.96875, 0.984375, 0.9921875, 1]
    nums = [num_data, 50000, 20000, 8000, 3000, 1000, 300, 75, 0]

    num_test_data_per_class = 1000  # number of test data of each class

    # the values of the i-th dimension of data are uniformly drawn over the interval [data_range[i,0], data_range[i,1])
    data_range = dict()
    data_range[0] = np.array([[0.4, 1.3], [1.5, 2.3], [2.5, 4.2], [2.3, 4.5], [4.5, 6.5]])
    data_range[1] = np.array([[0.2, 1.3], [1.1, 2.3], [2.5, 3.5], [2.3, 4.5], [4.3, 6.2]])
    data_range[2] = np.array([[0.4, 1.3], [1.5, 2.3], [2.5, 4.2], [2.3, 4.5], [4.5, 5.5]])
    data_range[3] = np.array([[0.4, 1.3], [1.5, 2.3], [2.5, 4.2], [3, 4.5], [4.5, 5.5]])
    data_range[4] = np.array([[0.6, 1.3], [1.5, 2.3], [2.5, 3.5], [3, 4.5], [4.5, 5.5]])
    data_range[5] = np.array([[0.6, 1.3], [1.5, 2.3], [2.5, 3.2], [3.5, 4.2], [4.8, 5.2]])
    data_range[6] = np.array([[0.8, 1.3], [1.8, 2.3], [2.8, 3.2], [3.8, 4.2], [4.8, 5.2]])
    data_range[7] = np.array([[0.9, 1.1], [1.9, 2.1], [2.9, 3.1], [3.9, 4.1], [4.9, 5.1]])

    (train_data, train_labels), (test_data, test_labels), (pred_data, pred_labels) = \
        generate_syn_data(mean_, cov_, num_data, boundary, nums, data_range, num_test_data_per_class)

    # np.savez(os.path.join(data_path, dataset_name + "_train.npz"), train_data=train_data, train_labels=train_labels)
    # np.savez(os.path.join(data_path, dataset_name + "_test.npz"), test_data=test_data, test_labels=test_labels)
    np.savez(os.path.join(data_path, dataset_name + "_pred.npz"), pred_data=pred_data, pred_labels=pred_labels)


def generate_gaussian_2():
    """Generates synthetic data that has three peak values.
    """
    dataset_name = "gaussian_2"
    data_path = "../datasets/synthetic/" + dataset_name
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # first
    print("1st:")
    mean_ = np.array([1, 2, 3, 4, 5])
    cov_ = np.array([[1, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0],
                     [0, 0, 1, 0, 0],
                     [0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 1]])
    num_data = 40000
    boundary = [0, 0.5, 0.75, 0.875, 0.9375, 0.96875, 0.984375, 0.9921875, 1]
    nums = [num_data, 20000, 8000, 3200, 1200, 400, 120, 30, 0]
    num_test_data_per_class = 400
    data_range = dict()
    data_range[0] = np.array([[0.4, 1.3], [1.5, 2.3], [2.5, 4.2], [2.3, 4.5], [4.5, 6.5]])
    data_range[1] = np.array([[0.2, 1.3], [1.1, 2.3], [2.5, 3.5], [2.3, 4.5], [4.3, 6.2]])
    data_range[2] = np.array([[0.4, 1.3], [1.5, 2.3], [2.5, 4.2], [2.3, 4.5], [4.5, 5.5]])
    data_range[3] = np.array([[0.4, 1.3], [1.5, 2.3], [2.5, 4.2], [3, 4.5], [4.5, 5.5]])
    data_range[4] = np.array([[0.6, 1.3], [1.5, 2.3], [2.5, 3.5], [3, 4.5], [4.5, 5.5]])
    data_range[5] = np.array([[0.6, 1.3], [1.5, 2.3], [2.5, 3.2], [3.5, 4.2], [4.8, 5.2]])
    data_range[6] = np.array([[0.8, 1.3], [1.8, 2.3], [2.8, 3.2], [3.8, 4.2], [4.8, 5.2]])
    data_range[7] = np.array([[0.9, 1.1], [1.9, 2.1], [2.9, 3.1], [3.9, 4.1], [4.9, 5.1]])
    (train_data_1, train_labels_1), (test_data_1, test_labels_1) = generate_syn_data(mean_, cov_, num_data, boundary,
                                                                                     nums, data_range,
                                                                                     num_test_data_per_class)
    print(" ")

    # second
    print("2nd")
    mean_ = np.array([-1, 2, -3, 4, 5])
    cov_ = np.array([[1, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0],
                     [0, 0, 1, 0, 0],
                     [0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 1]])
    num_data = 40000
    boundary = [0, 0.5, 0.75, 0.875, 0.9375, 0.96875, 0.984375, 0.9921875, 1]
    nums = [num_data, 20000, 8000, 3200, 1200, 400, 120, 30, 0]
    num_test_data_per_class = 400
    data_range = dict()
    data_range[0] = np.array([[-1.6, -0.7], [1.5, 2.3], [-3.5, -1.8], [2.3, 4.5], [4.5, 6.5]])
    data_range[1] = np.array([[-1.8, -0.7], [1.1, 2.3], [-3.5, -2.5], [2.3, 4.5], [4.3, 6.2]])
    data_range[2] = np.array([[-1.6, -0.7], [1.5, 2.3], [-3.5, -1.8], [2.3, 4.5], [4.5, 5.5]])
    data_range[3] = np.array([[-1.6, -0.7], [1.5, 2.3], [-3.5, -1.8], [3, 4.5], [4.5, 5.5]])
    data_range[4] = np.array([[-1.4, -0.7], [1.5, 2.3], [-3.5, -2.5], [3, 4.5], [4.5, 5.5]])
    data_range[5] = np.array([[-1.4, -0.7], [1.5, 2.3], [-3.5, -2.8], [3.5, 4.2], [4.8, 5.2]])
    data_range[6] = np.array([[-1.2, -0.7], [1.8, 2.3], [-3.2, -2.8], [3.8, 4.2], [4.8, 5.2]])
    data_range[7] = np.array([[-1.1, -0.9], [1.9, 2.1], [-3.1, -2.9], [3.9, 4.1], [4.9, 5.1]])
    (train_data_2, train_labels_2), (test_data_2, test_labels_2) = generate_syn_data(mean_, cov_, num_data, boundary,
                                                                                     nums, data_range,
                                                                                     num_test_data_per_class)
    print(" ")

    # third
    print("3rd:")
    mean_ = np.array([1, -2, 3, -4, -5])
    cov_ = np.array([[1, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0],
                     [0, 0, 1, 0, 0],
                     [0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 1]])
    num_data = 40000
    boundary = [0, 0.5, 0.75, 0.875, 0.9375, 0.96875, 0.984375, 0.9921875, 1]
    nums = [num_data, 20000, 8000, 3200, 1200, 400, 120, 30, 0]
    num_test_data_per_class = 400
    data_range = dict()
    data_range[0] = np.array([[0.4, 1.3], [-2.5, -1.7], [2.5, 4.2], [-5.7, -3.5], [-5.5, -3.5]])
    data_range[1] = np.array([[0.2, 1.3], [-2.9, -1.7], [2.5, 3.5], [-5.7, -3.5], [-5.7, -3.8]])
    data_range[2] = np.array([[0.4, 1.3], [-2.5, -1.7], [2.5, 4.2], [-5.7, -3.5], [-5.5, -4.5]])
    data_range[3] = np.array([[0.4, 1.3], [-2.5, -1.7], [2.5, 4.2], [-5, -3.5], [-5.5, -4.5]])
    data_range[4] = np.array([[0.6, 1.3], [-2.5, -1.7], [2.5, 3.5], [-5, -3.5], [-5.5, -4.5]])
    data_range[5] = np.array([[0.6, 1.3], [-2.5, -1.7], [2.5, 3.2], [-4.5, -3.8], [-5.2, -4.8]])
    data_range[6] = np.array([[0.8, 1.3], [-2.2, -1.7], [2.8, 3.2], [-4.2, -3.8], [-5.2, -4.8]])
    data_range[7] = np.array([[0.9, 1.1], [-2.1, -1.9], [2.9, 3.1], [-4.1, -3.9], [-5.1, -4.9]])
    (train_data_3, train_labels_3), (test_data_3, test_labels_3) = generate_syn_data(mean_, cov_, num_data, boundary,
                                                                                     nums, data_range,
                                                                                     num_test_data_per_class)
    print(" ")

    assert(distance.cdist(train_data_1, train_data_2).min() > 0)
    assert(distance.cdist(train_data_1, train_data_3).min() > 0)
    assert(distance.cdist(train_data_2, train_data_3).min() > 0)

    assert(distance.cdist(test_data_1, test_data_2).min() > 0)
    assert(distance.cdist(test_data_1, test_data_3).min() > 0)
    assert(distance.cdist(test_data_2, test_data_3).min() > 0)

    train_data = np.concatenate((train_data_1, train_data_2, train_data_3), axis=0)
    train_labels = np.concatenate((train_labels_1, train_labels_2, train_labels_3), axis=0)
    test_data = np.concatenate((test_data_1, test_data_2, test_data_3), axis=0)
    test_labels = np.concatenate((test_labels_1, test_labels_2, test_labels_3), axis=0)
    print(train_data.shape)
    print(train_labels.shape)
    print(test_data.shape)
    print(test_labels.shape)

    np.savez(os.path.join(data_path, dataset_name + "_train.npz"), train_data=train_data, train_labels=train_labels)
    np.savez(os.path.join(data_path, dataset_name + "_test.npz"), test_data=test_data, test_labels=test_labels)


if __name__ == "__main__":
    generate_gaussian_1()

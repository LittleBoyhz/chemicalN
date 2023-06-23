import numpy as np


def real_to_binary(labels, threshold):
    """Converts real-value labels to binary values
    """
    binary_labels = np.where(labels <= threshold, 0, 1)
    return binary_labels


def split_dataset(x, y, ratio_train=0.8):
    num_data = x.shape[0]
    num_train = round(num_data * ratio_train)

    random_permutation = np.random.default_rng().permutation(num_data)
    x = x[random_permutation, ...]
    y = y[random_permutation, ...]

    train_data, train_labels = x[:num_train, ...], y[:num_train, ...]
    test_data, test_labels = x[num_train:, ...], y[num_train:, ...]

    print(train_labels.min())
    print(train_labels.max())
    print(test_labels.min())
    print(test_labels.max())

    return (train_data, train_labels), (test_data, test_labels)

def getNewModel(model_class,model_init_kwargs,preprocessor_class,boundarys,train,labels,train_x,train_y):
    models = []
    preprocessors = []
    for boundary in boundarys:
        model = model_class(**model_init_kwargs)
        preprocessor = preprocessor_class()
        train(train_x[labels, ...],train_y[labels, ...],model,preprocessor,boundary)
        models.append(model)
        preprocessors.append(preprocessor)
    return models,preprocessors

def getNewModel2(model_class,model_init_kwargs,preprocessor_class,boundarys,train,train_x,train_y):
    models = []
    preprocessors = []
    for boundary in boundarys:
        model = model_class(**model_init_kwargs)
        preprocessor = preprocessor_class()
        train(train_x,train_y,model,preprocessor,boundary)
        models.append(model)
        preprocessors.append(preprocessor)
    return models,preprocessors
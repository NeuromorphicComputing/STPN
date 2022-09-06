import os
from pathlib import Path
import pickle

import numpy as np
import torch


def create_all_data(
        data_path: str, train_size: int, valid_size: int, test_size: int, num_chars: int,
        keyset_size: int, valueset_size: int, separatorset_size: int,
):
    '''Generate all datasets'''
    # Sampling
    sample_x, sample_y = create_sequence(num_chars, keyset_size, valueset_size, separatorset_size)
    print("Sample:", ordinal_to_alpha([np.argmax(x) for x in sample_x]), ordinal_to_alpha([np.argmax(sample_y)]))

    # Train/valid sets
    train_x, train_y = create_data(train_size, num_chars, keyset_size, valueset_size, separatorset_size)
    print("train_x:", np.shape(train_x), ",train_y:", np.shape(train_y))
    valid_x, valid_y = create_data(valid_size, num_chars, keyset_size, valueset_size, separatorset_size)
    print("valid_x:", np.shape(valid_x), ",valid_y:", np.shape(valid_y))
    test_x, test_y = create_data(test_size, num_chars, keyset_size, valueset_size, separatorset_size)
    print("test_x:", np.shape(test_x), ",test_y:", np.shape(test_y))

    # Save data into pickle files
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    with open(os.path.join(data_path, f'train_{train_size}.p'), 'wb') as f:
        pickle.dump([train_x, train_y], f)
    with open(os.path.join(data_path, f'valid_{valid_size}.p'), 'wb') as f:
        pickle.dump([valid_x, valid_y], f)
    with open(os.path.join(data_path, f'test_{test_size}.p'), 'wb') as f:
        pickle.dump([test_x, test_y], f)


def get_all_data(
        data_path: str,
        train_size=100000, valid_size=10000, test_size=20000, num_chars=9,
        keyset_size=26, valueset_size=10, separatorset_size=1,
):
    """
    Create or load datasets
    TODO: probably better to store datasets with all/more parameters in their name as well, not only by size
    """

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    data_file = Path(os.path.join(data_path, f'train_{train_size}.p'))
    if data_file.is_file():
        print("Loading training data...")
        with open(data_file, 'rb') as f:
            [train_x, train_y] = pickle.load(f)
    else:
        print("Creating training data...")
        train_x, train_y = create_data(train_size, num_chars, keyset_size, valueset_size, separatorset_size)
        with open(os.path.join(data_path, f'train_{train_size}.p'), 'wb') as f:
            pickle.dump([train_x, train_y], f)

    data_file = Path(os.path.join(data_path, f'valid_{valid_size}.p'))
    if data_file.is_file():
        with open(data_file, 'rb') as f:
            [valid_x, valid_y] = pickle.load(f)
    else:
        valid_x, valid_y = create_data(valid_size, num_chars, keyset_size, valueset_size, separatorset_size)
        with open(os.path.join(data_path, f'valid_{valid_size}.p'), 'wb') as f:
            pickle.dump([valid_x, valid_y], f)

    data_file = Path(os.path.join(data_path, f'test_{test_size}.p'))
    if data_file.is_file():
        with open(data_file, 'rb') as f:
            [test_x, test_y] = pickle.load(f)
    else:
        test_x, test_y = create_data(test_size, num_chars, keyset_size, valueset_size, separatorset_size)
        with open(os.path.join(data_path, f'test_{test_size}.p'), 'wb') as f:
            pickle.dump([test_x, test_y], f)

    return [train_x, train_y], [valid_x, valid_y], [test_x, test_y]


def vec2index(array, dim=0):
    return np.argmax(array, axis=dim)


def dataset2index(dataset):
    return [vec2index(dataset[0], dim=2), vec2index(dataset[1], dim=1)]


def onehot_dataset2index(dataset):
    return [dataset[0], vec2index(dataset[1], dim=1)]


def get_tensor_dataloader(dataset: [np.array, np.array], **kwargs):
    train_data = torch.from_numpy(dataset[0]).type(torch.float32)  # No .to(device), keep in CPU for now
    train_labels = torch.from_numpy(dataset[1]).type(torch.long)  # LongTensor required by NLLLoss
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, **kwargs)
    return train_dataloader


def load_data(
        batch_size, data_path: str, onehot: bool = True, train_size=100000, valid_size=10000, test_size=20000
):
    """Get dataloaders"""
    train_dataset, valid_dataset, test_dataset = get_all_data(
        data_path=data_path, train_size=train_size, valid_size=valid_size, test_size=test_size)

    if onehot is True:
        onehot_train_dataset = onehot_dataset2index(train_dataset)
        onehot_valid_dataset = onehot_dataset2index(valid_dataset)
        onehot_test_dataset = onehot_dataset2index(test_dataset)

        train_dataloader = get_tensor_dataloader(onehot_train_dataset, batch_size=batch_size, shuffle=False)
        validation_dataloader = get_tensor_dataloader(onehot_valid_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = get_tensor_dataloader(onehot_test_dataset, batch_size=batch_size, shuffle=False)
    else:
        train_dataset = dataset2index(train_dataset)
        valid_dataset = dataset2index(valid_dataset)
        test_dataset = dataset2index(test_dataset)

        train_dataloader = get_tensor_dataloader(train_dataset, batch_size=batch_size, shuffle=False)
        validation_dataloader = get_tensor_dataloader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = get_tensor_dataloader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, validation_dataloader, test_dataloader


# Adapted from https://github.com/GokuMohandas/fast-weight
# Create and handle datasets for Associative Retrieval taks, following description in https://arxiv.org/abs/1610.06258
# Extension to larger sequences, configurable parameters


def get_n_letters(n, keyset_size):
    """
    Retrieve three random letters (a-z)
    without replacement.
    """
    return np.random.choice(range(0, keyset_size), n, replace=False)


def get_n_numbers(n, keyset_size, valueset_size):
    """
    Retrieve three random numbers (0-9)
    with replacement.
    """
    return np.random.choice(range(keyset_size, keyset_size + valueset_size), n, replace=True)


def create_sequence(num_chars, keyset_size, valueset_size, separatorset_size):
    """
    Concatenate keys and values with
    ?? and one of the keys.
    Returns the input and output.
    """
    # num keys is pairs letter,number + ?? + query letter
    assert (num_chars - 3 > 0) and ((num_chars - 3) % 2 == 0)
    n_pairs = int((num_chars - 3) / 2)
    letters = get_n_letters(n=n_pairs, keyset_size=keyset_size)
    numbers = get_n_numbers(n=n_pairs, keyset_size=keyset_size, valueset_size=valueset_size)
    x = np.zeros(num_chars)
    # substract the 3 final chars ( '??' + query) and early stop, so substract one extra
    for i in range(0, num_chars - 4, 2):
        x[i] = letters[int(i / 2)]
        x[i + 1] = numbers[int(i / 2)]

    # TODO add configuration of # of separators. Probably best to change from n_chars to n_pairs as a config param
    # append '??'
    x[n_pairs * 2] = keyset_size + valueset_size
    x[n_pairs * 2 + 1] = keyset_size + valueset_size

    # last key and respective value (y)
    index = np.random.choice(range(0, n_pairs), 1, replace=False)
    x[n_pairs * 2 + 2] = letters[index]
    y = numbers[index]

    # one hot encode x and y
    x_one_hot = np.eye(keyset_size + valueset_size + separatorset_size)[np.array(x).astype('int')]
    y_one_hot = np.eye(keyset_size + valueset_size + separatorset_size)[y][0]

    return x_one_hot, y_one_hot


def ordinal_to_alpha(sequence):
    """
    Convert from ordinal to alpha-numeric representations.
    Just for funsies :)
    """
    corpus = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
              'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
              0, 1, 2, 3, 4, 5, 6, 7, 8, 9, '?']

    conversion = ""
    for item in sequence:
        conversion += str(corpus[int(item)])
    return conversion


def create_data(num_samples, num_chars, keyset_size, valueset_size, separatorset_size):
    """
    Create a num_samples long set of x and y.
    """
    x = np.zeros([num_samples, num_chars, keyset_size + valueset_size + separatorset_size], dtype=np.int32)
    y = np.zeros([num_samples, keyset_size + valueset_size + separatorset_size], dtype=np.int32)
    for i in range(num_samples):
        x[i], y[i] = create_sequence(num_chars, keyset_size, valueset_size, separatorset_size)
    return x, y


def generate_epoch(x, y, num_epochs, batch_size):
    for epoch_num in range(num_epochs):
        yield generate_batch(x, y, batch_size)


def generate_batch(x, y, batch_size):
    data_size = len(x)
    num_batches = (data_size // batch_size)
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield x[start_index:end_index], y[start_index:end_index]

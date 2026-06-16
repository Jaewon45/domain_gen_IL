# Code adopted from https://github.com/facebookresearch/DomainBed/blob/main/domainbed/datasets.py

import torch
from torch.utils.data import TensorDataset
from torchvision.datasets import MNIST


def torch_bernoulli_(p, size):
    return (torch.rand(size) < p).float()


def torch_xor_(a, b):
    return (a - b).abs()


def _subsample_tensor_dataset(dataset, size, mode="random"):
    if size is None:
        return dataset

    dataset_size = len(dataset)
    if size <= 0:
        raise ValueError(f"Requested training environment size must be positive, got {size}.")
    if size > dataset_size:
        raise ValueError(
            f"Requested training environment size {size} exceeds available samples {dataset_size}."
        )
    if size == dataset_size:
        return dataset

    if mode == "random":
        indices = torch.randperm(dataset_size)[:size]
    elif mode == "first":
        indices = torch.arange(size)
    else:
        raise ValueError(f"Unknown train environment size mode: {mode}")

    x, y = dataset.tensors
    return TensorDataset(x[indices], y[indices])


def color_dataset(images, labels, environment, label_noise_rate=0.25, subsample=True, int_target=False, cuda=True):
    if subsample:
        # Subsample 2x for computational convenience
        images = images.reshape((-1, 28, 28))[:, ::2, ::2]

    # Assign a binary label based on the digit
    labels = (labels < 5).float()

    # Flip label with probability 0.25
    labels = torch_xor_(labels, torch_bernoulli_(label_noise_rate, len(labels)))

    # Assign a color based on the label; flip the color with probability e
    colors = torch_xor_(labels, torch_bernoulli_(environment, len(labels)))

    # Apply the color to the image by zeroing out the other color channel
    images = torch.stack([images, images], dim=1)
    images[torch.tensor(range(len(images))), (1 - colors).long(), :, :] *= 0

    x = images.float().div_(255.0)
    y = labels.view(-1, 1)
    if int_target:
        y = y.long()
    if cuda:
        x, y = x.cuda(), y.cuda()

    return TensorDataset(x, y)


def get_cmnist_datasets(root, train_envs=(0.1, 0.2), test_envs=(0.9,), label_noise_rate=0.25,
                        dataset_transform=color_dataset, subsample=True, int_target=False, cuda=True,
                        use_test_set=False, train_env_sizes=None, train_env_size_mode="random"):
    if root is None:
        raise ValueError('Data directory not specified!')
    if train_env_sizes is not None and len(train_env_sizes) != len(train_envs):
        raise ValueError(
            "train_env_sizes must have the same length as train_envs. "
            f"Got {len(train_env_sizes)} sizes for {len(train_envs)} environments."
        )

    orig_data_tr = MNIST(root, train=True, download=True)
    perm_inds_tr = torch.randperm(len(orig_data_tr.data))        # permute / shuffle

    if use_test_set:
        # make use of mnist test set to create test envs
        orig_data_tst = MNIST(root, train=False, download=True)
        perm_inds_tst = torch.randperm(len(orig_data_tst.data))
        train_images, train_labels = orig_data_tr.data[perm_inds_tr], orig_data_tr.targets[perm_inds_tr]
        test_images, test_labels = orig_data_tst.data[perm_inds_tst], orig_data_tst.targets[perm_inds_tst]
    else:
        # ignore mnist test set, as in original cmnist dataset
        train_images, train_labels = orig_data_tr.data[perm_inds_tr][:50000], orig_data_tr.targets[perm_inds_tr][:50000]
        test_images, test_labels = orig_data_tr.data[perm_inds_tr][50000:], orig_data_tr.targets[perm_inds_tr][50000:]

    datasets = []

    for i in range(len(train_envs)):
        # Divide original train set into non-overlapping image sets
        images = train_images[i::len(train_envs)]
        labels = train_labels[i::len(train_envs)]
        train_dataset = dataset_transform(images, labels, train_envs[i],
                                          label_noise_rate, subsample, int_target, cuda)
        if train_env_sizes is not None:
            train_dataset = _subsample_tensor_dataset(
                train_dataset,
                train_env_sizes[i],
                mode=train_env_size_mode,
            )
        datasets.append(train_dataset)

    for i in range(len(test_envs)):
        # Use entire test set for each test domain (different image transformations)
        datasets.append(dataset_transform(test_images, test_labels, test_envs[i],
                                          label_noise_rate, subsample, int_target, cuda))

    return datasets

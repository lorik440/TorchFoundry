
from importlib import import_module
from pathlib import Path
from typing import Literal

from torch.utils.data import DataLoader


def getDatasetClass(name: str ):
    """
    Get a dataset class from torchvision.datasets by name.

    Parameters
    ----------
    name : str
        Name of the dataset class to retrieve.

    Returns
    -------
    type
        The requested torchvision dataset class.

    Raises
    ------
    ValueError
        If name is not a string.
    ImportError
        If torchvision.datasets cannot be imported.
    """

    if not isinstance(name, str):
        raise ValueError("name for the dataset parameter must be string")

    try:
        datasets_module=import_module("torchvision.datasets")
        datasets_class =getattr(datasets_module, name)
    except ArithmeticError as exc:
        raise ValueError(f"Dataset '{name}' was not found in torchvision.dataset"
                         ) from exc
    except ImportError as exc:
        raise ImportError("loading this dataset requires torchvision to be installed"
                          )from exc
    return datasets_class

Selector=Literal["train","test"]

def prepareDataset(dataset, type: Selector, root, transformer):
    """
    Create a torchvision dataset for the selected split.

    Parameters
    ----------
    dataset : str
        Name of the dataset class in torchvision.datasets.
    type : {"train", "test"}
        Select the training or testing split.
    root : str, pathlib.Path, or None
        Dataset directory. If None, uses "data" without downloading.
    transformer : callable or None
        Optional transformation applied to each dataset sample.

    Returns
    -------
    Dataset
        An instance of the selected torchvision dataset class.

    Raises
    ------
    ValueError
        If type, root, or transformer has an invalid value.
    ImportError
        If torchvision.datasets cannot be imported.
    """

    if not isinstance(type, str) or type not in ("train", "test"):
        raise ValueError("type must have value 'train' or 'test'")
    if root is not None and not isinstance(root, (str, Path)):
        raise ValueError("root must be a string, pathlib.Path, or None")
    if transformer is not None and not callable(transformer):
        raise ValueError("transfrom must be callable or None")

    if root is None:
        root ="data"
        download=False
    else:
        download=True

    train = type == "train"

    dataset_class=getDatasetClass(dataset)

    dataset =dataset_class(
        root=root,
        download= download,
        train=train,
        transform=transformer
    )
    return dataset

def loadDataset(
    dataset,
    type: Selector | None = None,
    batch_size: int = 64,
    shuffle: bool | None = None,
    root: str | Path | None = None,
    transformer=None,
):
    """
    Get a FashionMNIST DataLoader.

    Parameters
    ----------
    selector : {"train", "test"} or None, default=None
        Select which dataset to return.
        None returns both DataLoaders.

    batch_size : int, default=64
        Number of samples per batch (1-128).

    Returns
    -------
    DataLoader or tuple[training_data, testing_data]
        The selected DataLoader, or both DataLoaders.
    """
    
    if not isinstance(batch_size, int) or isinstance(batch_size, bool):
        raise ValueError("batch_size must be an integer")
    if shuffle is not None and not isinstance(shuffle, bool):
        raise ValueError("shuffle must be a boolean or None")

    if shuffle is None:
        shuffle=False 

    if type =="train":

        train_data=prepareDataset(dataset, type, root, transformer)
        training_data= DataLoader(train_data,batch_size=batch_size, shuffle=shuffle)

        return training_data
    
    elif type =="test":
        
        test_data=prepareDataset(dataset, type, root, transformer)
        testing_data= DataLoader(test_data, batch_size=batch_size, shuffle=shuffle)

        return testing_data
    
    elif type is None:
        type="test"
        test_data=prepareDataset(dataset, type, root,  transformer)
        type="train"
        train_data=prepareDataset(dataset, type, root,  transformer)

        training_data= DataLoader(train_data, batch_size=batch_size, shuffle=shuffle) 
        testing_data= DataLoader(test_data, batch_size=batch_size, shuffle=shuffle)
        return training_data, testing_data

from collections.abc import Sized
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .model import loadModel, saveModel


def trainModel(model, dataloader, loss_fn, optimizer, epoch, path=None, ):
    """
    Trains a PyTorch model for the specified number of epochs and optionally
    loads weights from ``path`` when a checkpoint exists, trains the model,
    and saves the updated state dictionary to ``path``.

    Args:
        model (nn.Module): PyTorch model to train.
        dataloader (DataLoader): DataLoader providing the training batches.
        loss_fn: Loss function used to compare predictions with targets.
        optimizer: Optimizer used to update the model parameters.
        epoch (int): Number of training epochs.
        path (str | Path | None): Optional checkpoint path. Existing weights
            are loaded before training, and the updated weights are saved there.

    Returns:
        None

    Raises:
        TypeError: If model or dataloader has an invalid type.
        ValueError: If epoch or save_dict has an invalid type.
        RuntimeError: If saving the model state dictionary fails.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("module must be a Pytorch nn.Module")
    if not isinstance(dataloader,DataLoader):
        raise TypeError( "dataloader must be an instance of torch.utils.data.DataLoader")
    if not isinstance(epoch,int):
        raise ValueError("epoch must be integer")
    if epoch <0:
        raise ValueError("epoch must be positive integer")


    if torch.accelerator.is_available():
        device=torch.accelerator.current_accelerator()
    else:
        device="cpu"
    
    model = model.to(device)
    if path is not None:
        model = loadModel(model, path)

    dataset =dataloader.dataset

    if isinstance(dataset, Sized):
        size=len(dataset)
    else:
        size=0

    model.train()
    for e in range(epoch):
        print("")
        print(f"epoch {e+1}\n-----------------------------------", flush=True)
        for batch, (x,y) in enumerate(dataloader):
            x=x.to(device)
            y=y.to(device)

            pred=model(x)
            loss=loss_fn(pred,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100==0:
                loss=loss.item()
                current=(batch+1)*len(x)

                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
     
    if path is not None:
        if not isinstance(path, (str, Path)):
            raise ValueError("save_dict must be string or pathlib.Path")
        
        saveModel(model,path)

        

    return model
from pathlib import Path

import torch
from torch import nn
from torchinfo import summary

from collections.abc import Sequence

def createModel(
    layers: nn.Module | Sequence[nn.Module],
    path=None,
    show_Summary: bool = False,
    input_size: tuple[int, ...] | None=None,
):
    """
       Creates a PyTorch neural network and optionally loads pretrained weights
    and displays a model summary.

    Args:
        layers (list[nn.Module]): PyTorch layers used to construct the model.
        load_dict (str | None): Path to a saved model state dictionary.
        Summary (bool): Whether to display the model summary.
        input_size (tuple | None): Dimensions of the model input, excluding
            the batch dimension. Required if Summary is True.

    Returns:
        nn.Module: The constructed neural network.

    Raises:
        ValueError: If the path or input_size is invalid.
        FileNotFoundError: If the checkpoint file does not exist.
    """

    # model device runtime
    if torch.accelerator.is_available():
        device=torch.accelerator.current_accelerator()
    else:
        device='cpu'

    # Validate layers input
    if layers is None:
        raise ValueError("layers is a required parameter")
    
    if not isinstance(layers, (nn.Module, list, tuple)):
        raise TypeError("layers must be an nn.Module, list, or tuple of nn.Module objects")
    
    if isinstance(layers, nn.Module):
        layers = [layers]


    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()

            self.layers=nn.Sequential(*layers)

        def forward(self,x):
            x=self.layers(x)
            return x

    model =NeuralNetwork().to(device)

    # loading model
    print("loading model ...", flush=True)
    if path is not None:
       model= loadModel(model, path)

    # printing the model summary
    if show_Summary:
        if input_size is None:
            raise ValueError("input_size canot be None when Summary is True")
        summary(
            model,
            input_size=input_size,
            col_names=["input_size","output_size","num_params","trainable"],
            depth=1
        )
    
    return model

def  saveModel(model: nn.Module,path:str|Path):
    """
    Save a PyTorch model's state dictionary to a file.

    Args:
        model (nn.Module): PyTorch model whose state dictionary is saved.
        path (str | Path): Destination file for the state dictionary.

    Returns:
        None

    Raises:
        ValueError: If model or path is missing or has an invalid type.
        RuntimeError: If saving the state dictionary fails.
    """

    if not model:
        raise ValueError("model is a required parameter")
    if not path:
        raise ValueError("path is a required parameter")
    if not isinstance(path, (str,Path)):
        raise ValueError("path must be string or pathlib.Path")
    if not isinstance(model, nn.Module):
        raise ValueError("module must be a Pytorch nn.Module")

    try:
        torch.save(model.state_dict(), path)
        print(f"model was saved at '{path}'")
    except (OSError, RuntimeError) as e:
        raise RuntimeError(f"Failed to save model state_dict: {e}") from e
    

def loadModel(model: nn.Module, path: str|Path):
    """
    Load saved weights into a PyTorch model when the checkpoint exists.

    Args:
        model (nn.Module): Model to receive the loaded state dictionary.
        path (str | Path): Path to the saved state dictionary.

    Returns:
        nn.Module: The model with loaded weights, or the unchanged model if
            the checkpoint does not exist.

    Raises:
        ValueError: If model or path is missing or has an invalid type.
    """
    
    if not path:
        raise ValueError("path is a required parameter")
    if not isinstance(path, (str, Path)):
        raise ValueError("path must be string or pathlib.Path")
    
    if not model:
        raise ValueError("model is a required parameter")
    if not isinstance(model, nn.Module):
        raise ValueError("module must be a Pytorch nn.Module")
    
    if torch.accelerator.is_available():
        device=torch.accelerator.current_accelerator()
    else:
        device='cpu'
    
    model_path= Path(path)

    if model_path.exists():
        model.load_state_dict(
            torch.load(
                model_path,
                map_location=device,
                weights_only=True
            )
        )
        print("model loaded", flush=True)
    else:
        print("model path not found. Starting fresh")

    return model
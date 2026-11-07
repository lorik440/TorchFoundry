import torch
from torch import nn
from torch.utils.data import DataLoader

from collections.abc import Callable

def testModel(model, dataloader, loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], path ):
    """
    Evaluates a PyTorch model on a dataset and reports its average loss
    and accuracy.

    Args:
        model (nn.Module): PyTorch model to evaluate.
        dataloader (DataLoader): DataLoader providing the test batches.
        loss_fn: Loss function used to compare predictions with targets.

    Returns:
        dict[str, float]: Average loss and accuracy, with keys "loss"
            and "accuracy".

    Raises:
        TypeError: If model, dataloader, or loss_fn has an invalid type.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("module must be a Pytorch nn.Module")
    if not isinstance(dataloader, DataLoader):
        raise TypeError( "dataloader must be an instance of torch.utils.data.DataLoader")
    if not callable(loss_fn):
        raise TypeError("loss_fn must be callable")
    if not path:
        raise ValueError("path is a required parameter")
    if torch.accelerator.is_available():
        device=torch.accelerator.current_accelerator()
    else:
        device='cpu'
    

    state_dict =torch.load(
        path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()

    total_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.inference_mode():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = loss_fn(pred, y)

            batch_size = x.size(0)
            total_samples += batch_size

            total_loss += loss.item() * batch_size
            correct += (pred.argmax(dim=1) == y).sum().item()

    if total_samples == 0:
        raise ValueError("the test dataloader is empty")
    
    average_loss = total_loss / total_samples
    accuracy = correct / total_samples

    print(
        f"\nTest results:",
        "\n-----------------------------------",
        f"\nAccuracy: {accuracy * 100:.1f}%"
        f"\nAverage loss: {average_loss:.6f}",
        flush=True
    )
    return { "loss": average_loss, "accuracy": accuracy, }

    
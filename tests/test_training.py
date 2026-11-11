import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from torchfoundry.training import trainModel


def test_train_model_reports_sample_weighted_epoch_loss(capsys):
    model = nn.Linear(1, 2)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    optimizer = torch.optim.SGD(model.parameters(), lr=0)
    dataloader = DataLoader(
        TensorDataset(
            torch.tensor([[1.0], [2.0], [3.0]]),
            torch.tensor([0, 1, 1]),
        ),
        batch_size=2,
    )

    trainModel(model, dataloader, nn.CrossEntropyLoss(), optimizer, epoch=1)

    assert "Average training loss: 0.693147" in capsys.readouterr().out

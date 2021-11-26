import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional


class BaseTrainer:
    def __init__(
        self, model, optimizer, loss, num_epochs: int, ckpt: Optional[Path] = None
    ):
        self.optimizer = optimizer
        self.model = model
        self.loss = loss
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device=self.device)
        self.ckpt = ckpt
        self.num_epochs = num_epochs

    # ======================================================================================
    def run_epoch(self, train: bool, epoch: int, data_loader: DataLoader):
        total_loss = 0
        if train is True:
            self.model.train()
        else:
            self.model.eval()

        for iter_id, batch in enumerate(data_loader):
            batch[0] = batch[0].to(self.device)
            output = self.model(batch[0].float())
            # the input will be changed according to the definition of loss
            loss_value = self.loss(output, batch[1].squeeze(dim=1))
            total_loss += loss_value.float()

            if train is True:
                self.optimizer.zero_grad()
                loss_value.to(torch.float).backward()
                self.optimizer.step()
        print(f"====> Epoch {epoch} Average loss: {total_loss}")
        if epoch == self.num_epochs and self.ckpt is not None:
            torch.save(
                self.model.state_dict(),
                str(self.ckpt / f"{epoch}th_ckpt_total_loss{total_loss:.02e}.pth"),
            )

    def val(self, epoch, data_loader):
        return self.run_epoch(False, epoch, data_loader)

    def train(self, epoch, data_loader):
        return self.run_epoch(True, epoch, data_loader)

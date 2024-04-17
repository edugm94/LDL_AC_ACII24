import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics.classification import Accuracy, MulticlassF1Score


class SLL(pl.LightningModule):
    """
    PyTorch Lightning module to train the Network architecture passed as
    an argument following Single Label Distribution Learning (LDL)
    """
    def __init__(self,
                 network: nn.Module,
                 n_classes: int,
                 config: dict) -> None:
        super().__init__()
        self.save_hyperparameters(config)
        self.model = network
        self.test_acc = Accuracy(task="multiclass", num_classes=n_classes)
        self.test_f1 = MulticlassF1Score(average="macro", num_classes=n_classes)
        self.config = config
        # self.ce_fn = nn.CrossEntropyLoss()


    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        x, targets, hot_labels = batch
        logits = self(x)
        preds = F.log_softmax(logits, dim=1)
        targets = hot_labels.float()
        # targets = F.softmax(hot_labels.float(), dim=1)
        kl_loss = F.kl_div(input=preds, target=targets, reduction="batchmean")
        # pred_loss = F.cross_entropy(logits, torch.argmax(hot_labels, dim=1), reduction="mean")
        loss = kl_loss #+ pred_loss
        # loss = self.ce_fn(logits, targets)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, targets, hot_labels = batch
        logits = self(x)
        preds = F.log_softmax(logits, dim=1)
        targets = hot_labels.float()
        # targets = F.softmax(hot_labels.float(), dim=1)
        kl_loss = F.kl_div(input=preds, target=targets, reduction="batchmean")
        # pred_loss = F.cross_entropy(logits, torch.argmax(hot_labels, dim=1), reduction="mean")
        loss = kl_loss #+ pred_loss
        # loss = self.ce_fn(logits, targets)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, targets, logic_gt = batch
        targets = torch.argmax(logic_gt, dim=1)
        preds = self(x)
        preds = nn.functional.softmax(preds, dim=1)
        preds = torch.argmax(preds, dim=1)
        self.test_acc(preds, targets)
        self.test_f1(preds, targets)

    def on_test_epoch_end(self):
        self.test_acc.compute()
        self.test_f1.compute()
        self.log("accuracy", self.test_acc)
        self.log("f1", self.test_f1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.config["lr"],
                                     weight_decay=self.config["weight_decay"])

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=20,
            min_lr=1e-6)
        return [optimizer], [dict(scheduler=lr_scheduler, interval="epoch",
                                  monitor="train_loss")]

    def predict_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        return y_hat


class LDL(pl.LightningModule):
    """
    PyTorch Lightning module to train the Network architecture passed as
    an argument following Single Label Distribution Learning (LDL)
    """
    def __init__(self,
                 network: nn.Module,
                 n_classes: int,
                 config: dict) -> None:
        super().__init__()
        self.save_hyperparameters(config)
        self.model = network
        self.test_acc = Accuracy(task="multiclass", num_classes=n_classes)
        self.test_f1 = MulticlassF1Score(average="macro", num_classes=n_classes)
        self.config = config
        # self.ce_fn = nn.CrossEntropyLoss()


    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        x, targets, hot_labels = batch
        logits = self(x)
        preds = F.log_softmax(logits, dim=1)
        # targets = F.softmax(targets, dim=1)
        kl_loss = F.kl_div(input=preds, target=targets, reduction="batchmean")
        # pred_loss = F.cross_entropy(logits, torch.argmax(hot_labels, dim=1), reduction="mean")
        loss = kl_loss #+ pred_loss
        # loss = self.ce_fn(logits, targets)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, targets, hot_labels = batch
        logits = self(x)
        preds = F.log_softmax(logits, dim=1)
        # targets = F.softmax(targets, dim=1)
        kl_loss = F.kl_div(input=preds, target=targets, reduction="batchmean")
        # pred_loss = F.cross_entropy(logits, torch.argmax(hot_labels, dim=1), reduction="mean")
        loss = kl_loss #+ pred_loss
        # loss = self.ce_fn(logits, targets)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, targets, logic_gt = batch
        targets = torch.argmax(logic_gt, dim=1)
        preds = self(x)
        preds = nn.functional.softmax(preds, dim=1)
        preds = torch.argmax(preds, dim=1)
        self.test_acc(preds, targets)
        self.test_f1(preds, targets)

    def on_test_epoch_end(self):
        self.test_acc.compute()
        self.test_f1.compute()
        self.log("accuracy", self.test_acc)
        self.log("f1", self.test_f1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.config["lr"],
                                     weight_decay=self.config["weight_decay"])

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=20,
            min_lr=1e-6)
        return [optimizer], [dict(scheduler=lr_scheduler, interval="epoch",
                                  monitor="train_loss")]

    def predict_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        return y_hat


import os
import logging
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

NUM_CLASSES = 6
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
DROPOUT_P = 0.1
RESNET_OUT_DIM = 2048

losses = []

logging.basicConfig(level=logging.INFO)


class GatedFusion(nn.Module):
    def __init__(self, text_dim, image_dim, fusion_output_size):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(text_dim + image_dim, fusion_output_size),
            nn.Sigmoid()
        )
        self.output = nn.Linear(text_dim + image_dim, fusion_output_size)

    def forward(self, text, image):
        combined = torch.cat([text, image], dim=1)
        gate = self.gate(combined)
        output = self.output(combined)
        return gate * output


class JointTextImageModel(nn.Module):
    def __init__(self, num_classes, loss_fn, text_module, image_module, text_feature_dim,
                 image_feature_dim, fusion_output_size, dropout_p, hidden_size=512):
        super().__init__()
        self.text_module = text_module
        self.image_module = image_module
        self.fusion = GatedFusion(text_feature_dim, image_feature_dim, fusion_output_size)
        self.fc1 = nn.Linear(fusion_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.loss_fn = loss_fn
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, text, image, label=None):
        text_features = torch.relu(self.text_module(text))
        image_features = torch.relu(self.image_module(image))
        fused = self.dropout(torch.relu(self.fusion(text_features, image_features)))
        hidden = torch.relu(self.fc1(fused))
        logits = self.fc2(hidden)
        if label is not None:
            loss = self.loss_fn(logits, label)
            return logits, loss
        return logits


class MultimodalFakeNewsDetectionModel(pl.LightningModule):
    def __init__(self, hparams=None, use_vit=False):
        super().__init__()
        self.save_hyperparameters(ignore=['use_vit'])
        self.use_vit = use_vit

        self.embedding_dim = self.hparams.get("embedding_dim", 768)
        self.text_feature_dim = self.hparams.get("text_feature_dim", 300)
        self.image_feature_dim = self.hparams.get("image_feature_dim", self.text_feature_dim)

        self.model = self._build_model()

    def forward(self, text, image):
        return self.model(text, image)

    def training_step(self, batch, batch_idx):
        global losses
        text, image, label = batch["text"], batch["image"], batch["label"]
        pred, loss = self.model(text, image, label)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        losses.append(loss.item())
        return loss

    def test_step(self, batch, batch_idx):
        text, image, label = batch["text"], batch["image"], batch["label"]
        pred, loss = self.model(text, image, label)
        pred_label = torch.argmax(pred, dim=1)
        acc = torch.sum(pred_label == label).float() / len(label)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return {"test_loss": loss, "test_acc": acc}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["test_acc"] for x in outputs]).mean()
        self.test_results = {"test_loss": avg_loss, "test_acc": avg_acc}
        print(f"\nTest Loss: {avg_loss:.4f}, Test Accuracy: {avg_acc:.4f}")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.get("learning_rate", LEARNING_RATE))

    def _build_model(self):
        text_module = nn.Linear(self.embedding_dim, self.text_feature_dim)

        if self.use_vit:
            vit = torchvision.models.vit_b_16(pretrained=True)
            vit.heads = nn.Identity()
            image_module = nn.Sequential(
                vit,
                nn.Linear(768, self.image_feature_dim)
            )
        else:
            image_module = torchvision.models.resnet152(pretrained=True)
            image_module.fc = nn.Linear(RESNET_OUT_DIM, self.image_feature_dim)

        return JointTextImageModel(
            num_classes=self.hparams.get("num_classes", NUM_CLASSES),
            loss_fn=nn.CrossEntropyLoss(),
            text_module=text_module,
            image_module=image_module,
            text_feature_dim=self.text_feature_dim,
            image_feature_dim=self.image_feature_dim,
            fusion_output_size=self.hparams.get("fusion_output_size", 512),
            dropout_p=self.hparams.get("dropout_p", DROPOUT_P)
        )


class PrintCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training started...")

    def on_train_end(self, trainer, pl_module):
        print("Training done...")
        for loss_val in losses:
            print(loss_val)

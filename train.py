import os

import pandas as pd
import pytorch_lightning as pl
from box import Box
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import StratifiedKFold

from data import PetfinderDataModule

from model import Model

config = {
    "seed": 2021,
    "root": "./",
    "data_path": "petfinder-pawpularity-score",
    "n_splits": 5,
    "epoch": 20,
    "trainer": {
        "gpus": 1,
        "accumulate_grad_batches": 1,
        "progress_bar_refresh_rate": 1,
        "fast_dev_run": False,
        "num_sanity_val_steps": 0,
        "resume_from_checkpoint": None,
    },
    "transform": {"name": "get_default_transforms", "image_size": 224},
    "train_loader": {
        "batch_size": 64,
        "shuffle": True,
        "num_workers": 4,
        "pin_memory": False,
        "drop_last": True,
    },
    "val_loader": {
        "batch_size": 64,
        "shuffle": False,
        "num_workers": 4,
        "pin_memory": False,
        "drop_last": False,
    },
    # "model": {"name": "swin_tiny_patch4_window7_224", "output_dim": 1},
    "model": {"name": "resnet152d", "output_dim": 1},
    "optimizer": {
        "name": "optim.AdamW",
        "params": {"lr": 1e-5},
    },
    "scheduler": {
        "name": "optim.lr_scheduler.CosineAnnealingWarmRestarts",
        "params": {
            "T_0": 20,
            "eta_min": 1e-4,
        },
    },
    "loss": "nn.BCEWithLogitsLoss",
}
config = Box(config)

df = pd.read_csv(os.path.join(config.data_path, "train.csv"))
df["Id"] = df["Id"].apply(lambda x: os.path.join(config.data_path, "train", x + ".jpg"))


skf = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)

for fold, (train_idx, val_idx) in enumerate(skf.split(df["Id"], df["Pawpularity"])):
    train_df = df.loc[train_idx].reset_index(drop=True)
    val_df = df.loc[val_idx].reset_index(drop=True)
    datamodule = PetfinderDataModule(train_df, val_df, config)
    model = Model(config)
    earystopping = EarlyStopping(monitor="val_loss")
    lr_monitor = callbacks.LearningRateMonitor()
    loss_checkpoint = callbacks.ModelCheckpoint(
        filename="best_loss",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        save_last=False,
    )
    logger = TensorBoardLogger(config.model.name)

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=config.epoch,
        callbacks=[lr_monitor, loss_checkpoint, earystopping],
        **config.trainer,
    )
    trainer.fit(model, datamodule=datamodule)

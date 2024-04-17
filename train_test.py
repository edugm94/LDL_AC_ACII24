import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from Models.networks import MLP, CNN, DGCNN, DBN
from torch.utils.data import DataLoader
from Models.LightModels import SLL, LDL
from torch.utils.data import random_split
from sklearn.metrics import f1_score, accuracy_score
from Loaders.CrossTrialSEED import CrossTrialLoader as LoaderSEED
from Loaders.CrossTrialSEED5 import CrossTrialLoader as LoaderSEED5
from pytorch_lightning.callbacks import (
    LearningRateMonitor, ModelCheckpoint, EarlyStopping)
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import gc

import sys
sys.path.append("..")
from utils.constants import SENSOR_LOCS
from utils.dl_utils import get_edge_weight_from_electrode

def train_test_session(subject, learn_type, network, sess_ids, dataset, config):

    # Get flags to set up experiment
    is_le = True if learn_type == "LDL" else False
    img_fmt = True if network == "CNN" else False
    is_flat = False if network == "DGCNN" else True

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if dataset == "SEED":
        loader = LoaderSEED(img_fmt=img_fmt, is_le=is_le, is_flat=is_flat)
        n_classes = 3
    else:
        loader = LoaderSEED5(img_fmt=img_fmt, is_le=is_le, is_flat=is_flat)
        n_classes = 5

    # Setting experiment paths
    test_subject = "Subject-{}".format(str(subject))
    if is_le:
        le_method = "LEVI"
        save_path = f'./{dataset}/{learn_type}/{le_method}/{network}/{test_subject}/'
    else:
        save_path = f'./{dataset}/{learn_type}/{network}/{test_subject}/'

    for sess_id in sess_ids:
        print("####################################\n")
        print(f'\t Training Fold {sess_id} - Subject {subject}\n')
        print("####################################\n")
        # Load corresponding datasets
        train_ds, test_ds = loader.get_experiment_sets(subject, sess_id)

        # Creating validation set:
        val_size = int(len(train_ds) * 0.2)
        train_size = len(train_ds) - val_size
        train_ds, val_ds = random_split(train_ds, lengths=[train_size, val_size])
        train_dl = DataLoader(train_ds, batch_size=config["bs"], shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=config["bs"], shuffle=True)
        test_dl = DataLoader(test_ds, batch_size=config["bs"], shuffle=False)

        # Setting logger for each fold experiment
        logger = TensorBoardLogger(save_dir=f'./experiments/',
                                   name=save_path,
                                   version="Session-{}".format(str(sess_id)))

        # Instantiate the network architecture to train
        if network == "MLP":
            net = MLP(n_classes=n_classes)
        
        else:
            raise NotImplementedError

        # Instantiate the type of training: single label or label distribution
        model = SLL(net, n_classes, config) if not is_le else LDL(net, n_classes, config)

        # Create callbacks
        cb_lr_monitor = LearningRateMonitor(logging_interval="step")

        cb_checkpoint = ModelCheckpoint(monitor="train_loss",
                                        enable_version_counter=False,
                                        save_top_k=1, verbose=True)

        cb_early_stop = EarlyStopping(monitor="val_loss", patience=30,
                                      verbose=True, mode="min", min_delta=0.01)

        # Create Trainer
        trainer = pl.Trainer(
            max_epochs=1000,
            devices=1,
            logger=logger,
            callbacks=[cb_lr_monitor, cb_checkpoint, cb_early_stop],
            accelerator="gpu"
        )
        # Train the model
        trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)

        # Test the model and save metrics to disk
        metrics = trainer.test(model=model, dataloaders=test_dl)[0]
        acc_ = np.around(metrics["accuracy"], 2)
        f1_ = np.around(metrics["f1"], 2)
        metrics_fold = "Acc={}\t\tF1={}\n".format(acc_, f1_)
        save_path_fold = "./experiments/" + save_path + "Session-{}/".format(str(sess_id))
        f = open(save_path_fold + "metrics.txt", "w")
        f.write(metrics_fold)
        f.close()


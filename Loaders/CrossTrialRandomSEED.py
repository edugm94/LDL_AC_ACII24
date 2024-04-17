import sys
sys.path.append("..")
import glob
import torch
import pickle
import numpy as np
import scipy.io as sio
from Loaders.LE import LEVI
from utils.constants import SENSOR_LOCS
from utils.dl_utils import CreatePyTorchDataset
from utils.constants import PATH_DE_FEATURES as feat_path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

from sklearn.model_selection import train_test_split


int_to_emotion = {0: "Negative", 1: "Neutral", 2: "Positive"}


class CrossTrialLoader:
    def __init__(self,  img_fmt, is_le=False, is_flat=True, feature="de_LDS"):
        self.labels = np.array([1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]) + 1
        self.feature = feature
        self.is_le = is_le
        self.is_flat = is_flat
        self.img_fmt = img_fmt
        self.sensor_locs = np.asarray(SENSOR_LOCS)
        self.columns = ['Train', 'Trial', 'Emotion']
        # Create an empty DataFrame
        self.df = pd.DataFrame(columns=self.columns)
        self.train_id = None
        self.test_id = None

        X_train, X_test, self.train_id, self.test_id = train_test_split(
            self.labels, range(len(self.labels)), test_size=6,
            random_state=42, stratify=self.labels
        )
        self.train_id = [x + 1 for x in self.train_id]
        self.test_id = [x + 1 for x in self.test_id]

    def __generating_data(self, data_dict, clip_label):
        x_train = []
        y_train = []
        for id in self.train_id:
            data = data_dict[self.feature + str(id)]
            _, num, _ = data.shape
            data = np.reshape(data, (num, -1))
            labels = np.zeros(num, dtype=int) + clip_label[id - 1]
            x_train.append(data)
            y_train.append(labels)
            ##########################
            emotion = list(labels)
            emo_col = [int_to_emotion[val] for val in emotion]
            trial_col = [id] * num
            train_col = [1] * num
            df_aux = pd.DataFrame({"Train": train_col, "Trial": trial_col, "Emotion": emo_col})
            self.df = pd.concat([self.df, df_aux])
            self.df = self.df.reset_index(drop=True)
            ##########################

        x_test = []
        y_test = []
        for id in self.test_id:
            data = data_dict[self.feature + str(id)]
            _, num, _ = data.shape
            data = np.reshape(data, (num, -1))
            labels = np.zeros(num, dtype=int) + clip_label[id - 1]
            x_test.append(data)
            y_test.append(labels)
            ##########################
            emotion = list(labels)
            emo_col = [int_to_emotion[val] for val in emotion]
            trial_col = [id] * num
            train_col = [0] * num
            df_aux = pd.DataFrame({"Train": train_col, "Trial": trial_col, "Emotion": emo_col})
            self.df = pd.concat([self.df, df_aux])
            self.df = self.df.reset_index(drop=True)
            ##########################

        x_train = np.concatenate(x_train, axis=0)
        x_test = np.concatenate(x_test, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        y_test = np.concatenate(y_test, axis=0)
        return x_train, x_test, y_train, y_test



    def get_experiment_sets(self, subject, sess_id):
        self.df = pd.DataFrame(columns=self.columns)
        sessions = sorted(glob.glob(feat_path +
                                                    "{}_*".format(subject)))

        target_sess = sessions[sess_id - 1]
        data_sess = sio.loadmat(target_sess)

        x_train, x_test, y_train, y_test = self.__generating_data(data_sess,
                                                                  self.labels)

        sc = StandardScaler()
        sc.fit(x_train)
        x_train = sc.transform(x_train)
        x_test = sc.transform(x_test)
        y_train_hot = torch.nn.functional.one_hot(torch.tensor(y_train).to(torch.int64)).numpy()
        y_test_hot = torch.nn.functional.one_hot(torch.tensor(y_test).to(torch.int64)).numpy()

        # Apply LE when dealing 1D DE features, else learn autoencoder
        if self.is_le and not self.img_fmt:
            le = LEVI()
            y_dist_train = le.fit_transform(x_train, y_train_hot, lr=0.001, epochs=5000)
            y_dist_test = le.transform(x_test, y_test_hot)
            y_train = y_dist_train
            y_test = y_dist_test

            ###########################################
            y_neg = np.concatenate([y_train_hot[:, 0], y_test_hot[:, 0]], axis=0)
            y_neu = np.concatenate([y_train_hot[:, 1], y_test_hot[:, 1]], axis=0)
            y_pos = np.concatenate([y_train_hot[:, 2], y_test_hot[:, 2]], axis=0)

            d_neg = np.concatenate([y_train[:, 0], y_test[:, 0]], axis=0)
            d_neu = np.concatenate([y_train[:, 1], y_test[:, 1]], axis=0)
            d_pos = np.concatenate([y_train[:, 2], y_test[:, 2]], axis=0)

            logits = np.concatenate([y_train_hot, y_test_hot], axis=0)
            distributions = np.concatenate([y_train, y_test], axis=0)
            is_correct = np.argmax(logits, axis=1) == np.argmax(distributions, axis=1)
            is_correct = np.asarray(is_correct).astype(int)

            data = {
                "y_neg": y_neg,
                "y_neu": y_neu,
                "y_pos": y_pos,
                "d_neg": d_neg,
                "d_neu": d_neu,
                "d_pos": d_pos,
                "is_correct": is_correct
            }
            df_dist = pd.DataFrame(data)
            self.df = pd.concat([self.df, df_dist], axis=1)
            ###########################################


        # We transform back to original shape to train DGCNN model
        if not self.is_flat:
            x_train = x_train.reshape(x_train.shape[0], 62, 5)
            x_test = x_test.reshape(x_test.shape[0], 62, 5)


        x_train_ds = CreatePyTorchDataset(data=x_train, labels=y_train, logic_gt=y_train_hot)
        x_test_ds = CreatePyTorchDataset(data=x_test, labels=y_test, logic_gt=y_test_hot)
        return x_train_ds, x_test_ds

if __name__ == '__main__':

    loader = CrossTrialLoader(img_fmt=False, is_le=True)
    x_train_ds, x_test_ds = loader.get_experiment_sets(9, 2)

    print("Stop")
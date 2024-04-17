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

int_to_emotion = {0: "Negative", 1: "Neutral", 2: "Positive"}

class CrossTrialLoader:
    def __init__(self,  img_fmt, is_le=False, is_flat=True, feature="de_LDS"):
        self.labels = np.array([1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]) + 1
        self.feature = feature
        self.is_le = is_le
        self.is_flat = is_flat
        self.img_fmt = img_fmt
        self.sensor_locs = np.asarray(SENSOR_LOCS)
        self.columns = ['Trial', 'Emotion']
        # Create an empty DataFrame
        self.df = pd.DataFrame(columns=self.columns)

    def __generating_data(self, data_dict, clip_label):
        # first 9 as training, the last 6 as testing
        train_data = data_dict[self.feature + '1']
        _, num, _ = train_data.shape
        train_label = np.zeros(num, dtype=int) + clip_label[0]

        ##########################
        emotion = list(train_label)
        emo_col = [int_to_emotion[val] for val in emotion]
        trial_col = [1] * num
        df_aux = pd.DataFrame({"Trial": trial_col, "Emotion": emo_col})
        self.df = pd.concat([self.df, df_aux])
        ##########################

        train_data = np.swapaxes(train_data, 0, 1)
        train_data = np.reshape(train_data, (num, -1))
        train_residual_index = [2, 3, 4, 5, 6, 7, 8, 9]
        for ind, i in enumerate(train_residual_index):
            used_data = data_dict[self.feature + str(i)]
            _, num, _ = used_data.shape
            used_label = np.zeros(num, ) + clip_label[ind + 1]

            ##########################
            emotion = list(used_label)
            emo_col = [int_to_emotion[val] for val in emotion]
            trial_col = [i] * num
            df_aux = pd.DataFrame({"Trial": trial_col, "Emotion": emo_col})
            self.df = pd.concat([self.df, df_aux])
            self.df = self.df.reset_index(drop=True)
            ##########################

            used_data = np.swapaxes(used_data, 0, 1)
            used_data = np.reshape(used_data, (num, -1))
            train_data = np.vstack((train_data, used_data))
            train_label = np.hstack((train_label, used_label))

        test_data = data_dict[self.feature + '10']
        _, num, _ = test_data.shape
        test_label = np.zeros(num, ) + clip_label[9]

        ##########################
        emotion = list(test_label)
        emo_col = [int_to_emotion[val] for val in emotion]
        trial_col = [10] * num
        df_aux = pd.DataFrame({"Trial": trial_col, "Emotion": emo_col})
        self.df = pd.concat([self.df, df_aux])
        self.df = self.df.reset_index(drop=True)
        ##########################

        test_data = np.swapaxes(test_data, 0, 1)
        test_data = np.reshape(test_data, (num, -1))
        test_residual_index = [11, 12, 13, 14, 15]
        for ind, i in enumerate(test_residual_index):
            used_data = data_dict[self.feature + str(i)]
            _, num, _ = used_data.shape
            used_label = np.zeros(num, ) + clip_label[ind + 10]

            ##########################
            emotion = list(used_label)
            emo_col = [int_to_emotion[val] for val in emotion]
            trial_col = [i] * num
            df_aux = pd.DataFrame({"Trial": trial_col, "Emotion": emo_col})
            self.df = pd.concat([self.df, df_aux])
            self.df = self.df.reset_index(drop=True)
            ##########################

            used_data = np.swapaxes(used_data, 0, 1)
            used_data = np.reshape(used_data, (num, -1))
            test_data = np.vstack((test_data, used_data))
            test_label = np.hstack((test_label, used_label))
        return train_data, test_data, train_label, test_label

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
            #------------------------------------#
            # Cheating option
            n_train = x_train.shape[0]
            X = np.concatenate([x_train, x_test], axis=0)
            y = np.concatenate([y_train_hot, y_test_hot], axis=0)
            y_dist = le.fit_transform(X, y, lr=0.001, epochs=1000)
            y_dist_train = y_dist[:n_train]
            y_dist_test = y_dist[n_train:]
            # ------------------------------------#
            # y_dist_train = le.fit_transform(x_train, y_train_hot, lr=0.001, epochs=5000)
            # # y_dist_test = le.fit_transform(x_test, y_test_hot, lr=0.001, epochs=5000)
            # y_dist_test = le.transform(x_test, y_test_hot)
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
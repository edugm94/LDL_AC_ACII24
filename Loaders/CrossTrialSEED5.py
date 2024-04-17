import sys
sys.path.append("..")
import glob
import torch
import pickle
import numpy as np
import pandas as pd
from Loaders.LE import LEVI
from utils.constants import SENSOR_LOCS
from utils.dl_utils import CreatePyTorchDataset
from sklearn.preprocessing import StandardScaler
from utils.constants import PATH_DE_FEATURES_SEED5 as feat_path

int_to_emotion = {0: "Disgust", 1: "Fear", 2: "Sad", 3: "Neutral", 4: "Happy"}

class CrossTrialLoader:
    def __init__(self, img_fmt, is_le=False, is_flat=True, feature="de_LDS"):
        self.feature = feature
        self.is_le = is_le
        self.is_flat = is_flat
        self.img_fmt = img_fmt
        self.sensor_locs = np.asarray(SENSOR_LOCS)
        self.path_feature = feat_path
        self.subject_data = None
        self.sess_ids = {
            1: [i for i in range(15)],
            2: [i for i in range(15, 30)],
            3: [i for i in range(30, 45)]
        }
        self.columns = ['Trial', 'Emotion']
        # Create an empty DataFrame
        self.df = pd.DataFrame(columns=self.columns)

    def _load_subject_data(self, subject):
        filename = glob.glob(feat_path + "{}_*".format(subject))[0]
        data = np.load(filename)
        return data

    def get_experiment_sets(self, subject, sess_id):
        self.df = pd.DataFrame(columns=self.columns)

        subject_data = self._load_subject_data(subject)
        features = pickle.loads(subject_data['data'])
        labels = pickle.loads(subject_data['label'])

        idx = self.sess_ids[sess_id]
        if sess_id == 2:
            offset = 15
        elif sess_id == 3:
            offset = 30
        else:
            offset = 0
        # train_id = idx[0:9]
        # test_id = idx[9:]
        train_id = idx[0:10]
        test_id = idx[10:]
        x_train = list()
        y_train = list()


        for i in train_id:
            x_train.append(features[i])
            y_train.append(labels[i])
            ###############
            n_samples = features[i].shape[0]
            trial_col = [(i+1) - offset] * n_samples
            emo_col = [int_to_emotion[val] for val in list(labels[i])]
            df_aux = pd.DataFrame({"Trial": trial_col, "Emotion": emo_col})
            self.df = pd.concat([self.df, df_aux])
            self.df = self.df.reset_index(drop=True)
            ###############

        x_test = list()
        y_test = list()
        for i in test_id:
            x_test.append(features[i])
            y_test.append(labels[i])
            ###############
            n_samples = features[i].shape[0]
            trial_col = [(i + 1) - offset] * n_samples
            emo_col = [int_to_emotion[val] for val in list(labels[i])]
            df_aux = pd.DataFrame({"Trial": trial_col, "Emotion": emo_col})
            self.df = pd.concat([self.df, df_aux])
            self.df = self.df.reset_index(drop=True)
            ###############

        x_train = np.concatenate(x_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        x_test = np.concatenate(x_test, axis=0)
        y_test = np.concatenate(y_test, axis=0)

        sc = StandardScaler()
        sc.fit(x_train)
        x_train = sc.transform(x_train)
        x_test = sc.transform(x_test)
        y_train_hot = torch.nn.functional.one_hot(torch.tensor(y_train).to(torch.int64)).numpy()
        y_test_hot = torch.nn.functional.one_hot(torch.tensor(y_test).to(torch.int64)).numpy()
        # Apply LE when dealing 1D DE features, else learn autoencoder
        if self.is_le and not self.img_fmt:
            le = LEVI()
            # ------------------------------------#
            n_train = x_train.shape[0]
            X = np.concatenate([x_train, x_test], axis=0)
            y = np.concatenate([y_train_hot, y_test_hot], axis=0)
            y_dist = le.fit_transform(X, y, lr=0.001, epochs=1000)
            y_dist_train = y_dist[:n_train]
            y_dist_test = y_dist[n_train:]
            # ------------------------------------#
            # y_dist_train = le.fit_transform(x_train, y_train_hot, lr=0.001, epochs=5000)
            # y_dist_test = le.transform(x_test, y_test_hot)
            y_train = y_dist_train
            y_test = y_dist_test
            ###################################
            y_dis = np.concatenate([y_train_hot[:, 0], y_test_hot[:, 0]], axis=0)
            y_fear = np.concatenate([y_train_hot[:, 1], y_test_hot[:, 1]], axis=0)
            y_sad = np.concatenate([y_train_hot[:, 2], y_test_hot[:, 2]], axis=0)
            y_neu = np.concatenate([y_train_hot[:, 3], y_test_hot[:, 3]], axis=0)
            y_hap = np.concatenate([y_train_hot[:, 4], y_test_hot[:, 4]], axis=0)

            d_dis = np.concatenate([y_train[:, 0], y_test[:, 0]], axis=0)
            d_fear = np.concatenate([y_train[:, 1], y_test[:, 1]], axis=0)
            d_sad = np.concatenate([y_train[:, 2], y_test[:, 2]], axis=0)
            d_neu = np.concatenate([y_train[:, 3], y_test[:, 3]], axis=0)
            d_hap = np.concatenate([y_train[:, 4], y_test[:, 4]], axis=0)

            logits = np.concatenate([y_train_hot, y_test_hot], axis=0)
            distributions = np.concatenate([y_train, y_test], axis=0)
            is_correct = np.argmax(logits, axis=1) == np.argmax(distributions, axis=1)
            is_correct = np.asarray(is_correct).astype(int)

            data = {
                "y_disgust": y_dis,
                "y_fear": y_fear,
                "y_sad": y_sad,
                "y_neutral": y_neu,
                "y_happy": y_hap,
                "d_disgust": d_dis,
                "d_fear": d_fear,
                "d_sad": d_sad,
                "d_neutral": d_neu,
                "d_happy": d_hap,
                "is_correct": is_correct
            }
            df_dist = pd.DataFrame(data)
            self.df = pd.concat([self.df, df_dist], axis=1)
            ###################################

        # We transform back to original shape to train DGCNN model
        if not self.is_flat:
            x_train = x_train.reshape(x_train.shape[0], 62, 5)
            x_test = x_test.reshape(x_test.shape[0], 62, 5)

        x_train_ds = CreatePyTorchDataset(data=x_train, labels=y_train, logic_gt=y_train_hot)
        x_test_ds = CreatePyTorchDataset(data=x_test, labels=y_test, logic_gt=y_test_hot)
        return x_train_ds, x_test_ds


if __name__ == '__main__':
    loader = CrossTrialLoader(img_fmt=False, is_le=False)
    train, test = loader.get_experiment_sets(1,3)
    print("STOP")

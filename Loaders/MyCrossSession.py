import sys
sys.path.append("..")
import numpy as np
from Loaders.LE import LEVI
from utils.dl_utils import CreatePyTorchDataset
from utils.constants import MY_DE_FEATURES as PATH_DE_FEATURES
import glob
import scipy.io as sio
from utils.data_utils import gen_images
from utils.constants import SENSOR_LOCS
from sklearn.preprocessing import StandardScaler, scale, minmax_scale
import torch


class CrossSessionLoader:
    def __init__(self, img_fmt, is_le=False, is_flat=True, feature="de_movingAve"):
        self.img_fmt = img_fmt
        self.feature = feature
        self.labels = sio.loadmat(PATH_DE_FEATURES + 'label.mat')['label'][0]
        # Set sessions ID destinated for the training and testing set
        # Following original paper: first 9 for training, remaining 6 for testing
        self.sess_id_train = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
        self.sess_id_test = ["10", "11", "12", "13", "14", "15"]
        self.sensor_locs = np.asarray(SENSOR_LOCS)
        self.is_le = is_le
        self.is_flat = is_flat

    def _merge_data(self, sess_feat, is_train):
        """
        Method to merge features into train or test sets
        :param sess_feat: Dictionary of features for each session
        :param is_train: Flag to set if it is train or test set returned
        :return: feature and labels sets
        """
        x_data = []
        y_data = []
        sess_ids = self.sess_id_train if is_train else self.sess_id_test

        for sess_id in sess_ids:
            aux_feat = sess_feat["sess" + str(int(sess_id)-1)].reshape(-1, 62 * 5)
            n_samples = aux_feat.shape[0]
            aux_labels = np.ones(n_samples) * (self.labels[int(sess_id) - 1] + 1)
            x_data.append(aux_feat)
            y_data.append(aux_labels)

        x_data = np.concatenate(x_data, axis=0)
        y_data = np.concatenate(y_data, axis=0)
        return x_data, y_data

    def get_experiment_sets(self, subject, sess_id):

        # Sessions file sorted by date --> index 0: first recording, 3: last record
        sessions = sorted(glob.glob(PATH_DE_FEATURES +
                                    "{}_*".format(subject)))

        target_sess = sessions[sess_id-1]
        data = sio.loadmat(target_sess)

        keys_to_filter = ["sess{}".format(i) for i in range(0, 16)]
        sess_feat = dict(filter(lambda item: item[0] in keys_to_filter, data.items()))

        x_train, y_train = self._merge_data(sess_feat, is_train=True)
        x_test, y_test = self._merge_data(sess_feat, is_train=False)

        # #Standarize data
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)
        # x_train = scale(x_train)
        # x_train = minmax_scale(x_train)
        # x_test = scale(x_test)
        # x_test = minmax_scale(x_test)

        # We transform back to original shape to train DGCNN model
        if not self.is_flat:
            x_train = x_train.reshape(x_train.shape[0], 62, 5)
            x_test = x_test.reshape(x_test.shape[0], 62, 5)

        # Apply LE when dealing 1D DE features, else learn autoencoder
        if self.is_le and not self.img_fmt:
            le = LEVI()
            y_train_hot = torch.nn.functional.one_hot(torch.tensor(y_train).to(torch.int64)).numpy()
            y_test_hot = torch.nn.functional.one_hot(torch.tensor(y_test).to(torch.int64)).numpy()
            y_dist_train = le.fit_transform(x_train, y_train_hot, learning_rate=0.001, epochs=3000)
            y_dist_test = le.transform(x_test, y_test_hot)
            y_train = y_dist_train
            y_test = y_dist_test

        if self.img_fmt:
            x_train = gen_images(self.sensor_locs, x_train, n_gridpoints=32,
                                 normalize=True, edgeless=False)
            x_test = gen_images(self.sensor_locs, x_test, n_gridpoints=32,
                                normalize=True, edgeless=False)
            #TODO: Fill up this with an autoencoder to apply later on LE...

        x_train_ds = CreatePyTorchDataset(data=x_train, labels=y_train)
        x_test_ds = CreatePyTorchDataset(data=x_test, labels=y_test)
        return x_train_ds, x_test_ds


if __name__ == '__main__':
    img_format = False
    loader = CrossSessionLoader(img_fmt=img_format, is_le=False)
    x_train_ds, x_test_ds = loader.get_experiment_sets(1, 1)
    print("Stop")




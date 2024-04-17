import sys
sys.path.append("..")

import numpy as np
import torch.random
import torch.nn as nn
from tqdm import tqdm
from utils.dl_utils import score
import torch.nn.functional as F
from scipy.special import softmax
from sklearn.base import BaseEstimator, TransformerMixin


class _Base():
    def __init__(self, random_state=None):
        if not random_state is None:
            np.random.seed(random_state)

        self._n_features = None
        self._n_outputs = None

    def fit(self, X, y=None):
        self._X = X
        self._y = y
        self._n_features = self._X.shape[1]
        if not self._y is None:
            self._n_outputs = self._y.shape[1]

    def _not_been_fit(self):
        raise ValueError("The model has not yet been fit. "
                         "Try to call 'fit()' first with some training data.")

    @property
    def n_features(self):
        if self._n_features is None:
            self._not_been_fit()
        return self._n_features

    @property
    def n_outputs(self):
        if self._n_outputs is None:
            self._not_been_fit()
        return self._n_outputs


class _BaseLE(_Base):

    def fit_transform(self, X, l):
        super().fit(X, None)
        self._l = l
        self._n_outputs = self._l.shape[1]

    def score(self, X, l, y,
              metrics=["chebyshev", "clark", "canberra", "kl_divergence", "cosine", "intersection"]):
        return score(y, self.fit_transform(X, l), metrics=metrics)


class BaseLE(_BaseLE, TransformerMixin, BaseEstimator):

    def __init__(self, random_state=None):
        super().__init__(random_state)


class _BaseDeep(nn.Module):
    def __init__(self, n_hidden=None, n_latent=None, random_state=None):
        nn.Module.__init__(self)
        if not random_state is None:
            torch.random.set_rng_state(torch.tensor(69))
        self._n_latent = n_latent
        self._n_hidden = n_hidden


class BaseDeepLE(_BaseLE, _BaseDeep):
    def __init__(self, n_hidden=None, n_latent=None, random_state=None):
        _BaseLE.__init__(self, random_state)
        _BaseDeep.__init__(self, n_hidden, n_latent, random_state)

    def fit_transform(self, X, l):
        _BaseLE.fit_transform(self, X, l)
        self._X = torch.from_numpy(self._X).float()
        self._l = torch.from_numpy(self._l).float()


class Encoder(nn.Module):
    def __init__(self, input_shape, n_outputs):
        super(Encoder, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features=input_shape, out_features=500),
            nn.Softplus(),
            nn.Linear(in_features=500, out_features=500),
            nn.Softplus(),
            nn.Linear(in_features=500, out_features=500),
            nn.Softplus(),
            nn.Linear(in_features=500, out_features=n_outputs * 2)
        )
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # nn.init.xavier_normal_(m.weight.data)
                nn.init.normal_(m.weight.data, mean=0, std=0.01)
                nn.init.zeros_(m.bias.data)

    def forward(self, inputs):
        return self.layer(inputs)


class Decoder(nn.Module):
    def __init__(self, n_outputs, n_features):
        super(Decoder, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features=n_outputs, out_features=500),
            nn.Softplus(),
            nn.Linear(in_features=500, out_features=500),
            nn.Softplus(),
            nn.Linear(in_features=500, out_features=500),
            nn.Softplus(),
            nn.Linear(in_features=500, out_features=n_features)
        )
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, mean=0, std=0.01)
                nn.init.zeros_(m.bias.data)

    def forward(self, inputs):
        return self.layer(inputs)

class LEVI(BaseDeepLE):
    def __init__(self, n_hidden=None, n_latent=None, random_state=None):
        super().__init__(n_hidden, n_latent, random_state)
        self._optimizer = None
        self._decoder = None
        self._encoder = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = "cpu"

    def _loss(self, X, l):
        l.requires_grad = True
        X.requires_grad = True
        inputs = torch.cat((X, l), dim=1)

        latent = self._encoder(inputs)
        mean = latent[:, :self._n_outputs]
        var = F.softplus(latent[:, self._n_outputs:])
        mean = mean.to(self.device)
        var = var.to(self.device)

        if torch.isnan(mean).any().item() or torch.isnan(var).any().item():
            mean = torch.randn_like(mean)
            var = torch.randn_like(var)

        d = torch.distributions.normal.Normal(loc=mean, scale=var)

        std_d = torch.distributions.normal.Normal(
            loc=torch.zeros(self._n_outputs, dtype=torch.float32, device=self.device),
            scale=torch.ones(self._n_outputs, dtype=torch.float32, device=self.device))

        samples = d.rsample()

        y_hat = F.softmax(samples, dim=1)
        outputs = self._decoder(samples)

        # Reconstruction losses
        kl = torch.mean(torch.distributions.kl.kl_divergence(d, std_d), dim=1)
        rec_X = torch.mean(F.mse_loss(outputs, X, reduction="none"), dim=1)
        rec_y = torch.mean(F.binary_cross_entropy(y_hat, l, reduction="none"), dim=1)

        # Total loss
        total_loss = torch.sum(kl + rec_X + rec_y)
        return total_loss

    def fit_transform(self, X, l, lr=1e-5, epochs=3000):
        super().fit_transform(X, l)
        self._X = self._X.to(self.device)
        self._l = self._l.to(self.device)

        input_shape = self._n_features + self._n_outputs

        self._encoder = Encoder(input_shape=input_shape,
                                n_outputs=self.n_outputs).to(self.device)

        self._decoder = Decoder(n_outputs=self.n_outputs,
                                n_features=self.n_features).to(self.device)

        self._optimizer = torch.optim.Adam(params=self.parameters(),
                                           lr=lr)

        for _ in tqdm(range(epochs), desc="Enhancing labels..."):
            self._optimizer.zero_grad()
            loss = self._loss(self._X, self._l)
            loss.backward()
            self._optimizer.step()

        with torch.no_grad():
            inputs = torch.cat((self._X, self._l), dim=1)
            latent = self._encoder(inputs)
            mean = latent[:, :self.n_outputs].cpu().detach().numpy()
        return softmax(mean, axis=1)

    # def _loss(self, X, l):
    #     l.requires_grad = True
    #     X.requires_grad = True
    #     inputs = torch.cat((X, l), dim=1)
    #
    #     latent = self._encoder(inputs)
    #     mean = latent[:, :self._n_outputs]
    #     var = F.softplus(latent[:, self._n_outputs:])
    #
    #     d = torch.distributions.normal.Normal(loc=mean, scale=var)
    #
    #     std_d = torch.distributions.normal.Normal(
    #         loc=torch.zeros(self._n_outputs, dtype=torch.float32, device=self.device),
    #         scale=torch.ones(self._n_outputs, dtype=torch.float32, device=self.device))
    #
    #     samples = d.rsample()
    #
    #     X_hat = self._decoder_X(samples)
    #     l_hat = self._decoder_l(samples)
    #     # y_hat = F.softmax(samples, dim=1)
    #
    #     # Reconstruction losses
    #     kl = torch.mean(torch.distributions.kl.kl_divergence(d, std_d), dim=1)
    #     rec_X = torch.mean(F.mse_loss(X_hat, X, reduction="none"), dim=1)
    #     # rec_y = torch.mean(F.binary_cross_entropy(torch.sigmoid(l_hat), l, reduction="none"), dim=1)
    #     rec_y = torch.mean(F.binary_cross_entropy_with_logits(l_hat, l, reduction="none"), dim=1)
    #
    #     # Total loss
    #
    #     loss1 = torch.sum((l-samples)**2)
    #     loss2 = torch.sum(kl + rec_X + rec_y)
    #     total_loss = loss1 + self._alpha * loss2
    #     return total_loss
    #
    # def fit_transform(self, X, l, lr=1e-5, epochs=3000):
    #     super().fit_transform(X, l)
    #     self._X = self._X.to(self.device)
    #     self._l = self._l.to(self.device)
    #
    #     self._alpha = 1.
    #
    #     input_shape = self._n_features + self._n_outputs
    #
    #     self._encoder = Encoder(input_shape=input_shape,
    #                             n_outputs=self.n_outputs).to(self.device)
    #
    #     self._decoder_X = Decoder(n_outputs=self.n_outputs,
    #                             n_features=self.n_features).to(self.device)
    #
    #     self._decoder_l = Decoder(n_outputs=self.n_outputs,
    #                               n_features=self.n_outputs).to(self.device)
    #
    #     self._optimizer = torch.optim.Adam(params=self.parameters(),
    #                                        lr=lr)
    #
    #     for _ in tqdm(range(epochs), desc="Enhancing labels..."):
    #         self._optimizer.zero_grad()
    #         loss = self._loss(self._X, self._l)
    #         loss.backward()
    #         self._optimizer.step()
    #
    #     with torch.no_grad():
    #         inputs = torch.cat((self._X, self._l), dim=1)
    #         latent = self._encoder(inputs)
    #         mean = latent[:, :self.n_outputs].cpu().detach().numpy()
    #     return softmax(mean, axis=1)

    def transform(self, X, l):
        X = torch.from_numpy(X).float().to(self.device)
        l = torch.from_numpy(l).float().to(self.device)
        with torch.no_grad():
            inputs = torch.cat((X, l), dim=1)
            latent = self._encoder(inputs)
            mean = latent[:, :self.n_outputs].cpu().detach().numpy()
        return softmax(mean, axis=1)


# class Encoder(nn.Module):
#     def __init__(self, input_shape, n_hidden, n_outputs):
#         super(Encoder, self).__init__()
#         self.layer = nn.Sequential(
#             nn.Linear(in_features=input_shape, out_features=n_hidden),
#             nn.Sigmoid(),
#             nn.Linear(in_features=n_hidden, out_features=n_outputs * 2)
#         )
#         self._init_weight()
#
#     def _init_weight(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight.data)
#                 nn.init.zeros_(m.bias.data)
#
#     def forward(self, inputs):
#         return self.layer(inputs)
#
#
# class Decoder(nn.Module):
#     def __init__(self, n_outputs, n_hidden, n_features):
#         super(Decoder, self).__init__()
#         self.layer = nn.Sequential(
#             nn.Linear(in_features=n_outputs, out_features=n_hidden),
#             nn.Sigmoid(),
#             nn.Linear(in_features=n_hidden, out_features=n_features)
#         )
#         self._init_weight()
#
#     def _init_weight(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight.data)
#                 nn.init.zeros_(m.bias.data)
#
#     def forward(self, inputs):
#         return self.layer(inputs)

# class LEVI(BaseDeepLE):
#     def __init__(self, n_hidden=None, n_latent=None, random_state=None):
#         super().__init__(n_hidden, n_latent, random_state)
#         self._optimizer = None
#         self._decoder = None
#         self._encoder = None
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     def _loss(self, X, l):
#         l.requires_grad = True
#         X.requires_grad = True
#         inputs = torch.cat((X, l), dim=1)
#
#         latent = self._encoder(inputs)
#         mean = latent[:, :self._n_outputs]
#         var = F.softplus(latent[:, self._n_outputs:])
#
#         d = torch.distributions.normal.Normal(loc=mean, scale=var)
#
#         std_d = torch.distributions.normal.Normal(
#             loc=torch.zeros(self._n_outputs, dtype=torch.float32, device=self.device),
#             scale=torch.ones(self._n_outputs, dtype=torch.float32, device=self.device))
#
#         samples = d.rsample()
#
#         y_hat = F.softmax(samples, dim=1)
#         outputs = self._decoder(samples)
#
#         # Reconstruction losses
#         kl = torch.mean(torch.distributions.kl.kl_divergence(d, std_d), dim=1)
#         rec_X = torch.mean(F.mse_loss(outputs, X, reduction="none"), dim=1)
#         rec_y = torch.mean(F.binary_cross_entropy(y_hat, l, reduction="none"), dim=1)
#
#         # Total loss
#         total_loss = torch.sum(kl + rec_X + rec_y)
#         return total_loss
#
#     def fit_transform(self, X, l, learning_rate=1e-5, epochs=3000):
#         super().fit_transform(X, l)
#         self._X = self._X.to(self.device)
#         self._l = self._l.to(self.device)
#
#         input_shape = self._n_features + self._n_outputs
#
#         if self._n_hidden is None:
#             self._n_hidden = self._n_features * 3 // 2
#
#         self._encoder = Encoder(input_shape=input_shape,
#                                 n_hidden=self._n_hidden,
#                                 n_outputs=self.n_outputs).to(self.device)
#
#         self._decoder = Decoder(n_outputs=self.n_outputs,
#                                 n_hidden=self._n_hidden,
#                                 n_features=self.n_features).to(self.device)
#
#         self._optimizer = torch.optim.Adam(params=self.parameters(),
#                                            lr=learning_rate)
#
#         for _ in tqdm(range(epochs), desc="Enhancing labels..."):
#             self._optimizer.zero_grad()
#             loss = self._loss(self._X, self._l)
#             loss.backward()
#             self._optimizer.step()
#
#         with torch.no_grad():
#             inputs = torch.cat((self._X, self._l), dim=1)
#             latent = self._encoder(inputs)
#             mean = latent[:, :self.n_outputs].cpu().detach().numpy()
#         return softmax(mean, axis=1)
#
#     def transform(self, X, l):
#         X = torch.from_numpy(X).float().to(self.device)
#         l = torch.from_numpy(l).float().to(self.device)
#         with torch.no_grad():
#             inputs = torch.cat((X, l), dim=1)
#             latent = self._encoder(inputs)
#             mean = latent[:, :self.n_outputs].cpu().detach().numpy()
#         return softmax(mean, axis=1)

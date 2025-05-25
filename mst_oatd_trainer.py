import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.mixture import GaussianMixture
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset

from logging_set import get_logger
from mst_oatd import MST_OATD
from utils import auc_score, make_mask, make_len_mask

from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    # batch is a list of trajectories, where each trajectory is a list of [id, grid, [timestamp]]
    seq_lengths = np.array([len(traj) for traj in batch])
    grid_seqs, timestamp_seqs = [], []

    for sequence in batch:
        grids, timestamp = zip(*[(row[1], row[2]) for row in sequence])
        grid_seqs.append(torch.tensor(grids))
        timestamp_seqs.append(torch.tensor(timestamp, dtype=torch.float))

    # Pad sequences to the same length (batch-first)
    padded_grids = pad_sequence(grid_seqs, batch_first=True)
    padded_timestamps = pad_sequence(timestamp_seqs, batch_first=True)

    return padded_grids, padded_timestamps, seq_lengths


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class TrajectoryDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.start_indices = self._compute_start_indices()

    def _compute_start_indices(self):
        start_indices = [0]
        last_id = self.data[0][0]
        for i in range(1, len(self.data)):
            current_id = self.data[i][0]
            if current_id != last_id:
                start_indices.append(i)
                last_id = current_id
        return start_indices

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        start_idx = self.start_indices[idx]
        end_idx = self.start_indices[idx + 1] if idx + 1 < len(self.start_indices) else len(self.data)
        return self.data[start_idx:end_idx]


def time_convert(times, time_interval):
    return torch.Tensor((times[:, :, 2] + times[:, :, 1] * 60 + times[:, :, 0] * 3600) // time_interval).long()


def savecheckpoint(state, file_name):
    torch.save(state, file_name)


class train_mst_oatd:
    def __init__(self, s_token_size, t_token_size, time_interval, args):

        self.MST_OATD_S = MST_OATD(s_token_size, s_token_size, args).to(args.device)
        self.MST_OATD_T = MST_OATD(s_token_size, t_token_size, args).to(args.device)

        self.device = args.device
        self.dataset = args.dataset
        self.n_cluster = args.n_cluster
        self.hidden_size = args.hidden_size

        self.crit = nn.CrossEntropyLoss()
        self.detec = nn.CrossEntropyLoss(reduction='none')

        self.pretrain_optimizer_s = optim.AdamW([
            {'params': self.MST_OATD_S.parameters()},
        ], lr=args.pretrain_lr_s)

        self.pretrain_optimizer_t = optim.AdamW([
            {'params': self.MST_OATD_T.parameters()},
        ], lr=args.pretrain_lr_t)

        self.optimizer_s = optim.AdamW([
            {'params': self.MST_OATD_S.parameters()},
        ], lr=args.lr_s)

        self.optimizer_t = optim.Adam([
            {'params': self.MST_OATD_T.parameters()},
        ], lr=args.lr_t)

        self.lr_pretrain_s = StepLR(self.pretrain_optimizer_s, step_size=2, gamma=0.9)
        self.lr_pretrain_t = StepLR(self.pretrain_optimizer_t, step_size=2, gamma=0.9)

        self.pretrained_path = f'models/pretrain_mstoatd_{args.dataset}.pth'
        self.path_checkpoint = f'models/mstoatd_{args.dataset}.pth'
        self.gmm_path = f"models/gmm_{args.dataset}.pt"
        self.gmm_update_path = f"models/gmm_update_{args.dataset}.pt"
        self.logger = get_logger(f"./logs/{args.dataset}.log")

        self.time_interval = time_interval
        self.mode = 'train'

        self.s1_size = args.s1_size
        self.s2_size = args.s2_size

    def pretrain(self, train_loader, epoch):
        self.MST_OATD_S.train()
        self.MST_OATD_T.train()
        epo_loss = 0

        for batch in train_loader:
            trajectories, timestamps, trajectory_lengths = batch
            batch_size = len(trajectories)

            mask = make_mask(make_len_mask(trajectories)).to(self.device)

            self.pretrain_optimizer_s.zero_grad()
            self.pretrain_optimizer_t.zero_grad()

            output_s, _, _, _ = self.MST_OATD_S(trajectories, timestamps, trajectory_lengths, batch_size, "pretrain", -1)
            output_t, _, _, _ = self.MST_OATD_T(trajectories, timestamps, trajectory_lengths, batch_size, "pretrain", -1)

            timestamps = time_convert(timestamps, self.time_interval)

            loss = self.crit(output_s[mask == 1], trajectories.to(self.device)[mask == 1])
            loss += self.crit(output_t[mask == 1], timestamps.to(self.device)[mask == 1])

            loss.backward()

            self.pretrain_optimizer_s.step()
            self.pretrain_optimizer_t.step()
            epo_loss += loss.item()

        self.lr_pretrain_s.step()
        self.lr_pretrain_t.step()
        epo_loss = "%.4f" % (epo_loss / len(train_loader))
        self.logger.info(f"Epoch {epoch + 1} pretrain loss: {epo_loss}")
        checkpoint = {
            "model_state_dict_s": self.MST_OATD_S.state_dict(),
            "model_state_dict_t": self.MST_OATD_T.state_dict()
        }
        torch.save(checkpoint, self.pretrained_path)

    def get_hidden(self, train_loader):
        checkpoint = torch.load(self.path_checkpoint, weights_only=False)
        self.MST_OATD_S.load_state_dict(checkpoint['model_state_dict_s'])
        self.MST_OATD_S.eval()
        with torch.no_grad():
            z = []
            for batch in train_loader:
                trajs, times, seq_lengths = batch
                batch_size = len(trajs)
                _, _, _, hidden = self.MST_OATD_S(trajs, times, seq_lengths, batch_size, "pretrain", -1)
                z.append(hidden.squeeze(0))
            z = torch.cat(z, dim=0)
        return z

    def train_gmm(self, train_loader):
        self.MST_OATD_S.eval()
        self.MST_OATD_T.eval()
        checkpoint = torch.load(self.pretrained_path, weights_only=False)
        self.MST_OATD_S.load_state_dict(checkpoint['model_state_dict_s'])
        self.MST_OATD_T.load_state_dict(checkpoint['model_state_dict_t'])

        with torch.no_grad():
            z_s = []
            z_t = []
            for batch in train_loader:
                trajs, times, seq_lengths = batch
                batch_size = len(trajs)
                _, _, _, hidden_s = self.MST_OATD_S(trajs, times, seq_lengths, batch_size, "pretrain", -1)
                _, _, _, hidden_t = self.MST_OATD_T(trajs, times, seq_lengths, batch_size, "pretrain", -1)

                z_s.append(hidden_s.squeeze(0))
                z_t.append(hidden_t.squeeze(0))
            z_s = torch.cat(z_s, dim=0)
            z_t = torch.cat(z_t, dim=0)

        self.logger.info('Start fitting gaussian mixture model!')

        self.gmm_s = GaussianMixture(n_components=self.n_cluster, covariance_type="diag", n_init=1)
        self.gmm_s.fit(z_s.cpu().numpy())

        self.gmm_t = GaussianMixture(n_components=self.n_cluster, covariance_type="diag", n_init=1)
        self.gmm_t.fit(z_t.cpu().numpy())

    def save_weights_for_MSTOATD(self):
        savecheckpoint({"gmm_s_mu_prior": self.gmm_s.means_,
                        "gmm_s_pi_prior": self.gmm_s.weights_,
                        "gmm_s_logvar_prior": self.gmm_s.covariances_,
                        "gmm_t_mu_prior": self.gmm_t.means_,
                        "gmm_t_pi_prior": self.gmm_t.weights_,
                        "gmms_t_logvar_prior": self.gmm_t.covariances_}, self.gmm_path)

    def train_gmm_update(self, train_loader):

        checkpoint = torch.load(self.path_checkpoint, weights_only=False)
        self.MST_OATD_S.load_state_dict(checkpoint['model_state_dict_s'])
        self.MST_OATD_S.eval()

        with torch.no_grad():
            z = []
            for batch in train_loader:
                trajs, times, seq_lengths = batch
                batch_size = len(trajs)
                _, _, _, hidden = self.MST_OATD_S(trajs, times, seq_lengths, batch_size, "pretrain", -1)
                z.append(hidden.squeeze(0))
            z = torch.cat(z, dim=0)

        self.logger.info('Start fitting gaussian mixture model!')

        self.gmm = GaussianMixture(n_components=self.n_cluster, covariance_type="diag", n_init=3)
        self.gmm.fit(z.cpu().numpy())

        savecheckpoint({"gmm_update_weights": self.gmm.weights_,
                        "gmm_update_means": self.gmm.means_,
                        "gmm_update_covariances": self.gmm.covariances_,
                        "gmm_update_precisions_cholesky": self.gmm.precisions_cholesky_}, self.gmm_update_path)

    def train(self, train_loader, epoch):
        self.MST_OATD_S.train()
        self.MST_OATD_T.train()
        total_loss = 0
        for batch in train_loader:
            trajs, times, seq_lengths = batch
            batch_size = len(trajs)

            mask = make_mask(make_len_mask(trajs)).to(self.device)

            self.optimizer_s.zero_grad()
            self.optimizer_t.zero_grad()

            x_hat_s, mu_s, log_var_s, z_s = self.MST_OATD_S(trajs, times, seq_lengths, batch_size, "train", -1)
            loss = self.Loss(x_hat_s, trajs.to(self.device), mu_s.squeeze(0), log_var_s.squeeze(0),
                             z_s.squeeze(0), 's', mask)
            x_hat_t, mu_t, log_var_t, z_t = self.MST_OATD_T(trajs, times, seq_lengths, batch_size, "train", -1)
            times = time_convert(times, self.time_interval)
            loss += self.Loss(x_hat_t, times.to(self.device), mu_t.squeeze(0), log_var_t.squeeze(0),
                              z_t.squeeze(0), 't', mask)

            loss.backward()
            self.optimizer_s.step()
            self.optimizer_t.step()
            total_loss += loss.item()

        if self.mode == "train":
            total_loss = "%.4f" % (total_loss / len(train_loader))
            self.logger.info(f'Epoch {epoch + 1} loss: {total_loss}')
            checkpoint = {"model_state_dict_s": self.MST_OATD_S.state_dict(),
                          "model_state_dict_t": self.MST_OATD_T.state_dict()}
            torch.save(checkpoint, self.path_checkpoint)

    def detection(self, outliers_loader, labels):

        self.MST_OATD_S.eval()
        all_likelihood_s = []
        self.MST_OATD_T.eval()
        all_likelihood_t = []

        with torch.no_grad():

            for batch in outliers_loader:
                trajs, times, seq_lengths = batch
                batch_size = len(trajs)
                mask = make_mask(make_len_mask(trajs)).to(self.device)
                times_token = time_convert(times, self.time_interval)

                c_likelihood_s = []
                c_likelihood_t = []

                for c in range(self.n_cluster):
                    output_s, _, _, _ = self.MST_OATD_S(trajs, times, seq_lengths, batch_size, "test", c)
                    likelihood_s = - self.detec(output_s.reshape(-1, output_s.shape[-1]),
                                                trajs.to(self.device).reshape(-1))
                    likelihood_s = torch.exp(
                        torch.sum(mask * (likelihood_s.reshape(batch_size, -1)), dim=-1) / torch.sum(mask, 1))

                    output_t, _, _, _ = self.MST_OATD_T(trajs, times, seq_lengths, batch_size, "test", c)
                    likelihood_t = - self.detec(output_t.reshape(-1, output_t.shape[-1]),
                                                times_token.to(self.device).reshape(-1))
                    likelihood_t = torch.exp(
                        torch.sum(mask * (likelihood_t.reshape(batch_size, -1)), dim=-1) / torch.sum(mask, 1))

                    c_likelihood_s.append(likelihood_s.unsqueeze(0))
                    c_likelihood_t.append(likelihood_t.unsqueeze(0))

                all_likelihood_s.append(torch.cat(c_likelihood_s).max(0)[0])
                all_likelihood_t.append(torch.cat(c_likelihood_t).max(0)[0])

        likelihood_s = torch.cat(all_likelihood_s, dim=0)
        likelihood_t = torch.cat(all_likelihood_t, dim=0)

        pr_auc = auc_score(labels, (1 - likelihood_s * likelihood_t).cpu().detach().numpy())
        return pr_auc

    def gaussian_pdf_log(self, x, mu, log_var):
        return -0.5 * (torch.sum(np.log(np.pi * 2) + log_var + (x - mu).pow(2) / torch.exp(log_var), 1))

    def gaussian_pdfs_log(self, x, mus, log_vars):
        G = []
        for c in range(self.n_cluster):
            G.append(self.gaussian_pdf_log(x, mus[c:c + 1, :], log_vars[c:c + 1, :]).view(-1, 1))
        return torch.cat(G, 1)

    def Loss(self, x_hat, targets, z_mu, z_sigma2_log, z, mode, mask):
        if mode == 's':
            pi = self.MST_OATD_S.pi_prior
            log_sigma2_c = self.MST_OATD_S.log_var_prior
            mu_c = self.MST_OATD_S.mu_prior
        elif mode == 't':
            pi = self.MST_OATD_T.pi_prior
            log_sigma2_c = self.MST_OATD_T.log_var_prior
            mu_c = self.MST_OATD_T.mu_prior

        reconstruction_loss = self.crit(x_hat[mask == 1], targets[mask == 1])

        gaussian_loss = torch.mean(torch.mean(self.gaussian_pdf_log(z, z_mu, z_sigma2_log).unsqueeze(1) -
                                              self.gaussian_pdfs_log(z, mu_c, log_sigma2_c), dim=1), dim=-1).mean()

        z = z.unsqueeze(1)
        mu_c = mu_c.unsqueeze(0)
        log_sigma2_c = log_sigma2_c.unsqueeze(0)

        logits = - torch.sum(torch.pow(z - mu_c, 2) / torch.exp(log_sigma2_c), dim=-1)
        logits = F.softmax(logits, dim=-1) + 1e-10
        category_loss = torch.mean(torch.sum(logits * (torch.log(logits) - torch.log(pi).unsqueeze(0)), dim=-1))

        loss = reconstruction_loss + gaussian_loss / self.hidden_size + category_loss * 0.1
        return loss

    def load_mst_oatd(self):
        checkpoint = torch.load(self.pretrained_path, weights_only=False)
        self.MST_OATD_S.load_state_dict(checkpoint['model_state_dict_s'])
        self.MST_OATD_T.load_state_dict(checkpoint['model_state_dict_t'])

        gmm_params = torch.load(self.gmm_path, weights_only=False)

        self.MST_OATD_S.pi_prior.data = torch.from_numpy(gmm_params['gmm_s_pi_prior']).to(self.device)
        self.MST_OATD_S.mu_prior.data = torch.from_numpy(gmm_params['gmm_s_mu_prior']).to(self.device)
        self.MST_OATD_S.log_var_prior.data = torch.from_numpy(gmm_params['gmm_s_logvar_prior']).to(self.device)

        self.MST_OATD_T.pi_prior.data = torch.from_numpy(gmm_params['gmm_t_pi_prior']).to(self.device)
        self.MST_OATD_T.mu_prior.data = torch.from_numpy(gmm_params['gmm_t_mu_prior']).to(self.device)
        self.MST_OATD_T.log_var_prior.data = torch.from_numpy(gmm_params['gmms_t_logvar_prior']).to(self.device)

    def get_prob(self, z):
        gmm = GaussianMixture(n_components=self.n_cluster, covariance_type='diag')
        gmm_params = torch.load(self.gmm_update_path, weights_only=False)
        gmm.precisions_cholesky_ = gmm_params['gmm_update_precisions_cholesky']
        gmm.weights_ = gmm_params['gmm_update_weights']
        gmm.means_ = gmm_params['gmm_update_means']
        gmm.covariances_ = gmm_params['gmm_update_covariances']

        probs = gmm.predict_proba(z)

        for label in range(self.n_cluster):
            np.save(f'probs/probs_{label}_{self.dataset}.npy', np.sort(-probs[:, label]))

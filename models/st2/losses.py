import os
import numpy as np


class SemiSupLossGMM_M:
    def __init__(self, iters=10):
        self.iters = iters
        self.heads = {1: None, 0: None}

    def norm_pdf(self, x, m, s):
        s = np.maximum(s, 1e-6)
        return np.exp(-0.5 * ((x - m) / s) ** 2) / (np.sqrt(2 * np.pi) * s)

    def clean_posterior(self, p_c, p_n):
        return p_c / (p_c + p_n + 1e-12)

    def fit_single_head(self, loss, is_tr):
        u = np.log(loss + 1e-6)
        u_tr = u[is_tr]
        mu_tr = float(u_tr.mean())
        std_tr = float(u_tr.std() + 1e-6)
        z = (u - mu_tr) / std_tr
        z_tr = z[is_tr]

        mu_c = float(z_tr.mean())
        sig_c = float(z_tr.std() + 1e-3)
        z_te = z[~is_tr]
        mu_n = float(np.clip(np.quantile(z_te, 0.8), mu_c + 0.5, mu_c + 3.0))
        sig_n = float(z_te.std() + 1e-3)
        pi_c = (is_tr.sum() + 0.5 * z_te.size) / (z.size + 1e-12)
        pi_n = 1.0 - pi_c
        q10, q25, q50, q75, q90 = np.quantile(z_tr, [0.10, 0.25, 0.50, 0.75, 0.90])
        clean_iqr = float(max(q75 - q25, 1e-3))

        for _ in range(self.iters):
            p_c = pi_c * self.norm_pdf(z, mu_c, sig_c)
            p_n = pi_n * self.norm_pdf(z, mu_n, sig_n)
            gamma_c = p_c / (p_c + p_n + 1e-12)
            gamma_c[is_tr] = 1.0

            nc = gamma_c.sum()
            mu_c = float((gamma_c * z).sum() / (nc + 1e-12))
            sig_c = float(np.sqrt((gamma_c * (z - mu_c) ** 2).sum() / (nc + 1e-12) + 1e-6))

            mask_te = ~is_tr
            gamma_n_te = (1.0 - gamma_c) * mask_te.astype(np.float64)
            nn = gamma_n_te.sum()
            if nn < 1e-6:
                mu_n = mu_c + 2.0
                sig_n = max(sig_c, 1.0)
            else:
                mu_n = float((gamma_n_te * z).sum() / (nn + 1e-12))
                sig_n = float(np.sqrt((gamma_n_te * (z - mu_n) ** 2).sum() / (nn + 1e-12) + 1e-6))

            pi_c = float((is_tr.sum() + (gamma_c * (~is_tr)).sum()) / z.size)
            pi_n = 1.0 - pi_c

        return {
            "mu_tr": mu_tr,
            "std_tr": std_tr,
            "params": (pi_c, pi_n, mu_c, sig_c, mu_n, sig_n),
            "clean_quantiles": (float(q10), float(q25), float(q50), float(q75), float(q90), clean_iqr),
        }

    def fit(self, epoch_loss_unique, x_or_c, o_c):
        ce = np.asarray(epoch_loss_unique, dtype=np.float64)
        is_tr = np.asarray([xc == 0 for xc in x_or_c], dtype=bool)
        is_close = np.asarray([oc == 1 for oc in o_c], dtype=bool)

        idx_cl = np.where(is_close)[0]
        idx_op = np.where(~is_close)[0]
        self.heads = {
            1: self.fit_single_head(ce[idx_cl], is_tr[idx_cl]),
            0: self.fit_single_head(ce[idx_op], is_tr[idx_op]),
        }

    def posterior_clean(
        self,
        ce_losses,
        o_c_batch,
        posterior_temperature=1.0,
        clean_rank_weight=0.0,
        clean_rank_temperature=1.0,
    ):
        loss = np.asarray(ce_losses, dtype=np.float64)
        is_close = np.asarray(o_c_batch, dtype=bool)
        w = np.ones_like(ce_losses, dtype=np.float64)
        posterior_temperature = max(float(posterior_temperature), 1e-6)
        clean_rank_weight = min(max(float(clean_rank_weight), 0.0), 1.0)
        clean_rank_temperature = max(float(clean_rank_temperature), 1e-6)

        for head, selector in ((1, is_close), (0, ~is_close)):
            idx = np.where(selector)[0]
            if len(idx) == 0:
                continue
            mu_tr, std_tr = self.heads[head]["mu_tr"], self.heads[head]["std_tr"]
            pi_c, pi_n, mu_c, sig_c, mu_n, sig_n = self.heads[head]["params"]
            z = (np.log(loss[idx] + 1e-6) - mu_tr) / (std_tr + 1e-6)
            p_c = pi_c * self.norm_pdf(z, mu_c, sig_c)
            p_n = pi_n * self.norm_pdf(z, mu_n, sig_n)
            posterior = self.clean_posterior(p_c, p_n)
            if posterior_temperature != 1.0:
                posterior = np.clip(posterior, 1e-6, 1.0 - 1e-6)
                logits = np.log(posterior) - np.log1p(-posterior)
                posterior = 1.0 / (1.0 + np.exp(-logits / posterior_temperature))

            if clean_rank_weight > 0.0:
                q = self.heads[head].get("clean_quantiles", (0.0, -0.674, 0.0, 0.674, 1.282, 1.349))
                q75 = float(q[3])
                clean_iqr = max(float(q[5]), 1e-3)
                rank_scale = clean_iqr * clean_rank_temperature
                clean_rank_conf = 1.0 / (1.0 + np.exp((z - q75) / rank_scale))
                clean_rank_conf = np.clip(clean_rank_conf, 1e-6, 1.0)
                posterior = (posterior ** (1.0 - clean_rank_weight)) * (clean_rank_conf ** clean_rank_weight)

            w[idx] = np.clip(posterior, 0.0, 1.0)

        return w


def save_gmm_params(gmm, save_dir, epoch):
    os.makedirs(save_dir, exist_ok=True)
    path = f"{save_dir}/gmm_epoch{epoch}.npz"

    def pack(state):
        clean_quantiles = state.get("clean_quantiles", (0.0, -0.674, 0.0, 0.674, 1.282, 1.349))
        return (
            float(state["mu_tr"]),
            float(state["std_tr"]),
        ) + tuple(float(x) for x in state["params"]) + tuple(float(x) for x in clean_quantiles)

    np.savez(path, h1=np.array(pack(gmm.heads[1])), h0=np.array(pack(gmm.heads[0])))


def load_gmm_params(gmm, path):
    d = np.load(path)

    def unpack(arr):
        params = tuple(float(x) for x in arr[2:8])
        if len(arr) >= 14:
            clean_quantiles = tuple(float(x) for x in arr[8:14])
        else:
            clean_quantiles = (0.0, -0.674, 0.0, 0.674, 1.282, 1.349)
        return {
            "mu_tr": float(arr[0]),
            "std_tr": float(arr[1]),
            "params": params,
            "clean_quantiles": clean_quantiles,
        }

    gmm.heads = {1: unpack(d["h1"]), 0: unpack(d["h0"])}

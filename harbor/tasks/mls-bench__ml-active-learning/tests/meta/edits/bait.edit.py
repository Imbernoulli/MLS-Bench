"""BAIT baseline for ml-active-learning.

Reference: vendor/external_packages/badge/query_strategies/bait_sampling.py
Paper: Ash et al. (2021), "Gone Fishing: Neural Active Learning with Fisher
       Embeddings" (NeurIPS 2021)
Uses Fisher information matrices with forward selection + backward pruning.
This task-level implementation keeps the BAIT objective but replaces the
memory-heavy full-pool selection pass with a projected, candidate-filtered
CPU adaptation so the `letter` environment can complete under the repo's
64 GB SLURM limit.
"""

_FILE = "badge/query_strategies/custom_sampling.py"

_CONTENT = """class CustomSampling(Strategy):
    \"\"\"BAIT — Batch Active Learning via Information Matrices (Fisher embeddings).
    CPU-adapted version of the original BAIT algorithm.

    This implementation keeps the Fisher-matrix objective, but makes the
    selection pass tractable on MLS-Bench's CPU setup by:
    1. building Fisher statistics in streaming batches,
    2. projecting very high-dimensional Fisher embeddings before selection,
    3. running BAIT on an entropy-filtered candidate pool instead of the full
       unlabeled set.
    \"\"\"

    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(CustomSampling, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.lamb = args.get('lamb', 1)
        self.max_proj_dim = int(args.get('bait_proj_dim', 128))
        self.candidate_pool = int(args.get('bait_candidate_pool', 0))
        self.selection_batch_size = int(args.get('bait_selection_batch_size', 256))
        self.seed = int(args.get('seed', 42))

    def _make_projection(self, full_dim):
        import torch

        if full_dim <= self.max_proj_dim:
            return None

        generator = torch.Generator(device='cpu')
        generator.manual_seed(self.seed)
        projection = torch.randn(
            full_dim,
            self.max_proj_dim,
            generator=generator,
            dtype=torch.float32,
        )
        projection /= np.sqrt(float(self.max_proj_dim))
        return projection

    def _build_batch_embeddings(self, embedding, probs, projection):
        import torch

        n_lab = probs.shape[1]
        coeffs = -probs.unsqueeze(1).expand(-1, n_lab, -1).clone()
        diag = torch.arange(n_lab)
        coeffs[:, diag, diag] += 1.0
        coeffs *= torch.sqrt(probs.clamp_min(1e-12)).unsqueeze(-1)

        fisher = coeffs.unsqueeze(-1) * embedding.unsqueeze(1).unsqueeze(2)
        fisher = fisher.reshape(embedding.shape[0], n_lab, -1)
        if projection is not None:
            fisher = torch.matmul(fisher, projection)
        return fisher.contiguous()

    def _candidate_pool_size(self, n, total):
        default_size = max(4 * n, 512)
        if self.candidate_pool > 0:
            default_size = self.candidate_pool
        return min(total, default_size)

    def _collect_statistics(self, idxs_unlabeled, n):
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader

        model = self.clf.eval()
        device = next(model.parameters()).device
        n_lab = int(torch.max(self.Y).item() + 1)
        emb_dim = model.get_embedding_dim()
        full_dim = emb_dim * n_lab
        projection = self._make_projection(full_dim)
        target_dim = full_dim if projection is None else projection.shape[1]

        fisher = torch.zeros(target_dim, target_dim, dtype=torch.float32)
        init = torch.zeros(target_dim, target_dim, dtype=torch.float32)
        n_labeled = max(int(np.sum(self.idxs_lb)), 1)
        unlabeled_scores = np.empty(len(idxs_unlabeled), dtype=np.float32)
        pool_to_unlabeled = np.full(self.n_pool, -1, dtype=np.int64)
        pool_to_unlabeled[idxs_unlabeled] = np.arange(len(idxs_unlabeled))

        loader = DataLoader(
            self.handler(self.X, self.Y, transform=self.args['transformTest']),
            shuffle=False,
            **self.args['loader_te_args']
        )

        with torch.no_grad():
            for x, _, idxs in loader:
                x = x.to(device)
                logits, embedding = model(x)
                probs = F.softmax(logits, dim=1).cpu()
                batch_xt = self._build_batch_embeddings(embedding.cpu(), probs, projection)
                fisher += torch.sum(
                    torch.matmul(batch_xt.transpose(1, 2), batch_xt),
                    dim=0,
                ) / float(self.n_pool)

                idxs_np = idxs.numpy()
                labeled_mask = torch.from_numpy(self.idxs_lb[idxs_np])
                if labeled_mask.any():
                    init += torch.sum(
                        torch.matmul(
                            batch_xt[labeled_mask].transpose(1, 2),
                            batch_xt[labeled_mask],
                        ),
                        dim=0,
                    ) / float(n_labeled)

                unlabeled_mask = ~labeled_mask
                if unlabeled_mask.any():
                    unlabeled_rows = pool_to_unlabeled[idxs_np[unlabeled_mask.numpy()]]
                    batch_probs = probs[unlabeled_mask]
                    entropy = -torch.sum(
                        batch_probs * torch.log(batch_probs.clamp_min(1e-12)),
                        dim=1,
                    )
                    unlabeled_scores[unlabeled_rows] = entropy.numpy()

        candidate_count = self._candidate_pool_size(n, len(idxs_unlabeled))
        if candidate_count == len(idxs_unlabeled):
            candidate_local = np.arange(len(idxs_unlabeled))
        else:
            candidate_local = np.argpartition(unlabeled_scores, -candidate_count)[-candidate_count:]
        candidate_local = candidate_local[np.argsort(unlabeled_scores[candidate_local])[::-1]]
        candidate_global = idxs_unlabeled[candidate_local]
        pool_to_candidate = np.full(self.n_pool, -1, dtype=np.int64)
        pool_to_candidate[candidate_global] = np.arange(len(candidate_global))
        candidate_xt = torch.empty(
            len(candidate_global),
            n_lab,
            target_dim,
            dtype=torch.float32,
        )

        with torch.no_grad():
            for x, _, idxs in loader:
                idxs_np = idxs.numpy()
                candidate_rows = pool_to_candidate[idxs_np]
                keep_mask_np = candidate_rows >= 0
                if not keep_mask_np.any():
                    continue

                x = x.to(device)
                logits, embedding = model(x)
                probs = F.softmax(logits, dim=1).cpu()
                batch_xt = self._build_batch_embeddings(embedding.cpu(), probs, projection)
                keep_mask = torch.from_numpy(keep_mask_np)
                candidate_xt[torch.from_numpy(candidate_rows[keep_mask_np])] = batch_xt[keep_mask]

        return fisher, init, candidate_global, candidate_xt

    def _trace_scores(self, xt_batch, current_inv, fisher, add_identity):
        import torch

        rank = xt_batch.shape[-2]
        eye = torch.eye(rank, dtype=xt_batch.dtype).unsqueeze(0)
        sign = 1.0 if add_identity else -1.0
        info = current_inv @ fisher @ current_inv
        gram = torch.matmul(torch.matmul(xt_batch, current_inv), xt_batch.transpose(1, 2))
        inner = gram + sign * eye
        inner_inv = torch.linalg.pinv(inner)
        middle = torch.matmul(torch.matmul(xt_batch, info), xt_batch.transpose(1, 2))
        scores = torch.diagonal(
            torch.matmul(middle, inner_inv),
            dim1=-2,
            dim2=-1,
        ).sum(-1)
        finfo = torch.finfo(scores.dtype)
        return torch.nan_to_num(scores, nan=-finfo.max, posinf=finfo.max, neginf=-finfo.max)

    def _woodbury_update(self, current_inv, xt_sample, add_identity):
        import torch

        xt_sample = xt_sample.unsqueeze(0)
        rank = xt_sample.shape[-2]
        eye = torch.eye(rank, dtype=xt_sample.dtype).unsqueeze(0)
        sign = 1.0 if add_identity else -1.0

        current = current_inv.unsqueeze(0)
        inner = torch.matmul(torch.matmul(xt_sample, current), xt_sample.transpose(1, 2))
        inner_inv = torch.linalg.pinv(inner + sign * eye)
        updated = current - torch.matmul(
            torch.matmul(torch.matmul(current, xt_sample.transpose(1, 2)), inner_inv),
            torch.matmul(xt_sample, current),
        )
        return updated[0].contiguous()

    def _best_forward_index(self, xt_scaled, current_inv, fisher, selected_mask):
        import torch

        best_idx = None
        best_score = -float('inf')
        for start in range(0, len(xt_scaled), self.selection_batch_size):
            end = min(start + self.selection_batch_size, len(xt_scaled))
            batch = xt_scaled[start:end]
            scores = self._trace_scores(batch, current_inv, fisher, add_identity=True)
            batch_mask = selected_mask[start:end]
            if np.any(batch_mask):
                scores[torch.from_numpy(batch_mask)] = -torch.finfo(scores.dtype).max
            score, local_idx = torch.max(scores, dim=0)
            score = score.item()
            if score > best_score:
                best_score = score
                best_idx = start + local_idx.item()
        return best_idx

    def query(self, n):
        import gc
        import torch

        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        fisher, init, candidate_global, xt_unlabeled = self._collect_statistics(
            idxs_unlabeled,
            n,
        )
        if len(candidate_global) <= n:
            return candidate_global

        n_labeled = int(np.sum(self.idxs_lb))
        K = n
        denom = float(max(n_labeled + K, 1))
        dim = xt_unlabeled.shape[-1]
        currentInv = torch.linalg.pinv(
            self.lamb * torch.eye(dim, dtype=torch.float32)
            + init * n_labeled / denom
        )
        xt_scaled = xt_unlabeled * np.sqrt(K / denom)

        indsAll = []
        selected_mask = np.zeros(len(candidate_global), dtype=bool)
        over_sample = 2

        for _ in range(min(int(over_sample * K), len(candidate_global))):
            ind = self._best_forward_index(xt_scaled, currentInv, fisher, selected_mask)
            if ind is None:
                break

            indsAll.append(ind)
            selected_mask[ind] = True
            currentInv = self._woodbury_update(
                currentInv,
                xt_scaled[ind],
                add_identity=True,
            )

        for _ in range(len(indsAll) - K):
            xt_selected = xt_scaled[indsAll]
            traceEst = self._trace_scores(xt_selected, currentInv, fisher, add_identity=False)
            delInd = torch.argmax(traceEst).item()
            currentInv = self._woodbury_update(
                currentInv,
                xt_scaled[indsAll[delInd]],
                add_identity=False,
            )
            del indsAll[delInd]

        gc.collect()
        return candidate_global[np.asarray(indsAll, dtype=int)]
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 28,
        "end_line": 54,
        "content": _CONTENT,
    },
]

import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import random
import os
import tempfile
import time
import gc
import scipy.sparse as sp
from .DNN import DNN

class ARDSR(nn.Module):
    def __init__(self, data, args, beta_fixed=True):
        super(ARDSR, self).__init__()
        self.data = data
        self.n_users = data.n_users
        self.args = args
        self.beta_fixed = beta_fixed
        
        # Noise settings
        self.noise_scale = args.noise_scale
        self.noise_min = args.noise_min
        self.noise_max = args.noise_max
        self.steps = args.steps
        self.device = args.device

        self.data_dir = os.path.join(args.data_path, args.dataset)
        self.temp = args.temp
        self.ddim = args.ddim

        # Seed setting
        if hasattr(args, 'seed'):
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            random.seed(args.seed)

        # Build MLP
        self.hidden_units = args.hidden_units
        self.input_trans = nn.Linear(self.args.embed_dim, self.args.embed_dim)
        
        dim = self.n_users + self.args.embed_dim + self.args.time_size
        out_dims = [self.hidden_units, self.n_users]
        in_dims = [dim, self.hidden_units]
        self.MLP = DNN(args, in_dims, out_dims).to(self.device)

        # Pre-calculate base variance
        self.variance_base = self.get_base_betas()
        self.cached_step_lookup = None

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def get_base_betas(self):
        start = self.noise_scale * self.noise_min
        end = self.noise_scale * self.noise_max
        variance_base = np.linspace(start, end, self.steps, dtype=np.float32)
        variance_base = torch.tensor(variance_base, dtype=torch.float32, device=self.device)
        if self.beta_fixed:
            variance_base[0] = 1e-5
        return variance_base

    def get_batch_betas(self, user_embed, idx, expand_steps=True):
        # [Optimization] Matrix Multiplication for Cosine Similarity
        # A: (Batch, Dim), B: (N_users, Dim) -> Result: (Batch, N_users)
        
        # 1. Normalize vectors for fast cosine similarity
        A = F.normalize(user_embed[idx, :], p=2, dim=1)
        B = F.normalize(user_embed, p=2, dim=1)
        
        # 2. Matrix multiplication (Much faster than broadcasting)
        # (Batch, Dim) @ (Dim, N_users) -> (Batch, N_users)
        cos_similarities = torch.matmul(A, B.t())
        
        gamma1 = cos_similarities
        gamma2 = 1 - 0.01 * torch.exp(self.temp * gamma1) # (Batch, N_users)
        
        if not expand_steps:
            return gamma2

        # Expand variance_base: (Steps) -> (Steps, 1, 1)
        var_base_view = self.variance_base.view(-1, 1, 1)

        # Expand gamma2: (Batch, N_users) -> (1, Batch, N_users)
        gamma2_view = gamma2.unsqueeze(0)

        # Broadcasting handles the rest: (Steps, Batch, N_users)
        score = gamma2_view * var_base_view

        return score

    def _map_cached_steps(self, t):
        if self.cached_step_lookup is None:
            return t

        if isinstance(t, int):
            t_tensor = torch.tensor([t], device=self.cached_step_lookup.device)
            mapped = self.cached_step_lookup[t_tensor][0].item()
            if mapped < 0:
                raise IndexError(f"Timestep {t} not cached for diffusion schedule")
            return mapped

        mapped = self.cached_step_lookup[t]
        if (mapped < 0).any():
            missing = t[mapped < 0]
            raise IndexError(f"Timesteps {missing.tolist()} not cached for diffusion schedule")
        return mapped

    def calculate_batch_for_diffusion(self, user_embed, idx, selected_steps=None):
        """
        Calculate diffusion statistics for a given batch.

        When ``selected_steps`` is provided, only the timesteps that will
        actually be used (and their previous steps) are cached. This avoids
        allocating a full (steps x batch x n_users) tensor, which was the
        major VRAM spike on large datasets such as Epinions when ``steps`` is
        50.
        """

        gamma2 = self.get_batch_betas(user_embed, idx, expand_steps=False)
        # Avoid negative or >1 scales that can create invalid betas leading to
        # NaNs in sqrt(1 - alpha) when alphas drift outside [0, 1].
        gamma2 = torch.clamp(gamma2, min=0.0, max=1.0)

        # Track which steps must be materialized. For training we only need
        # the sampled timesteps (and their predecessors for SNR weighting).
        if selected_steps is not None:
            selected_steps = selected_steps.detach()
            needed = torch.unique(torch.cat([
                selected_steps,
                torch.clamp(selected_steps - 1, min=0)
            ])).tolist()
        else:
            needed = list(range(self.steps))

        max_needed = max(needed)

        # Estimate cache size (we store ~4 tensors of this shape). If the
        # estimate exceeds ~1.5GB, fall back to CPU caching to avoid GPU OOM
        # during large-batch inference on datasets like Epinions.
        # During training (selected_steps provided) we only store a handful of
        # timesteps, so keep full precision to avoid underflow/NaN issues. For
        # full-schedule caching (sampling), use float16 to limit VRAM unless the
        # caller explicitly opts out by providing selected_steps.
        cache_dtype = torch.float32 if selected_steps is not None else torch.float16
        element_size = torch.tensor([], dtype=cache_dtype).element_size()
        est_bytes = len(needed) * gamma2.numel() * element_size * 4
        cache_device = self.device
        if selected_steps is None:
            # Heuristic: keep caches on GPU if we have enough free memory;
            # otherwise, offload to CPU to avoid OOM. This reduces CPU<->GPU
            # traffic for moderate schedules (e.g., 100 steps) while still
            # protecting against memory spikes on very large batches.
            if torch.cuda.is_available() and self.device.type == "cuda":
                free_mem, total_mem = torch.cuda.mem_get_info(device=self.device)
                # Leave at least 25% headroom on the device.
                gpu_budget = int(free_mem * 0.75)
                # Cap the considered budget to a reasonable upper bound to avoid
                # aggressive allocations on very large GPUs.
                gpu_budget = min(gpu_budget, 12 * 1024 ** 3)
                if est_bytes > gpu_budget:
                    cache_device = torch.device("cpu")
            elif est_bytes > 1.5 * 1024 ** 3:
                cache_device = torch.device("cpu")

        # Keep variance_base on the computation device for beta_t calculation.
        variance_base = self.variance_base.to(gamma2.device)

        # Start with alpha_cumprod = 1 for every user in the batch.
        alpha_cumprod = torch.ones(
            (gamma2.size(0), gamma2.size(1)), device=gamma2.device, dtype=gamma2.dtype
        )

        cached_steps = []
        cached_alpha = []
        cached_prev = []
        cached_beta = []

        for step in range(max_needed + 1):
            beta_t = gamma2 * variance_base[step]
            alpha_t = 1.0 - beta_t

            if step in needed:
                prev = alpha_cumprod
                alpha_cumprod = alpha_cumprod * alpha_t

                cached_steps.append(step)
                cached_beta.append(beta_t.to(cache_device, dtype=cache_dtype))
                cached_prev.append(prev.to(cache_device, dtype=cache_dtype))
                cached_alpha.append(alpha_cumprod.to(cache_device, dtype=cache_dtype))
            else:
                alpha_cumprod = alpha_cumprod * alpha_t

        # Stack cached tensors back to (len(needed), batch, n_users)
        alphas_cumprod = torch.stack(cached_alpha, dim=0)
        alphas_cumprod_prev = torch.stack(cached_prev, dim=0)

        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)

        # Build a lookup tensor so we can map the original timestep to the
        # cached slice without keeping the full schedule in memory.
        step_lookup = torch.full((self.steps,), -1, device=self.device, dtype=torch.long)
        for i, s in enumerate(cached_steps):
            step_lookup[s] = i

        # Store cached schedules
        self.cached_step_lookup = step_lookup
        self.cache_device = cache_device
        self.sqrt_alphas_cumprod = sqrt_alphas_cumprod
        self.sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        # The posterior coefficients are only required for p_mean_variance
        # during sampling. When we restrict to a subset of timesteps we still
        # build the matching cached slices.
        cached_beta = torch.stack(cached_beta, dim=0)

        self.fast_posterior_coef2 = torch.sqrt(1.0 - alphas_cumprod_prev) / sqrt_one_minus_alphas_cumprod
        self.fast_posterior_coef3 = (sqrt_alphas_cumprod * torch.sqrt(1.0 - alphas_cumprod_prev)) / sqrt_one_minus_alphas_cumprod

        denom = (1.0 - alphas_cumprod).clamp(min=1e-12)
        self.posterior_mean_coef1 = (cached_beta * torch.sqrt(alphas_cumprod_prev) / denom)
        self.posterior_mean_coef2 = ((1.0 - alphas_cumprod_prev) * torch.sqrt(1.0 - cached_beta) / denom)

        self.posterior_variance = (cached_beta * (1.0 - alphas_cumprod_prev) / denom)

        # Cleanup intermediate tensors we no longer need
        del gamma2, alpha_cumprod, cached_alpha, cached_prev, cached_beta

    def training_losses(self, idx, x_start, all_embed, all_social_embed):
        batch_size = x_start.size(0)
        ts, pt = self.sample_timesteps(batch_size, x_start.device)

        # 1. 배치 데이터 및 확산 스케줄 계산 (필요한 스텝만 캐싱)
        self.calculate_batch_for_diffusion(all_embed, idx, selected_steps=ts)

        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, ts, noise)

        user_embed = all_embed[idx, :]
        social_embed = all_social_embed[idx, :]

        # 2. Input 계산 최적화
        combined_embed = torch.sigmoid(self.input_trans(user_embed * social_embed))
        input_feats = combined_embed * user_embed

        model_output = self.MLP(x_t, input_feats, ts)

        loss = (x_start - model_output) ** 2
        
        # 3. SNR Weighting 계산
        weight = self.SNR(ts - 1) - self.SNR(ts)
        
        # --- [수정된 부분] ---
        # ts == 0 조건을 (Batch, 1) 형태로 바꿔주어 브로드캐스팅이 가능하게 합니다.
        ts_mask = (ts == 0).view(-1, 1)  # (Batch,) -> (Batch, 1)
        
        # torch.tensor(1.0)도 device를 맞춰줍니다.
        ones = torch.tensor(1.0, device=self.device)
        
        # 이제 ts_mask(64,1)와 weight(64,21991)의 차원이 호환됩니다.
        weight = torch.where(ts_mask, ones, weight)
        # ---------------------
        
        # 기존: weight는 이미 (Batch, N_users) 형태이므로 view(-1,1) 불필요
        # weight = weight.view(-1, 1) # 이 줄은 삭제하거나 주석 처리 (이미 위에서 (Batch, N_users) 형태임)
        
        # 평균 계산
        terms_loss = torch.mean(weight * loss, dim=1)
        terms_loss /= pt

        return {"loss": terms_loss}

    def SNR(self, t):
        # Improved extraction logic
        # t is (Batch,), alphas_cumprod is (Steps, Batch, N_users)
        # We need to gather the alpha for each batch item at its specific timestep

        # Gather logic: select [t[b], b, :] for all b
        # Optimized extract using gather/indexing
        # alphas: (Steps, Batch, N_users) -> Select step t for each batch

        # Create indices for gathering
        batch_indices = torch.arange(t.size(0), device=t.device)

        # Clamp to the earliest cached step to avoid negative lookups when
        # callers request t-1 for t == 0. The cached schedule always includes
        # step 0, so this preserves valid indexing while preventing
        # IndexError from _map_cached_steps.
        t_idx = self._map_cached_steps(torch.clamp(t, min=0))
        # Indexing: alphas[t, batch_indices, :]
        if self.alphas_cumprod.device != t_idx.device:
            t_idx = t_idx.to(self.alphas_cumprod.device)
            batch_indices = batch_indices.to(self.alphas_cumprod.device)
        alpha_t = self.alphas_cumprod[t_idx, batch_indices, :]
        if alpha_t.device != t.device:
            alpha_t = alpha_t.to(t.device)

        return alpha_t / (1 - alpha_t)

    def sample_timesteps(self, batch_size, device):
        t = torch.randint(0, self.steps, (batch_size,), device=device).long()
        pt = torch.ones_like(t).float()
        return t, pt

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        # Efficient extraction
        batch_indices = torch.arange(t.size(0), device=t.device)

        t_idx = self._map_cached_steps(t)
        cache_device = self.sqrt_alphas_cumprod.device
        if cache_device != t_idx.device:
            t_idx = t_idx.to(cache_device)
            batch_indices = batch_indices.to(cache_device)

        sqrt_alphas = self.sqrt_alphas_cumprod[t_idx, batch_indices, :]
        sqrt_one_minus_alphas = self.sqrt_one_minus_alphas_cumprod[t_idx, batch_indices, :]

        if sqrt_alphas.device != x_start.device:
            sqrt_alphas = sqrt_alphas.to(x_start.device)
            sqrt_one_minus_alphas = sqrt_one_minus_alphas.to(x_start.device)

        return sqrt_alphas * x_start + sqrt_one_minus_alphas * noise

    def p_mean_variance(self, x, t, batch_embed):
        # t는 여기서 정수(int) 값으로 들어옵니다 (Loop에서 i를 넘겨줌)

        mapped_t = self._map_cached_steps(t)

        # 1. Variance 추출 (정수 인덱싱은 텐서에서도 작동하므로 그대로 둡니다)
        coef_device = self.posterior_variance.device
        if not torch.is_tensor(mapped_t):
            mapped_t = torch.tensor(mapped_t, device=coef_device, dtype=torch.long)
        elif coef_device != mapped_t.device:
            mapped_t = mapped_t.to(coef_device)
        model_variance = self.posterior_variance[mapped_t]

        # 2. MLP 입력용 Timestep 텐서 생성 [수정된 부분]
        # 정수 t를 사용하여 (Batch_Size,) 크기의 텐서를 만듭니다.
        # 기존: t.repeat(x.size(0)) -> 에러 원인 (int에는 repeat가 없음)
        t_batch = torch.full((x.size(0),), t, device=self.device, dtype=torch.long)

        # 3. 모델 예측
        model_output = self.MLP(x, batch_embed, t_batch)

        pred_xstart = model_output
        if model_variance.device != x.device:
            # Move all cached slices required for this step to the current device
            model_variance = model_variance.to(x.device)

        # 정수 t를 사용하여 계수(Coefficient) 추출
        if model_variance.device != x.device:
            # Move all cached slices required for this step to the current device
            model_variance = model_variance.to(x.device)

        if not self.ddim:
            coef1 = self.posterior_mean_coef1[mapped_t]
            coef2 = self.posterior_mean_coef2[mapped_t]
            if coef1.device != x.device:
                coef1 = coef1.to(x.device)
                coef2 = coef2.to(x.device)
            model_mean = coef1 * pred_xstart + coef2 * x
        else:
            sqrt_alpha_prev = torch.sqrt(self.alphas_cumprod_prev[mapped_t])
            coef2 = self.fast_posterior_coef2[mapped_t]
            coef3 = self.fast_posterior_coef3[mapped_t]
            if sqrt_alpha_prev.device != x.device:
                sqrt_alpha_prev = sqrt_alpha_prev.to(x.device)
                coef2 = coef2.to(x.device)
                coef3 = coef3.to(x.device)
            model_mean = sqrt_alpha_prev * pred_xstart + coef2 * x - coef3 * pred_xstart

        return {"mean": model_mean, "variance": model_variance, "log_variance": torch.log(model_variance.clamp(min=1e-20))}

    @torch.no_grad()
    def p_sample(self, x_start, idx, all_embed, all_social, steps, noise_step, sampling_noise=False):
        self.calculate_batch_for_diffusion(all_embed, idx)
        
        if noise_step == 0:
            x_t = x_start
        else:
            # t_tensor = torch.full((x_start.size(0),), noise_step - 1, device=x_start.device, dtype=torch.long)
            # Use q_sample directly
            # For inference, q_sample expects t as a batch of indices.
            # But here let's simplify: usually we start from pure noise for generation, or noisy input for refinement.
            # Re-using q_sample logic requires proper t tensor
            t_tensor = torch.tensor([noise_step - 1] * x_start.shape[0], device=x_start.device).long()
            x_t = self.q_sample(x_start, t_tensor)

        reverse_step = steps // 10 if self.ddim else steps
        indices = list(range(reverse_step))[::-1]
        
        batch_embed = all_embed[idx, :]
        social_embed = all_social[idx, :]
        
        combined_embed = torch.sigmoid(self.input_trans(batch_embed * social_embed))
        input_embed = combined_embed * batch_embed

        for i in indices:
            # Pass scalar t for simpler indexing inside p_mean_variance
            out = self.p_mean_variance(x_t, i, input_embed)
            
            if sampling_noise and i > 0:
                noise = torch.randn_like(x_t)
                x_t = out["mean"] + torch.exp(0.5 * out["log_variance"]) * noise
            else:
                x_t = out["mean"]

        return x_t


# ---------------------------------------------------------------------------- #
# Optimized Helper Functions
# ---------------------------------------------------------------------------- #

def cosine_similarity_batched_fast(all_embed, batch_size=2048):
    # Normalize once
    all_embed_norm = F.normalize(all_embed, p=2, dim=1)
    
    # We don't need to loop here if we just need it for flip_tensor inside the batch loop.
    # We can compute it on the fly for the batch vs all. 
    # But if we must precompute all-pairs, use matmul block-wise to save memory.
    # For now, let's keep it simple: if N is huge, don't precompute N*N matrix.
    # Strategy: Compute relevant rows inside refine_social.
    return all_embed_norm


def refine_social(diffusion, social_data, score, all_embed, all_social, args, del_threshold, flip=True):
    diffusion.eval()
    
    # [Optimization] Normalize all_embed once for efficient Cosine Sim
    all_embed_norm = F.normalize(all_embed, p=2, dim=1)
    all_embed_norm_t = all_embed_norm.t().contiguous()
    
    num_rows = len(social_data)
    num_cols = social_data.shape[1]
    # [Optimization] Increase batch size based on available GPU memory for faster throughput.
    if args.device != "cpu":
        try:
            free_mem, _ = torch.cuda.mem_get_info(device=args.device)
        except TypeError:
            free_mem, _ = torch.cuda.mem_get_info()
        # Rough estimate: cos_sim + prediction + batch_social + batch_score (float16)
        bytes_per_row = num_cols * 2 * 2
        target_mem = int(free_mem * 0.5)
        est_batch = max(128, min(num_rows, target_mem // max(bytes_per_row, 1)))
        batch_size = min(est_batch, 4096)
        # Cap batch size to limit host RAM usage when materializing CPU slices.
        host_bytes_per_row = num_cols * 2
        host_target = 256 * 1024**2
        host_batch = max(64, host_target // max(host_bytes_per_row, 1))
        batch_size = min(batch_size, host_batch)
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        batch_size = 1024 if num_cols > 5000 else 2048

    # Stream predictions to a disk-backed memmap to avoid keeping a full
    # dense score matrix in RAM. Using float16 halves the footprint.
    tmp_dir = getattr(args, "tmp_dir", None) or tempfile.gettempdir()
    mmap_path = os.path.join(
        tmp_dir,
        f"new_score_{os.getpid()}_{int(time.time())}.mmap",
    )
    score_bytes = num_rows * num_cols * np.dtype(np.float16).itemsize
    max_in_memory_bytes = 0
    use_memmap = score_bytes > max_in_memory_bytes
    if use_memmap:
        new_score_mm = np.lib.format.open_memmap(
            mmap_path, mode="w+", dtype=np.float16, shape=(num_rows, num_cols)
        )
    else:
        try:
            new_score_mm = np.empty((num_rows, num_cols), dtype=np.float16)
        except MemoryError:
            new_score_mm = np.lib.format.open_memmap(
                mmap_path, mode="w+", dtype=np.float16, shape=(num_rows, num_cols)
            )

    # Preallocate CE buffer so we don't re-read the memmap after writing.
    ce_buffer = np.zeros(num_rows, dtype=np.float32)
    
    # Pre-move static large tensors to GPU if VRAM allows, otherwise keep on CPU and slice
    # Assuming social_data and score fit in memory but we slice them

    h_chunks = []
    t_chunks = []
    decay = False

    use_cuda = args.device != "cpu"

    with torch.inference_mode():
        start = 0
        while start < num_rows:
            end = min(start + batch_size, num_rows)

            try:
                idx_tensor = torch.arange(start, end, device=args.device)

                with torch.cuda.amp.autocast(enabled=(args.device != "cpu")):
                    if sp.issparse(social_data):
                        batch_social_np = social_data[start:end].toarray()
                    else:
                        batch_social_np = np.asarray(social_data[start:end])
                        if not batch_social_np.flags["C_CONTIGUOUS"]:
                            batch_social_np = np.ascontiguousarray(batch_social_np)

                    if score is None:
                        batch_score_np = batch_social_np
                    else:
                        batch_score_np = np.asarray(score[start:end])
                        if not batch_score_np.flags["C_CONTIGUOUS"]:
                            batch_score_np = np.ascontiguousarray(batch_score_np)
                    batch_social_cpu = torch.from_numpy(batch_social_np)
                    batch_score_cpu = torch.from_numpy(batch_score_np)
                    if use_cuda:
                        batch_social_cpu = batch_social_cpu.pin_memory()
                        batch_score_cpu = batch_score_cpu.pin_memory()
                    batch_social = batch_social_cpu.to(
                        args.device, dtype=torch.float16, non_blocking=use_cuda
                    )
                    batch_score = batch_score_cpu.to(
                        args.device, dtype=torch.float16, non_blocking=use_cuda
                    )

                    if flip:
                        batch_embed = all_embed_norm[idx_tensor]
                        cos_sim = torch.matmul(batch_embed, all_embed_norm_t)

                        flipped_batch = flip_tensor(batch_social, cos_sim, args.seed)
                        prediction = diffusion.p_sample(
                            flipped_batch,
                            idx_tensor,
                            all_embed,
                            all_social,
                            args.steps,
                            args.steps,
                        )
                    else:
                        prediction = diffusion.p_sample(
                            batch_social, idx_tensor, all_embed, all_social, args.steps, args.steps
                        )

                    prediction = torch.sigmoid(prediction)
                    avg_prediction = args.decay * batch_score + (1 - args.decay) * prediction
            except RuntimeError as exc:
                if use_cuda and "out of memory" in str(exc).lower() and batch_size > 1:
                    torch.cuda.empty_cache()
                    batch_size = max(64, batch_size // 2)
                    continue
                raise

            # Deletion candidates (working on GPU for speed)
            existing_mask = batch_social != 0
            if existing_mask.any():
                rows, cols = torch.nonzero(existing_mask, as_tuple=True)
                edge_logits = avg_prediction[rows, cols]
                under_mask = edge_logits < del_threshold

                if under_mask.any():
                    decay = True
                    del_rows = rows[under_mask]
                    del_cols = cols[under_mask]

                    h_chunks.append(
                        (del_rows + start).to("cpu", dtype=torch.int32).numpy()
                    )
                    t_chunks.append(del_cols.to("cpu", dtype=torch.int32).numpy())

                    # Row-wise add-backs using masked top-k
                    del_counts = torch.bincount(del_rows, minlength=avg_prediction.size(0))
                    candidate_rows = del_counts.nonzero(as_tuple=False).flatten()
                    if candidate_rows.numel() > 0:
                        # Mask existing edges once for all candidate rows
                        masked_scores = avg_prediction[candidate_rows]
                        masked_scores = masked_scores.masked_fill(
                            existing_mask[candidate_rows], -float("inf")
                        )
                        max_k = del_counts.max().item()
                        topk_vals, topk_idx = torch.topk(
                            masked_scores, k=min(max_k, num_cols)
                        )
                        for row_offset, k in zip(candidate_rows, del_counts[candidate_rows]):
                            k_int = int(k.item())
                            if k_int == 0:
                                continue
                            vals = topk_vals[candidate_rows == row_offset][0][:k_int]
                            cols_sel = topk_idx[candidate_rows == row_offset][0][:k_int]
                            valid = torch.isfinite(vals)
                            if valid.any():
                                cols_final = cols_sel[valid]
                                h_chunks.append(
                                    (cols_final.new_full((cols_final.numel(),), row_offset + start))
                                    .to("cpu", dtype=torch.int32)
                                    .numpy()
                                )
                                t_chunks.append(
                                    cols_final.to("cpu", dtype=torch.int32).numpy()
                                )

                # Per-row CE for existing edges (cpu buffer to avoid second pass)
                ce_rows = torch.zeros(end - start, device=args.device)
                counts = torch.zeros(end - start, device=args.device)
                ce_vals = F.binary_cross_entropy_with_logits(
                    edge_logits.float(),
                    torch.ones_like(edge_logits, dtype=torch.float32),
                    reduction="none",
                )
                ce_rows.index_add_(0, rows, ce_vals)
                counts.index_add_(0, rows, torch.ones_like(ce_vals))
                nonzero = counts > 0
                if nonzero.any():
                    ce_rows[nonzero] = ce_rows[nonzero] / counts[nonzero]
                    nz_idx = nonzero.nonzero(as_tuple=False).flatten()
                    ce_buffer[start + nz_idx.cpu().numpy()] = ce_rows[nz_idx].cpu().numpy()

            # Move to CPU immediately to free GPU memory for next batch and write into the memmap slice.
            new_score_mm[start:end] = avg_prediction.detach().to("cpu", dtype=torch.float16).numpy()

            del avg_prediction, prediction, batch_social, batch_score
            start = end

    if isinstance(new_score_mm, np.memmap):
        new_score_mm.flush()
    new_score_cpu = new_score_mm

    if h_chunks:
        h_out = np.concatenate(h_chunks)
        t_out = np.concatenate(t_chunks)
    else:
        h_out = np.empty((0,), dtype=np.int32)
        t_out = np.empty((0,), dtype=np.int32)

    return h_out, t_out, new_score_cpu, decay, ce_buffer

def flip_tensor(batch, cos_similarities, seed):
    # Optimized flip without repeated manual seeding inside batch ops if not strictly necessary
    # If reproducibility is key, keeping seed is fine, but it might slow down slightly.
    # Generating random numbers on GPU is fast.
    
    sigmoid_pos = torch.sigmoid(cos_similarities)
    sigmoid_neg = 1 - sigmoid_pos
    
    flip_prob = torch.where(batch == 1, sigmoid_neg, sigmoid_pos)
    
    # Generate random mask
    rand_mat = torch.rand_like(flip_prob)
    
    # Flip logic
    # If rand < flip_prob, flip (1->0 or 0->1). Else keep.
    # Flip is |1 - batch|
    should_flip = rand_mat < flip_prob
    flipped_batch = torch.where(should_flip, 1 - batch, batch)
    
    return flipped_batch


def to_tensor(coo_mat, args):
    """
    Scipy COO matrix를 PyTorch Sparse Tensor로 변환하는 함수
    """
    values = coo_mat.data
    indices = np.vstack((coo_mat.row, coo_mat.col))

    i = torch.tensor(indices, dtype=torch.long)
    v = torch.tensor(values, dtype=torch.float32)
    
    # torch.sparse_coo_tensor가 더 권장되는 방식입니다.
    # 생성과 동시에 device로 이동합니다.
    return torch.sparse_coo_tensor(i, v, coo_mat.shape, device=args.device)

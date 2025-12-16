import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import random
import os
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

    def get_batch_betas(self, user_embed, idx):
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
        
        # Expand variance_base: (Steps) -> (Steps, 1, 1)
        var_base_view = self.variance_base.view(-1, 1, 1)
        
        # Expand gamma2: (Batch, N_users) -> (1, Batch, N_users)
        gamma2_view = gamma2.unsqueeze(0)
        
        # Broadcasting handles the rest: (Steps, Batch, N_users)
        score = gamma2_view * var_base_view
        
        return score

    def calculate_batch_for_diffusion(self, user_embed, idx):
        # Calculate betas (Steps, Batch, N_users)
        betas = self.get_batch_betas(user_embed, idx)
        alphas = 1.0 - betas

        # Compute cumulative products
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)

        # Padding for previous alpha (add 1 at the beginning)
        # ones_tensor: (1, Batch, N_users)
        ones_tensor = torch.ones((1, alphas.size(1), alphas.size(2)), device=self.device)
        self.alphas_cumprod_prev = torch.cat([ones_tensor, self.alphas_cumprod[:-1]], dim=0)

        # Precompute coefficients
        # Note: All calculations remain on GPU
        self.fast_posterior_coef2 = torch.sqrt(1.0 - self.alphas_cumprod_prev) / self.sqrt_one_minus_alphas_cumprod
        self.fast_posterior_coef3 = (self.sqrt_alphas_cumprod * torch.sqrt(1.0 - self.alphas_cumprod_prev)) / self.sqrt_one_minus_alphas_cumprod
        
        self.posterior_mean_coef1 = (betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod))
        
        self.posterior_variance = (betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))

    def training_losses(self, idx, x_start, all_embed, all_social_embed):
        # [Optimization] Remove GC calls inside training loop
        
        self.calculate_batch_for_diffusion(all_embed, idx)
        
        batch_size = x_start.size(0)
        ts, pt = self.sample_timesteps(batch_size, x_start.device)
        
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, ts, noise)

        user_embed = all_embed[idx, :]
        social_embed = all_social_embed[idx, :]

        # Optimized input calculation
        combined_embed = torch.sigmoid(self.input_trans(user_embed * social_embed))
        input_feats = combined_embed * user_embed

        model_output = self.MLP(x_t, input_feats, ts)

        loss = (x_start - model_output) ** 2
        
        # SNR Weighting
        weight = self.SNR(ts - 1) - self.SNR(ts)
        # Apply mask for t=0
        weight = torch.where(ts == 0, torch.tensor(1.0, device=self.device), weight)
        
        # Expand weight for broadcasting: (Batch,) -> (Batch, 1) -> (Batch, N_users)
        weight = weight.view(-1, 1) 
        
        # Mean over users
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
        # Indexing: alphas[t, batch_indices, :]
        alpha_t = self.alphas_cumprod[t, batch_indices, :]
        
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
        
        sqrt_alphas = self.sqrt_alphas_cumprod[t, batch_indices, :]
        sqrt_one_minus_alphas = self.sqrt_one_minus_alphas_cumprod[t, batch_indices, :]
        
        return sqrt_alphas * x_start + sqrt_one_minus_alphas * noise

    def p_mean_variance(self, x, t, batch_embed):
        # t is a scalar here (current timestep for the whole batch)
        # In inference loop, t is usually constant across batch for synchronized sampling
        
        # Extract based on scalar t
        # Variance is (Steps, Batch, N_users) -> (Batch, N_users)
        model_variance = self.posterior_variance[t] 

        model_output = self.MLP(x, batch_embed, t.repeat(x.size(0)))

        pred_xstart = model_output
        
        # Use simple indexing since t is scalar integer in loop
        if not self.ddim:
            # posterior mean calculation
            coef1 = self.posterior_mean_coef1[t]
            coef2 = self.posterior_mean_coef2[t]
            model_mean = coef1 * pred_xstart + coef2 * x
        else:
            sqrt_alpha_prev = torch.sqrt(self.alphas_cumprod_prev[t])
            coef2 = self.fast_posterior_coef2[t]
            coef3 = self.fast_posterior_coef3[t]
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
    
    num_rows = len(social_data)
    # [Optimization] Increase batch size significantly (e.g., 2048 or 4096)
    batch_size = 4096 
    num_batches = (num_rows + batch_size - 1) // batch_size
    
    all_predictions = []
    
    # Pre-move static large tensors to GPU if VRAM allows, otherwise keep on CPU and slice
    # Assuming social_data and score fit in memory but we slice them
    
    with torch.no_grad():
        for i in range(num_batches):
            start = i * batch_size
            end = min(start + batch_size, num_rows)
            
            # Use slicing directly (assuming input is numpy or tensor)
            # Keeping inputs on CPU until batching is usually better for very large datasets
            idx_tensor = torch.arange(start, end, device=args.device)
            
            batch_social = torch.tensor(social_data[start:end], dtype=torch.float32, device=args.device)
            batch_score = torch.tensor(score[start:end], dtype=torch.float32, device=args.device)
            
            if flip:
                # Compute Cosine Similarity on the fly: (Batch, N)
                # (Batch, Dim) @ (Dim, N) -> (Batch, N)
                batch_embed = all_embed_norm[idx_tensor]
                cos_sim = torch.matmul(batch_embed, all_embed_norm.t())
                
                flipped_batch = flip_tensor(batch_social, cos_sim, args.seed)
                prediction = diffusion.p_sample(flipped_batch, idx_tensor, all_embed, all_social, args.steps, args.steps)
            else:
                prediction = diffusion.p_sample(batch_social, idx_tensor, all_embed, all_social, args.steps, args.steps)
            
            prediction = torch.sigmoid(prediction)
            avg_prediction = args.decay * batch_score + (1 - args.decay) * prediction
            
            # Move to CPU immediately to free GPU memory for next batch
            all_predictions.append(avg_prediction.cpu())
            
    # Concatenate all results
    new_score = torch.cat(all_predictions, dim=0) # (Total_Users, N_items/users)
    
    # [Optimization] GPU-based Top-K and Filtering
    # Moving logic from CPU list comprehension to Torch operations
    
    # 1. Identify Edges to Delete
    # social_data is scipy sparse or numpy array? Assuming numpy dense based on code usage
    social_data_tensor = torch.tensor(social_data, device=args.device) if not torch.is_tensor(social_data) else social_data
    new_score_device = new_score.to(args.device)
    
    # Mask of existing edges
    existing_mask = (social_data_tensor != 0)
    existing_scores = new_score_device[existing_mask]
    
    # Find edges below threshold
    # Get indices of existing edges
    existing_indices = torch.nonzero(existing_mask, as_tuple=False) # (N_edges, 2)
    
    # Filter by score
    scores_on_edges = new_score_device[existing_indices[:, 0], existing_indices[:, 1]]
    under_threshold_mask = scores_on_edges < del_threshold
    
    edges_to_delete_indices = existing_indices[under_threshold_mask]
    edges_to_delete_scores = scores_on_edges[under_threshold_mask]
    
    # Limit deletions (max 1% of edges)
    num_existing = existing_indices.size(0)
    max_deletions = int(num_existing * 0.01)
    
    if edges_to_delete_indices.size(0) > max_deletions:
        # Sort by score ascending to delete lowest scores first
        _, sorted_idx = torch.sort(edges_to_delete_scores)
        edges_to_delete_indices = edges_to_delete_indices[sorted_idx[:max_deletions]]
    
    # Create the base remaining edges (on GPU)
    # We can create a mask to remove deleted edges
    final_mask = existing_mask.clone()
    final_mask[edges_to_delete_indices[:, 0], edges_to_delete_indices[:, 1]] = 0
    
    # 2. Identify Edges to Add
    # Flatten scores to find global top-k, EXCLUDING existing edges
    # To exclude existing, set their score to -infinity temporarily
    temp_scores = new_score_device.clone()
    # Mask out existing edges so we don't re-add them (or keep them via final_mask)
    # Actually, we want to add *new* edges.
    temp_scores[existing_mask] = -float('inf') 
    
    num_to_add = edges_to_delete_indices.size(0)
    
    if num_to_add > 0:
        # Get global top k
        # Flatten
        flat_scores = temp_scores.view(-1)
        _, top_k_indices = torch.topk(flat_scores, num_to_add)
        
        # Convert flat indices back to 2D
        rows_to_add = top_k_indices // new_score.shape[1]
        cols_to_add = top_k_indices % new_score.shape[1]
        
        # Update final mask
        final_mask[rows_to_add, cols_to_add] = 1
        
    # 3. Convert final mask to h_list, t_list (if strictly needed for downstream)
    # or return sparse tensor / adjacency matrix
    final_edges = torch.nonzero(final_mask, as_tuple=False)
    h_list = final_edges[:, 0].tolist()
    t_list = final_edges[:, 1].tolist()
    
    decay = edges_to_delete_indices.size(0) > 0 # Simplified logic
    
    return h_list, t_list, new_score.numpy(), decay

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
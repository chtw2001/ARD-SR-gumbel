import enum
import math
import numpy as np
import torch 
import torch.nn.functional as F
import torch.nn as nn
import random
import sys
from .DNN import DNN
# import networkx as nx
import os, gc


class ARDSR(nn.Module):

    def __init__(self,data,args,beta_fixed=True):
        super(ARDSR, self).__init__()
        self.data=data
        self.n_users=data.n_users
        self.args=args
        self.beta_fixed = beta_fixed
        self.noise_scale = args.noise_scale  
        self.noise_min = args.noise_min
        self.noise_max = args.noise_max

        self.steps = args.steps # sample t
        self.device = args.device

        self.data_dir = os.path.join(args.data_path, args.dataset)

        self.temp = args.temp
        self.ddim = args.ddim
        
        # 재현성을 위한 seed 설정
        if hasattr(args, 'seed'):
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            random.seed(args.seed)

        ### Build MLP ###
        self.hidden_units=args.hidden_units
   
        self.input_trans=nn.Linear(self.args.embed_dim,self.args.embed_dim)
        dim = self.n_users+self.args.embed_dim+self.args.time_size
        out_dims = [self.hidden_units,self.n_users]
        in_dims = [dim,self.hidden_units]
        self.MLP = DNN(args,in_dims, out_dims).to(self.device)
        
        self.variance_base = self.get_base_betas()

    def get_base_betas(self):
        #base scedule
        start = self.noise_scale * self.noise_min
        end = self.noise_scale * self.noise_max
        variance_base = np.linspace(start, end, self.steps, dtype=np.float32)
        variance_base = torch.tensor(variance_base, dtype=torch.float32).to(self.device)#(1,step)  # Convert to tensor 
        if self.beta_fixed:
            variance_base[0] = 0.00001 
        return variance_base

    def get_batch_betas(self,user_embed,idx):
        variance_base_expanded = self.variance_base.unsqueeze(1).unsqueeze(1).expand(self.steps, len(idx), self.n_users)
        # return variance_base_expanded
        # user specific noise schedule
        with torch.no_grad():
            A_expanded = user_embed[idx,:].unsqueeze(1)
            B_expanded = user_embed.unsqueeze(0)
            cos_similarities = F.cosine_similarity(A_expanded, B_expanded, dim=2).to(dtype=torch.float32) #batch*n_users

        gamma1 =  cos_similarities
        gamma2 =    1 - 0.01*torch.exp(self.temp*gamma1)
        score=  gamma2.unsqueeze(0) *variance_base_expanded   #steps*batch*n_users

        del variance_base_expanded, A_expanded, B_expanded, cos_similarities, gamma1, gamma2
        import gc
        torch.cuda.empty_cache()
        gc.collect()
        
        return score
   
    def calculate_batch_for_diffusion(self,user_embed,idx):

        betas= self.get_batch_betas(user_embed,idx)  #step*batch*n_users
        alphas = 1.0 - betas  # Shape: (t, b, m)

        # self.sqrt_recip_alphas=torch.sqrt(1/alphas)

        self.alphas_cumprod = torch.cumprod(alphas, axis=0).to(self.device) # Shape: (t, m, m)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)


        ones_tensor = torch.ones((1,self.alphas_cumprod.size(1), self.alphas_cumprod.size(2)), dtype=torch.float32, device=self.device)
        # zeros_tensor = torch.zeros((1,self.alphas_cumprod.size(1), self.alphas_cumprod.size(2)), dtype=torch.float32, device=self.device)
        self.alphas_cumprod_prev = torch.cat([ones_tensor, self.alphas_cumprod[:-1,:,:]], dim=0)
        # self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:,:,:], zeros_tensor], dim=0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        # self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        # self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        # self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # self.posterior_mean_coef1 = (betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        # self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev)* torch.sqrt(alphas)/ (1.0 - self.alphas_cumprod))   
        self.fast_posterior_coef2=(torch.sqrt(1.0 - self.alphas_cumprod_prev)/self.sqrt_one_minus_alphas_cumprod)
        self.fast_posterior_coef3=(self.sqrt_alphas_cumprod* torch.sqrt(1.0 - self.alphas_cumprod_prev)/self.sqrt_one_minus_alphas_cumprod)
        self.posterior_variance = (betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod) )
        
        del betas, alphas
        import gc
        torch.cuda.empty_cache()
        gc.collect()


    def training_losses(self,idx,x_start,all_embed,all_social_embed):
        
        self.calculate_batch_for_diffusion(all_embed,idx)
        batch_size, device = x_start.size(0), x_start.device
        # ts: random timestep (batch, )
        ts, pt = self.sample_timesteps(batch_size, device)
        # Set seed for reproducible noise generation
        torch.manual_seed(self.args.seed)
        noise = torch.randn_like(x_start)
        
        # (batch, user)
        # random timestep t마다의 noise가 주입 된 noised input
        # bach 내의 user 마다 서로 다른 timestep t를 가진 noise를 주입받음
        x_t = self.q_sample(x_start, ts,noise)

        terms = {}
        user_embed = all_embed[idx,:]
        social_embed = all_social_embed[idx,:]
        
        # (batch, user)
        input= torch.mul(torch.sigmoid(self.input_trans(torch.mul(user_embed,social_embed))),user_embed)

        # noised input, condition, timestep embedding
        model_output = self.MLP(x_t.to(input.dtype),input,ts)

        loss = (x_start - model_output) ** 2
        weight = self.SNR(ts - 1) - self.SNR(ts)
        ts_expand = ts[:, None].expand(len(ts),self.n_users)
        weight = torch.where((ts_expand == 0), torch.tensor(1.0, dtype=weight.dtype).to(self.device), weight)
        terms["loss"] = torch.mean(weight*loss,dim=1)
        terms["loss"] /= pt
        
        del noise, loss, weight, ts, pt, x_t, user_embed, social_embed, input, model_output, ts_expand
        import gc
        torch.cuda.empty_cache()
        gc.collect()

        return terms
    
    def SNR(self, t):
        """
        Compute the signal-to-noise ratio for a single timestep.
        """
        self.alphas_cumprod = self.alphas_cumprod.to(t.device)
        return self.extract_from_tensor(self.alphas_cumprod,t) / (1 - self.extract_from_tensor(self.alphas_cumprod,t))

    def sample_timesteps(self, batch_size, device):
        # Set seed for reproducible timestep sampling
        torch.manual_seed(self.args.seed)
        
        t = torch.randint(0, self.steps, (batch_size,), device=device).long()
        pt = torch.ones_like(t).float()
        return t, pt
   
    def q_sample(self, x_start,t, noise=None):

        #x_start (b*n_user)
        #t,m,m
        if noise is None:
            # Set seed for reproducible noise generation
            torch.manual_seed(self.args.seed)
            noise = torch.randn_like(x_start)
        
        q_t= self.extract_from_tensor(self.sqrt_alphas_cumprod,t) *x_start + self.extract_from_tensor(self.sqrt_one_minus_alphas_cumprod,t)*noise
        
        return q_t
  
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape

        posterior_mean =self.extract_from_tensor(self.posterior_mean_coef1,t) * x_start+self.extract_from_tensor(self.posterior_mean_coef2,t)* x_t
        posterior_variance = self.extract_from_tensor(self.posterior_variance,t)

        return posterior_mean, posterior_variance#, posterior_log_variance_clipped  

    def fast_q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior with ddim
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape

        posterior_mean = self.extract_from_tensor(torch.sqrt(self.alphas_cumprod_prev),t)* x_start\
                        +self.extract_from_tensor(self.fast_posterior_coef2,t)*x_t-self.extract_from_tensor(self.fast_posterior_coef3,t)*x_start

        posterior_variance = self.extract_from_tensor(self.posterior_variance,t)
        
        return posterior_mean, posterior_variance#, posterior_log_variance_clipped
    
    def p_mean_variance(self,x,t,batch_embed):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        """
        B, C = x.shape[:2]
    
        model_variance = self.posterior_variance
        model_variance = self.extract_from_tensor(model_variance,t)

        model_output = self.MLP(x.to(batch_embed.dtype),batch_embed,t)

        pred_xstart = model_output
        if not self.ddim:
            model_mean, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
        else:
            model_mean, _= self.fast_q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
        
        del pred_xstart, model_output
        import gc
        torch.cuda.empty_cache()
        gc.collect()
        
        
        return { "mean": model_mean, "variance": model_variance}#, "log_variance": model_log_variance}
       
    def p_sample(self, x_start,idx,all_embed,all_social,steps,noise_step, sampling_noise=False):
        assert steps <= self.steps, "Too much steps in inference."
        self.calculate_batch_for_diffusion(all_embed,idx)
        if noise_step == 0:
            x_t = x_start            
        else:
            t = torch.tensor([noise_step - 1] * x_start.shape[0]).to(x_start.device)
            x_t = self.q_sample(x_start,t)
        if self.ddim:
            reverse_step=int(steps/10)
        else:
            reverse_step=steps

        indices = list(range(reverse_step))[::-1]
        batch_embed = all_embed[idx,:]
        social_embed = all_social[idx,:]

        input_embed= torch.mul(torch.sigmoid(self.input_trans(torch.mul(batch_embed,social_embed))),batch_embed)

        
        for i in indices:
            t = torch.tensor([i] * x_t.shape[0]).to(x_start.device)
            # noised input, 
            out = self.p_mean_variance(x_t,t,input_embed)
            if sampling_noise:
                # Set seed for reproducible noise generation
                torch.manual_seed(self.args.seed)
                noise = torch.randn_like(x_t)
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))) )  # no noise when t == 0
                x_t = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
            else:
                x_t = out["mean"]
                
        del t, reverse_step, indices, batch_embed, social_embed, input_embed, out
        import gc
        torch.cuda.empty_cache()
        gc.collect()
        
        return x_t


    def extract_from_tensor(self,A,C):

        return A[C, torch.arange(A.shape[1]), :]


def mean_flat(tensor):
  
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def to_tensor(coo_mat,args):
    values = coo_mat.data
    indices = np.vstack((coo_mat.row, coo_mat.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo_mat.shape
    tensor_sparse=torch.sparse.FloatTensor(i, v, torch.Size(shape))
    tensor_sparse=tensor_sparse.to(args.device)
    return tensor_sparse

def mean_cross_entropy_for_ones(social_data, new_score):
    social_data_coo = social_data.tocoo()   
    row_indices = social_data_coo.row
    row_indices_torch = torch.tensor(row_indices)
    col_indices = social_data_coo.col
    ones_scores = torch.tensor(new_score[row_indices, col_indices], dtype=torch.float32)

    labels = torch.ones_like(ones_scores)
    cross_entropy = F.binary_cross_entropy_with_logits(ones_scores, labels, reduction='none') 
    # cross_entropy = F.binary_cross_entropy(ones_scores, labels, reduction='none')
    # RuntimeError: all elements of input should be between 0 and 1
    mean_cross_entropy = np.zeros(social_data.shape[0])
    
    for i in range(social_data.shape[0]):
        row_mask = row_indices_torch == i
        if row_mask.sum() > 0:
            mean_cross_entropy[i] = cross_entropy[row_mask].mean().item()
    
    return mean_cross_entropy

def flip_tensor(idx_tensor, cos_similarities, seed):
    # Set seed for reproducible random sampling
    torch.manual_seed(seed)
    
    sigmoid_pos = torch.sigmoid(cos_similarities)  # sigmoid(x)
    sigmoid_neg = 1 - sigmoid_pos  # sigmoid(-x)

    flip_prob_cos = torch.where(idx_tensor == 1, sigmoid_neg, sigmoid_pos).float()
    random_selector = torch.rand(idx_tensor.shape, device=flip_prob_cos.device)

    # Choose between keeping original or flip based on cos_similarities
    final_flip_probabilities = torch.where(
        random_selector < 0.5, torch.tensor(0.0, device=flip_prob_cos.device), flip_prob_cos)
    random_values = torch.rand(idx_tensor.shape, device=final_flip_probabilities.device)
    flipped_tensor = torch.where(random_values < final_flip_probabilities, 1 - idx_tensor, idx_tensor)

    return flipped_tensor

def cosine_similarity_batched(all_embed, batch_size=256):
    num_rows = all_embed.size(0)
    cos_sim_list = []
    
    for start in range(0, num_rows, batch_size):
        end = min(start + batch_size, num_rows)  
        A_batch = all_embed[start:end]  
        cos_sim_batch = torch.cosine_similarity(A_batch.unsqueeze(1), all_embed.unsqueeze(0), dim=2)
        cos_sim_list.append(cos_sim_batch)

    cos_sim_all = torch.cat(cos_sim_list, dim=0)
    return cos_sim_all

def refine_social(diffusion, social_data, score, all_embed, all_social, args, del_threshold,flip=True):
    diffusion.eval()  
    all_prediction = []
    if flip:
        all_cos_sim = cosine_similarity_batched(all_embed)
    
    batch_size=1024
    num_rows = len(social_data)
    num_batches = (num_rows + batch_size - 1) // batch_size 

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, num_rows)

        social_batch = social_data[start:end]
        score_batch = score[start:end]

        binary_tensor = torch.tensor(social_batch, dtype=torch.float32).to(args.device)
        social_data_tensor = torch.tensor(score_batch, dtype=torch.float32).to(args.device)

        batch = binary_tensor
        idx_tensor = torch.arange(start, end, device=args.device)  

        with torch.no_grad():
            if flip:
                cos_similarities = all_cos_sim[idx_tensor].squeeze(0)
                flipped_batch = flip_tensor(batch, cos_similarities, args.seed).to(args.device)
                prediction = diffusion.p_sample(flipped_batch, idx_tensor, all_embed, all_social, args.steps, args.steps)
            else:
                prediction = diffusion.p_sample(batch, idx_tensor, all_embed, all_social, args.steps, args.steps)

        prediction = torch.sigmoid(prediction)
        avg_prediction = args.decay * social_data_tensor + (1 - args.decay) * prediction
        all_prediction.append(avg_prediction)
        
        del social_batch, score_batch, binary_tensor, social_data_tensor, batch, cos_similarities, prediction, avg_prediction
        if flip:
            del flipped_batch
        torch.cuda.empty_cache()
        gc.collect()
    
    del all_cos_sim
    torch.cuda.empty_cache()
    gc.collect()

    new_score = torch.cat(all_prediction, dim=0)
    del all_prediction
    torch.cuda.empty_cache()
    gc.collect()
    
    flattened_score = new_score.view(-1)
    flattened_social = torch.tensor(social_data).view(-1)
    assert torch.tensor(social_data).shape == new_score.shape, "Shape mismatch between social_data and new_score"

    existing_edges = torch.where(flattened_social != 0)[0].to(flattened_score.device)
    existing_scores = flattened_score[existing_edges]
    edges_to_delete = existing_edges[existing_scores < del_threshold]

    max_deletion_proportion = 0.01
    max_deletions = int(len(existing_edges) * max_deletion_proportion)
    if len(edges_to_delete) > max_deletions:
        sorted_existing_indices = torch.argsort(existing_scores[existing_scores <= del_threshold])
        edges_to_delete = edges_to_delete[sorted_existing_indices[:max_deletions]]
    remaining_edges = torch.tensor([e for e in existing_edges.tolist() if e not in edges_to_delete.tolist()])

    # Convert flat indices to (row, col) format
    num_features = new_score.size(1)
    # Edges to keep
    h_list = [e // num_features for e in remaining_edges.tolist()]
    t_list = [e % num_features for e in remaining_edges.tolist()]

    _, sorted_indices = torch.sort(new_score.view(-1), descending=True)
    top_indices = sorted_indices.cpu().numpy().tolist()
    insert_cnt=0
    for idx in top_indices:
        h = idx // num_features
        t = idx % num_features
        if (h, t) not in zip(h_list, t_list):
            h_list.append(h)
            t_list.append(t)
            insert_cnt+=1
        if insert_cnt == len(edges_to_delete):
            break
    assert insert_cnt ==len(edges_to_delete)
    assert len(h_list) == insert_cnt + len (remaining_edges)
    decay = len(existing_edges[existing_scores < del_threshold]) > max_deletions

    del existing_edges, existing_scores, flattened_score, flattened_social, edges_to_delete, sorted_indices, top_indices, remaining_edges
    torch.cuda.empty_cache()
    gc.collect()
    
    return h_list, t_list, new_score.cpu().numpy(), decay


import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from loader.data_loader import DataDiffusionCL

def worker_init_fn(worker_id):
    """Worker init function for DataLoader to ensure reproducibility in multi-processing"""
    import numpy as np
    import random
    import torch
    
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed(worker_seed)
    torch.cuda.manual_seed_all(worker_seed)

# ===================== ADD =====================
class LINESG(nn.Module):
    def __init__(self, nonzero_idx, latent_size, args, device, gumbel_temp=0.2):
        super(LINESG, self).__init__()
        # nonzero_idx: 에지가 존재하는 u-v 쌍
        idx_arr = torch.tensor(nonzero_idx, dtype=torch.long)  # shape [E,2]
        self.register_buffer('nonzero_idx', idx_arr)
        # print(f"[DBG init] nonzero_idx range: {idx_arr[:,0].min()}/{idx_arr[:,0].max()}  ,  {idx_arr[:,1].min()}/{idx_arr[:,1].max()}")
        # self.nonzero_idx = nonzero_idx
        self.latent_size = latent_size
        self.gumbel_temp = gumbel_temp
        # user-user emb을 concat하여 유지할지 삭제할지 선택하는 mlp
        self.mlp_s = nn.Linear(self.latent_size * 2, 2)
        self.cos_thr = args.cos_s; print(f'Hyperparameter (Cosine Thr. in SG): {self.cos_thr}')  # 0.5
        self.device = device
        
        # 재현성을 위한 seed 설정
        if hasattr(args, 'seed'):
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)

    # Step1
    # 에지가 없는 u-v 쌍에서 어느 것을 추가 할지 후보군 선택.
    # 기존에 존재하는 u-v 쌍은 제외, self-loop 제외
    def get_candidate_pair(self, user_emb):
        # 1) Normalize and compute cosine with PyTorch
        u = F.normalize(user_emb, dim=1)       # [n, d]
        cos = u @ u.t()                        # [n, n] FloatTensor

        # 2) Exclude self and original edges
        n_idx = self.nonzero_idx               # LongTensor [E,2]
        cos.fill_diagonal_(-1)
        # directed edge만 masking
        cos[n_idx[:,0], n_idx[:,1]] = -1

        # 3) Threshold and nonzero in torch
        cand_mask = cos > self.cos_thr         # BooleanTensor [n,n]
        candidate_pair_idx   = cand_mask.nonzero()  # LongTensor [K,2]
        candidate_pair_values= cos[cand_mask]      # FloatTensor [K]

        # print('Candidate Pair Thr. in SG:', self.cos_thr)
        # print('Candidate Pair in SG:', candidate_pair_idx.size(0))
        
        # --- 추가: 상호관계 계산 ---
        '''
        n = u.size(0)
        total_pairs = candidate_pair_idx.size(0)

        # 3.1) 인접 행렬 생성
        adj = torch.zeros((n, n), dtype=torch.bool, device=cos.device)
        adj[candidate_pair_idx[:,0], candidate_pair_idx[:,1]] = True

        # 3.2) 상호관계 마스크 (i->j AND j->i)
        mutual_mask = adj & adj.T  # [n, n]

        all_mutual = mutual_mask.nonzero()        # 방향성을 모두 포함한 (i→j),(j→i) 쌍
        a = all_mutual.size(0)                    # 방향성 쌍의 개수

        # i<j 필터로 중복 제거
        mutual_pairs = all_mutual[all_mutual[:,0] < all_mutual[:,1]]
        b = mutual_pairs.size(0)                  # (i,j) 하나만 셈

        print(f"방향성(mutual) 쌍 수: {a}, 유니크(mutual/2) 쌍 수: {b}")
        '''
        return candidate_pair_idx, candidate_pair_values
    
    # Step3
    def get_edge_weight(self, gumbel_retain, candidate_pair_value):
        candidate_pair_value = candidate_pair_value.unsqueeze(1).to(self.device)
        # 값이 0 미만인 pair는 모두 0으로 mask
        candidate_pair_value = candidate_pair_value.masked_fill(candidate_pair_value < 0, 0)  # neg to 0

        # 남겨치는 에지에는 가중치를 1로, 값이 0 미만인 에지에는 mask된 0을 곱하여 weight 계산
        orig_weights = torch.ones(len(self.nonzero_idx), 1, device=self.device)
        gumbel_retain_w = gumbel_retain * torch.cat([orig_weights, candidate_pair_value], dim=0).squeeze()
        gumbel_01 = gumbel_retain_w.clone()

        # (2, E+K) 사이즈로 만듬. 0 -> no edge, 1 -> edge
        gumbel_retain_w = torch.cat([torch.unsqueeze(gumbel_retain_w, dim=1), torch.unsqueeze(gumbel_retain_w, dim=1)], dim=0)

        return gumbel_01, gumbel_retain_w

    # Step1 + Step2 + Step3
    def forward(self, user_emb):

        # 추가할 에지 후보 인덱스 및 weight
        candidate_pair_idx, candidate_pair_value = self.get_candidate_pair(user_emb)
        
        # 원본 에지 + 후보 에지 합치기
        self.pair_idx = torch.cat([self.nonzero_idx, candidate_pair_idx], dim=0)

        # Step2: Edge Addition and Dropping
        u_idx = self.pair_idx[:,0]
        v_idx = self.pair_idx[:,1]

        u_embeddings  = F.embedding(u_idx.to(self.device), user_emb)
        v_embeddings  = F.embedding(v_idx.to(self.device), user_emb)
        concat_emb    = torch.cat([u_embeddings, v_embeddings], dim=1)

        mlp_output = self.mlp_s(concat_emb)
        gumbel_output = F.gumbel_softmax(mlp_output, tau=self.gumbel_temp, hard=True)[:, :]
        # gumbel_retain[:, 0]   -> 출력이 0인 u-v쌍을 남길 에지로 설정
        # gumbel_retain[i] == 1 -> 남기는것 True  -> 남기기
        # gumbel_retain[i] == 0 -> 남기는것 False -> 버리기
        gumbel_retain = gumbel_output[:, 0]
        # print(f'Retained Pairs in SG: {len(self.pair_idx)} -> {(gumbel_retain == 1.0).sum()}')
        # print(f'Retained O in SG: {len(self.nonzero_idx)} -> {(gumbel_retain[:len(self.nonzero_idx)] == 1.0).sum()}')  # original edges
        # print(f'Retained C in SG: {len(candidate_pair_idx)} -> {(gumbel_retain[len(self.nonzero_idx):] == 1.0).sum()}')  # candidate pairs

        # Step3: Edge Weight Assignment
        # if len(candidate_pair_idx) > 0:
        #     gumbel_retain = self.get_edge_weight(gumbel_retain, candidate_pair_value)
        # else:
        #     gumbel_retain = torch.cat([torch.unsqueeze(gumbel_retain, dim=1), torch.unsqueeze(gumbel_retain, dim=1)], dim=0)
        gumbel_01, gumbel_retain = self.get_edge_weight(gumbel_retain, candidate_pair_value)

        return gumbel_01, gumbel_retain, self.pair_idx
# ===============================================






def mean_cross_entropy_for_ones(social_csr, soft_score_np):
    """
    행별 난이도(크로스엔트로피) 계산.
    social_csr : csr_matrix  (u×u, {0,1})
    soft_score_np : 같은 shape 의 확률(logit) 행렬 (numpy)
    """
    r, c = social_csr.nonzero()
    logits = soft_score_np[r, c]
    ce_vals = F.binary_cross_entropy_with_logits(
        torch.from_numpy(logits),
        torch.ones_like(torch.from_numpy(logits)),
        reduction='none'
    ).numpy()

    out = np.zeros(social_csr.shape[0])
    for u in range(social_csr.shape[0]):
        m = (r == u)
        if m.any():
            out[u] = ce_vals[m].mean()
    return out

# ===================== ADD =====================  
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add

def normalize_laplacian(edge_index, edge_weight):
    num_nodes = maybe_num_nodes(edge_index)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    return edge_weight
# ===============================================


# ===================== ADD =====================  
def train_line_module_one_epoch(LINE_MODULE,
                                user_emb, item_emb,
                                cur_loader, optimizer,
                                neg_k=5, use_mse=False, A_target=None):
    
    LINE_MODULE.train()
    u_all = LINE_MODULE.nonzero_idx[:, 0]   # LongTensor[E] on GPU
    i_all = LINE_MODULE.nonzero_idx[:, 1]   # LongTensor[E] on GPU

    total_loss = 0
    n_batches = 0
    for batch_users, _ in cur_loader:           # batch_users : python list(int)
        batch_users = batch_users.to(user_emb.device).long()

        # (1) positive mask
        mask  = (u_all[:, None] == batch_users[None, :]).any(1)
        pos_u = u_all[mask]
        pos_i = i_all[mask]
        if len(pos_u) == 0:
            continue

        # (2) loss 계산
        if use_mse:
            g01, _, pair_idx = LINE_MODULE(user_emb)   # hard mask
            row_m = torch.isin(pair_idx[:,0], batch_users)
            pred = g01.float()[row_m]
            tgt  = A_target[pair_idx[row_m, 0], pair_idx[row_m, 1]]
            loss = F.mse_loss(pred, tgt.float())
        else:   # skip-gram
            z_u, z_i = user_emb[pos_u], item_emb[pos_i]
            pos_s = (z_u * z_i).sum(1)
            # Set seed for reproducible negative sampling
            torch.manual_seed(42)
            neg_i = torch.randint(0, item_emb.size(0),
                                  (len(pos_u), neg_k),
                                  device=user_emb.device)
            neg_s = (z_u.unsqueeze(1) * item_emb[neg_i]).sum(2)
            loss  = -F.logsigmoid(pos_s).mean() \
                    - F.logsigmoid(-neg_s).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
        
    return total_loss / n_batches
# ===============================================

# ===================== ADD =====================  
def refine_social_with_LINE(LINE_MODULE, user_emb):
    """
    LINE 모듈이 결정한 hard-mask 로 새로운 social edge 목록(h, t) 반환
    """
    LINE_MODULE.eval()
    with torch.no_grad():
        gumbel01, _, pair_idx = LINE_MODULE(user_emb)     # step1~3

    keep = gumbel01.bool()
    all_idx = pair_idx[keep]        # [M,2] LongTensor
    # split
    n_orig = LINE_MODULE.nonzero_idx.size(0)
    keep_orig = all_idx[:n_orig]
    keep_cand = all_idx[n_orig:]
    
    orig_edges = set(map(tuple, LINE_MODULE.nonzero_idx.cpu().tolist()))
    new_edges  = set(map(tuple, torch.cat([keep_orig, keep_cand], dim=0).cpu().tolist()))

    # 5) 추가/제거 계산
    U = user_emb.size(0)
    P = U * (U - 1)
    removed = orig_edges - new_edges
    added   = new_edges  - orig_edges
    removed_pct = len(removed) / P * 100
    added_pct   = len(added)   / P * 100
    print(f"[REFINE FULL] 원본 엣지: {len(orig_edges)}({round(len(orig_edges)/P*100, 2)}%), "
          f"제거된 엣지: {len(removed)}({round(removed_pct, 2)}%), "
          f"추가된 엣지: {len(added)}({round(added_pct, 2)}%)")
    
    # social-only: user–user 페어만
    # 합치고 반환
    final = torch.cat([keep_orig, keep_cand], dim=0)  # [E',2]
    h = final[:,0].cpu().numpy()
    t = final[:,1].cpu().numpy()
    return h, t

# ===============================================

# ===================== ADD:taekwon =====================  
def build_line_curriculum_loader(social_csr, epoch, ce_1s, line_batch, seed=42):
    line_dataset = DataDiffusionCL(social_csr.A, current_epoch=epoch, ce=ce_1s, seed=seed)
    return DataLoader(line_dataset, batch_size=line_batch, shuffle=False, drop_last=False, num_workers=2, worker_init_fn=worker_init_fn)
# ===============================================


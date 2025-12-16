import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from loader.data_loader import DataDiffusionCL
import gc, random

def worker_init_fn(worker_id):
    """Worker init function for DataLoader to ensure reproducibility in multi-processing"""
    import numpy as np
    import random
    import torch
    
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def check_memory_usage():
    """GPU 메모리 사용량을 확인하고 경고 출력"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        max_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        
        print(f"[Memory] Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Max: {max_memory:.2f}GB")
        
        # 메모리 사용량이 80%를 넘으면 경고
        if allocated / max_memory > 0.8:
            print(f"[Memory] WARNING: High memory usage ({allocated/max_memory*100:.1f}%)")
            return True
    return False

# ===================== ADD =====================
class LINESG(nn.Module):
    def __init__(self, nonzero_idx, latent_size, args, device, gumbel_temp=0.2):
        super(LINESG, self).__init__()
        # nonzero_idx: 에지가 존재하는 u-v 쌍
        idx_arr = torch.tensor(nonzero_idx, dtype=torch.long)  # shape [E,2]
        self.register_buffer('nonzero_idx', idx_arr)
        self.register_buffer('init_nonzero_idx', idx_arr.clone())
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
            random.seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(args.seed)
                torch.cuda.manual_seed_all(args.seed)
        
    # Step1
    # 에지가 없는 u-v 쌍에서 어느 것을 추가 할지 후보군 선택.
    # 기존에 존재하는 u-v 쌍은 제외, self-loop 제외
    def get_candidate_pair(self, user_emb):
        # 극도로 메모리 효율적인 코사인 유사도 계산
        # 1) GPU에서 normalize
        u = F.normalize(user_emb, dim=1)  # [n, d] GPU
        n_users = u.size(0)
        
        # 2) 배치 크기를 늘려서 forward 호출 횟수 감소
        batch_size = min(200, n_users)  # 배치 크기 증가
        all_candidate_pairs = []
        all_candidate_values = []
        
        # 원본 엣지 인덱스를 GPU로 이동
        n_idx = self.nonzero_idx.to(self.device)
        
        for i in range(0, n_users, batch_size):
            end_i = min(i + batch_size, n_users)
            u_batch = u[i:end_i]  # [batch_size, d] GPU
            
            # 현재 배치와 전체 사용자 간의 코사인 유사도 계산 (GPU)
            # u.t()를 직접 사용하지 않고 더 효율적으로 계산
            cos_batch = torch.mm(u_batch, u.t())  # [batch_size, n_users] GPU
            
            # 대각선 제외 (현재 배치 범위만)
            for j in range(i, end_i):
                cos_batch[j-i, j] = -1
            
            # 현재 배치에 해당하는 원본 엣지들 제외
            batch_mask = (n_idx[:, 0] >= i) & (n_idx[:, 0] < end_i)
            if batch_mask.any():
                batch_edges = n_idx[batch_mask].clone()
                batch_edges[:, 0] -= i  # 배치 내 인덱스로 조정
                cos_batch[batch_edges[:, 0], batch_edges[:, 1]] = -1
            
            # Threshold 적용하여 후보 선택
            cand_mask = cos_batch > self.cos_thr
            cand_idx = cand_mask.nonzero()
            cand_vals = cos_batch[cand_mask]
            
            # 후보 엣지 수 제한 (메모리 절약)
            if len(cand_idx) > 5000:  # 배치당 최대 5000개로 제한 (더 작게)
                # 유사도가 높은 순으로 정렬하여 상위 10000개만 선택
                sorted_indices = torch.argsort(cand_vals, descending=True)
                cand_idx = cand_idx[sorted_indices[:5000]]
                cand_vals = cand_vals[sorted_indices[:5000]]
            
            # 전역 인덱스로 변환
            cand_idx[:, 0] += i
            
            # 즉시 CPU로 이동하여 GPU 메모리 해제
            cand_idx_cpu = cand_idx.cpu()
            cand_vals_cpu = cand_vals.cpu()
            
            # GPU 텐서들 즉시 해제
            del cos_batch, cand_mask, cand_idx, cand_vals
            torch.cuda.empty_cache()
            
            # CPU에서 누적 (메모리 사용량이 훨씬 적음)
            all_candidate_pairs.append(cand_idx_cpu)
            all_candidate_values.append(cand_vals_cpu)
            
            # CPU 텐서들도 즉시 해제
            del cand_idx_cpu, cand_vals_cpu
            
            # 매 배치마다 강제 가비지 컬렉션
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        
        # 결과 합치기 (CPU에서)
        if all_candidate_pairs:
            candidate_pair_idx = torch.cat(all_candidate_pairs, dim=0).to(self.device)
            candidate_pair_values = torch.cat(all_candidate_values, dim=0).to(self.device)
            
            # 전체 후보 엣지 수도 제한 (추가 안전장치)
            if len(candidate_pair_idx) > 50000:  # 전체 최대 50000개로 제한 (더 작게)
                sorted_indices = torch.argsort(candidate_pair_values, descending=True)
                candidate_pair_idx = candidate_pair_idx[sorted_indices[:50000]]
                candidate_pair_values = candidate_pair_values[sorted_indices[:50000]]
        else:
            candidate_pair_idx = torch.empty((0, 2), dtype=torch.long, device=self.device)
            candidate_pair_values = torch.empty((0,), dtype=torch.float, device=self.device)
        
        # CPU 리스트들 정리
        del all_candidate_pairs, all_candidate_values
        torch.cuda.empty_cache()
        
        # cos 행렬은 더 이상 필요하지 않으므로 None 반환
        cos = None
        
        # 안전장치: 빈 텐서라도 반환
        if candidate_pair_idx is None:
            candidate_pair_idx = torch.empty((0, 2), dtype=torch.long, device=self.device)
        if candidate_pair_values is None:
            candidate_pair_values = torch.empty((0,), dtype=torch.float, device=self.device)
        
        # print('Candidate Pair Thr. in SG:', self.cos_thr)
        # print('Candidate Pair in SG:', candidate_pair_idx.size(0))
        
        return candidate_pair_idx, candidate_pair_values, cos
    
    # Step3
    def get_edge_weight(self, gumbel_retain, candidate_pair_value, cos):
        # cos가 None인 경우 간단한 처리
        if cos is None:
            # 원본 엣지들은 기본값 1.0 사용
            orig_len = len(self.nonzero_idx)
            orig_cos = torch.ones(orig_len, device=self.device)
            
            candidate_pair_value = candidate_pair_value.to(self.device)
            candidate_pair_value = candidate_pair_value.masked_fill(candidate_pair_value < 0, 0)
            
            base = torch.cat([orig_cos, candidate_pair_value], dim=0)
            gumbel_01 = gumbel_retain * base
            gumbel_retain_w = torch.stack([gumbel_01, gumbel_01], dim=0)
            
            return gumbel_01, gumbel_retain_w
        
        # 기존 로직 (cos가 있는 경우)
        cos = cos.to(self.device)
        orig_idx = self.nonzero_idx.to(self.device)
        orig_cos = cos[orig_idx[:, 0], orig_idx[:, 1]].unsqueeze(1)
        orig_cos = orig_cos.clamp(min=0)
        
        candidate_pair_value = candidate_pair_value.unsqueeze(1).to(self.device)
        candidate_pair_value = candidate_pair_value.masked_fill(candidate_pair_value < 0, 0)

        base = torch.cat([orig_cos, candidate_pair_value], dim=0).squeeze(1)
        gumbel_01 = gumbel_retain * base
        gumbel_retain_w = torch.stack([gumbel_01, gumbel_01], dim=0)

        return gumbel_01, gumbel_retain_w

    # Step2 + Step3 (후보 엣지가 이미 주어진 경우)
    def forward_with_pairs(self, user_emb, pair_idx):
        """
        후보 엣지가 이미 주어진 경우의 forward 함수 (get_candidate_pair 생략)
        """
        # 메모리 사용량 확인
        # check_memory_usage()
        
        self.pair_idx = pair_idx
        
        # Step2: Edge Addition and Dropping
        u_idx = self.pair_idx[:, 0].to(self.device).long()
        v_idx = self.pair_idx[:, 1].to(self.device).long()

        # 메모리 사용량이 많을 경우 배치 처리
        if len(u_idx) > 100000:  # 임계값 설정
            # 배치 크기로 나누어 처리
            batch_size = 50000
            gumbel_outputs = []
            
            for i in range(0, len(u_idx), batch_size):
                end_idx = min(i + batch_size, len(u_idx))
                u_batch = u_idx[i:end_idx]
                v_batch = v_idx[i:end_idx]
                
                u_embeddings = F.embedding(u_batch, user_emb)
                v_embeddings = F.embedding(v_batch, user_emb)
                concat_emb = torch.cat([u_embeddings, v_embeddings], dim=1)
                
                mlp_output = self.mlp_s(concat_emb)
                gumbel_output = F.gumbel_softmax(mlp_output, tau=self.gumbel_temp, hard=True)[:, :]
                gumbel_outputs.append(gumbel_output)
                
                # 메모리 정리
                del u_embeddings, v_embeddings, concat_emb, mlp_output, gumbel_output
                torch.cuda.empty_cache()
            
            gumbel_output = torch.cat(gumbel_outputs, dim=0)
            del gumbel_outputs
        else:
            u_embeddings  = F.embedding(u_idx.to(self.device), user_emb)
            v_embeddings  = F.embedding(v_idx.to(self.device), user_emb)
            concat_emb    = torch.cat([u_embeddings, v_embeddings], dim=1)
            mlp_output = self.mlp_s(concat_emb)
            gumbel_output = F.gumbel_softmax(mlp_output, tau=self.gumbel_temp, hard=True)[:, :]
        
        # gumbel_retain[:, 0]   -> 출력이 0인 u-v쌍을 남길 에지로 설정
        # gumbel_retain[i] == 1 -> 남기는것 True  -> 남기기
        # gumbel_retain[i] == 0 -> 남기는것 False -> 버리기
        gumbel_retain = gumbel_output[:, 0]
        
        # Step3: Edge Weight Assignment (간단한 버전)
        gumbel_retain_w = torch.stack([gumbel_retain, gumbel_retain], dim=0)
        
        return gumbel_retain, gumbel_retain_w, self.pair_idx

    # Step1 + Step2 + Step3
    def forward(self, user_emb):
        # 메모리 사용량 확인
        # check_memory_usage()

        # 추가할 에지 후보 인덱스 및 weight
        candidate_pair_idx, candidate_pair_value, cos = self.get_candidate_pair(user_emb)
        
        # 원본 에지 + 후보 에지 합치기
        self.pair_idx = torch.cat([self.nonzero_idx, candidate_pair_idx], dim=0)
        
        # 엣지 수가 너무 많으면 경고
        if len(self.pair_idx) > 300000:
            print(f"[WARNING] Too many edges: {len(self.pair_idx)}, this may cause OOM")
            # 상위 엣지들만 선택
            if len(candidate_pair_idx) > 0:
                # 후보 엣지 수를 더 제한
                max_candidates = 50000
                if len(candidate_pair_idx) > max_candidates:
                    sorted_indices = torch.argsort(candidate_pair_value, descending=True)
                    candidate_pair_idx = candidate_pair_idx[sorted_indices[:max_candidates]]
                    candidate_pair_value = candidate_pair_value[sorted_indices[:max_candidates]]
                    self.pair_idx = torch.cat([self.nonzero_idx, candidate_pair_idx], dim=0)

        # Step2: Edge Addition and Dropping
        u_idx = self.pair_idx[:, 0].to(self.device).long()
        v_idx = self.pair_idx[:, 1].to(self.device).long()

        # 메모리 사용량이 많을 경우 배치 처리
        if len(u_idx) > 100000:  # 임계값 설정
            # 배치 크기로 나누어 처리
            batch_size = 50000
            gumbel_outputs = []
            
            for i in range(0, len(u_idx), batch_size):
                end_idx = min(i + batch_size, len(u_idx))
                u_batch = u_idx[i:end_idx]
                v_batch = v_idx[i:end_idx]
                
                u_embeddings = F.embedding(u_batch, user_emb)
                v_embeddings = F.embedding(v_batch, user_emb)
                concat_emb = torch.cat([u_embeddings, v_embeddings], dim=1)
                
                mlp_output = self.mlp_s(concat_emb)
                gumbel_output = F.gumbel_softmax(mlp_output, tau=self.gumbel_temp, hard=True)[:, :]
                gumbel_outputs.append(gumbel_output)
                
                # 메모리 정리
                del u_embeddings, v_embeddings, concat_emb, mlp_output, gumbel_output
                torch.cuda.empty_cache()
            
            gumbel_output = torch.cat(gumbel_outputs, dim=0)
            del gumbel_outputs
        else:
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
        gumbel_01, gumbel_retain = self.get_edge_weight(gumbel_retain, candidate_pair_value, cos)

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
# torch_geometric 관련 함수들은 현재 사용하지 않으므로 주석 처리
# from torch_geometric.utils.num_nodes import maybe_num_nodes
# from torch_scatter import scatter_add

# def normalize_laplacian(edge_index, edge_weight):
#     num_nodes = maybe_num_nodes(edge_index)
#     row, col = edge_index[0], edge_index[1]
#     deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
# 
#     deg_inv_sqrt = deg.pow_(-0.5)
#     deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
#     edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
#     return edge_weight
# ===============================================


# ===================== ADD =====================  
def train_line_module_one_epoch(LINE_MODULE,
                                user_emb, item_emb,
                                cur_loader, optimizer,
                                neg_k=5, use_mse=False, A_target=None):
    
    LINE_MODULE.train()
    
    # 후보 엣지를 한 번만 생성하고 재사용 (메모리 절약)
    print("[LINE] Generating candidate pairs once...")
    try:
        candidate_pair_idx, candidate_pair_value, cos = LINE_MODULE.get_candidate_pair(user_emb)
        
        # 후보 엣지가 있는 경우에만 합치기
        if candidate_pair_idx is not None and len(candidate_pair_idx) > 0:
            pair_idx = torch.cat([LINE_MODULE.nonzero_idx, candidate_pair_idx], dim=0)
        else:
            pair_idx = LINE_MODULE.nonzero_idx
        print(f"[LINE] Generated {len(pair_idx)} total pairs")
    except Exception as e:
        print(f"[LINE] Error generating candidate pairs: {e}")
        pair_idx = LINE_MODULE.nonzero_idx
        print(f"[LINE] Using original pairs only: {len(pair_idx)} total pairs")
    
    # pair_idx를 전역 변수로 설정하여 스코프 문제 해결
    global_pair_idx = pair_idx
    
    total_loss = 0
    n_batches = 0
    for batch_users, _ in cur_loader:           # batch_users : python list(int)
        batch_users = batch_users.to(user_emb.device).long()

        # 후보 엣지는 이미 생성했으므로 forward에서 재사용
        g01, _, _ = LINE_MODULE.forward_with_pairs(user_emb, global_pair_idx)   # g01: [E+K]

        # 이번 배치의 head(u)가 포함된 쌍만 학습
        row_m = torch.isin(global_pair_idx[:, 0], batch_users)
        if not row_m.any():
            continue

        # 예측
        pred = g01.float()[row_m]

        # 타깃: "초기 원본 인접행렬(0/1)"에서 같은 (u,v) 위치 값
        u = global_pair_idx[row_m, 0].long()
        v = global_pair_idx[row_m, 1].long()
        tgt   = A_target[u, v].float()     # ← "직전 에폭" 기준 타깃
        loss  = F.mse_loss(pred, tgt)
            
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
        
        # 배치별 메모리 정리
        del g01, pred, u, v, tgt, loss
        torch.cuda.empty_cache()
        
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

    keep = gumbel01 > 0
    final = pair_idx[keep]                     # [E',2]
    
    # 엣지 수가 너무 많으면 상위 엣지들만 선택 (메모리 보호)
    max_edges = 200000  # 최대 엣지 수를 더 작게 제한 (메모리 절약)
    if len(final) > max_edges:
        # 가중치가 높은 순으로 정렬하여 상위 엣지들만 선택
        weights = gumbel01[keep].float()
        sorted_indices = torch.argsort(weights, descending=True)
        final = final[sorted_indices[:max_edges]]
        w = weights[sorted_indices[:max_edges]]
        print(f"[LINE refine] WARNING: Too many edges ({len(final)}), limited to {max_edges}")
    else:
        w = gumbel01[keep].float()
    
    h = final[:, 0].long()
    t = final[:, 1].long()
    
    # ---------- 통계 출력 ----------
    # 원래 엣지(현재 nonzero_idx)와 최종 엣지(final)를 비교
    U = user_emb.size(0)
    denom = max(U * (U - 1), 1)          # self-loop 제외 가정(분모 0 방지)

    # 디바이스 맞추기
    device = pair_idx.device
    orig_idx = LINE_MODULE.nonzero_idx.to(device)          # [E,2]

    # (u,v) → 선형키 u*N+v 로 집합 연산
    orig_keys  = (orig_idx[:, 0] * U + orig_idx[:, 1]).long()
    final_keys = (final[:,    0] * U + final[:,    1]).long()

    E_orig  = orig_keys.numel()
    E_final = final_keys.numel()

    # 집합 차집합으로 추가/삭제 개수 계산
    # (torch.isin은 브로드캐스팅 없이 GPU에서 빠르게 동작)
    removed_mask = ~torch.isin(orig_keys,  final_keys)     # 원래 있었는데 사라진 엣지
    added_mask   = ~torch.isin(final_keys, orig_keys)      # 새로 추가된 엣지

    num_removed = int(removed_mask.sum().item())
    num_added   = int(added_mask.sum().item())

    dens_orig  = E_orig  / denom
    dens_final = E_final / denom

    print(f"[LINE refine] orig:  {E_orig} edges (density {dens_orig:.4%})")
    print(f"[LINE refine] delta: +{num_added} added, -{num_removed} removed")
    print(f"[LINE refine] final: {E_final} edges (density {dens_final:.4%})")
    # --------------------------------
    
    
    return w, h, t   

# ===============================================

# ===================== ADD:taekwon =====================  
def build_line_curriculum_loader(social_csr, epoch, ce_1s, line_batch):
    line_dataset = DataDiffusionCL(social_csr.A,current_epoch=epoch,ce=ce_1s)
    return DataLoader(line_dataset,batch_size=line_batch, shuffle=False, drop_last=False)
# ===============================================
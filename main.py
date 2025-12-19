import os
# 환경 변수를 먼저 설정하여 결정적 동작 보장
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # cuBLAS 결정적 모드
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import argparse, os, time, gc, wandb, tempfile
import numpy as np, scipy.sparse as sp
import torch, torch.nn.functional as F
import torch.utils.data as torch_data
from torch.utils.data import DataLoader
import torch.optim as optim
from utils.evaluate_utils import *
from utils.log_helper import *

# MHCN 백본 · LINEIG · 데이터 로더는 원본 그대로 import
from model.ARDSR        import *
from model.MHCN         import MHCN
from model.LineSG       import LINESG
from model.LineSG       import *
from utils.evaluate_utils import evaluate, print_results
from utils.log_helper     import logging_config, create_log_id
from loader.data_loader   import SRDataLoader, DataDiffusionCL
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def init_score_memmap(social_csr, args):
    tmp_dir = getattr(args, "tmp_dir", None) or tempfile.gettempdir()
    mmap_path = os.path.join(
        tmp_dir,
        f"init_score_{os.getpid()}_{int(time.time())}.mmap",
    )
    score_mm = np.lib.format.open_memmap(
        mmap_path, mode="w+", dtype=np.float16, shape=social_csr.shape
    )
    score_mm[:] = 0.0
    indptr = social_csr.indptr
    indices = social_csr.indices
    for row in range(social_csr.shape[0]):
        row_start = indptr[row]
        row_end = indptr[row + 1]
        if row_end > row_start:
            score_mm[row, indices[row_start:row_end]] = 1.0
    score_mm.flush()
    return score_mm



def train(args,log_path):
    
    if args.wandb:
        wandb.init(
            project="ARD-SR-eval",
            name=f"ARD-SR-{args.dataset}_batch-wise",
        )
    
    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)
    device = args.device
    print("Starting time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))



    data = SRDataLoader(args, logging)
    # DataLoader에 seed 전달
    train_loader = torch_data.DataLoader(
        data, 
        batch_size=8192,  # 배치 크기를 크게 설정
        shuffle=False, 
        num_workers=4,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
        generator=torch.Generator().manual_seed(args.seed),  # 추가 seed 설정
        persistent_workers=True,  # worker 재사용으로 안정성 향상
        drop_last=False  # 마지막 배치 보존
    )
    original_social_data = sp.csr_matrix((np.ones_like(data.train_social_h_list), 
    (data.train_social_h_list, data.train_social_t_list)), dtype='float32', shape=(data.n_users, data.n_users))
    social_data = original_social_data.copy()

    ### Build SR and Diffusion Model ###
    # 모델 생성 전 seed 재설정
    set_seed(args.seed, args.gpu_id)
    
    model = MHCN(data,args).to(device)
    rec_optimizer  = optim.Adam(model.parameters(), lr=0.00005)

    if args.method == "gumbel":
        r, c = original_social_data.nonzero()
        idx_arr = torch.tensor(np.stack([r, c], axis=1), dtype=torch.long)
        LINE_MODULE    = LINESG(idx_arr, args.embed_dim, args, device, args.gumbel_temp).to(device)
        LINE_optimizer = optim.Adam(LINE_MODULE.parameters(), lr=args.lr)
        new_social, ce_prev = None, None
        A_target   = torch.FloatTensor(social_data.A).to(device)  # EMA target
        train_social_dataset = DataDiffusionCL(social_data,0)
        cur_loader = torch_data.DataLoader(
            train_social_dataset, 
            batch_size=args.line_batch,
            shuffle=False, 
            num_workers=4,
            worker_init_fn=worker_init_fn,
            pin_memory=True,
            generator=torch.Generator().manual_seed(args.seed),
            persistent_workers=True,
            drop_last=False
        )
    elif args.method == "original":
        train_social_dataset = DataDiffusionCL(social_data,0)
        diffusion_train_loader = torch_data.DataLoader(
            train_social_dataset, 
            batch_size=args.batch_size,
            shuffle=False, 
            num_workers=4,
            worker_init_fn=worker_init_fn,
            pin_memory=True,
            generator=torch.Generator().manual_seed(args.seed),
            persistent_workers=True,
            drop_last=False
        )
        new_score = init_score_memmap(social_data, args)
        diffusion = ARDSR(data,args).to(device)
        diffusion_optimizer = optim.Adam(diffusion.parameters(),lr=args.lr)
        del_threshold = 0.6
        
    # 모델 생성 후 seed 재설정하여 완벽한 재현성 보장
    set_seed(args.seed, args.gpu_id)
        
    print("Start training...")
    best_recall, best_epoch = -100, 0
    diffusion_epoch=0
    for epoch in range(1, args.epochs + 1):
        # train_loader.dataset.ng_sample()
        
        if epoch - best_epoch >= args.stopping_steps:
            print('-'*18)
            print('Exiting from training early')
            break

        print("\nTraining SR")
        model.train()
        start_time = time.time()
        total_loss = 0.0
        for batch_user, batch_pos_item, batch_neg_item in train_loader:
            batch_user = batch_user.long().to(device)
            batch_pos_item = batch_pos_item.long().to(device)
            batch_neg_item = batch_neg_item.long().to(device)
            rec_optimizer.zero_grad()
            losses = model.bpr_loss(batch_user, batch_pos_item, batch_neg_item)
            loss = losses[0]+losses[1]*args.lambda1+losses[2]*args.lambda2
            total_loss += loss.item()
            loss.backward()
            rec_optimizer.step()
            
            
        print('Backbone SR'+"Training Epoch {:03d} ".format(epoch) +'train loss {:.4f}'.format(total_loss) + " costs " + time.strftime(
                            "%H: %M: %S", time.gmtime(time.time()-start_time)))
        print('---'*18)
        valid_results = evaluate(model,args,data,test=False)
        test_results = evaluate(model,args,data,test=True)
        print_results(None, valid_results, test_results)

        if args.wandb:
            wandb.log({
                "val/epoch": epoch,
                "val/backboneloss": total_loss,
                "val/Precision@10": valid_results[0][0],
                "val/Recall@10": valid_results[1][0],
                "val/NDCG@10": valid_results[2][0],
                "val/MRR@10": valid_results[3][0],
                "val/HR@10": valid_results[4][0],
                "test/epoch": epoch,
                "test/Precision@10": test_results[0][0],
                "test/Recall@10": test_results[1][0],
                "test/NDCG@10": test_results[2][0],
                "test/MRR@10": test_results[3][0],
                "test/HR@10": test_results[4][0],
            })
        
        
        if args.method == "gumbel":
            # LINE 정제 타임인가?
            if epoch > args.pretrain_epochs:
                # (A) LINE 학습 (BPR은 그대로 진행 후 user/item embed 이용)
                model.eval()
                with torch.no_grad():
                    user_e, _ = model.infer_embedding()
                    
                # (2) LINE 학습 (edge-MSE 사용)
                for _ in range(args.line_epochs):
                    loss = train_line_module_one_epoch(LINE_MODULE, user_e, user_e,cur_loader, LINE_optimizer,neg_k=args.neg_num,use_mse=True,A_target=A_target)
                            
                    if args.wandb:
                        wandb.log({"val/LINE_loss": loss})

                print("LINE train done")
                # (3) retain mask → 새 social_data
                if (epoch-1) % args.line_period == 0:
                    with torch.no_grad():
                        weight, h, t = refine_social_with_LINE(LINE_MODULE, user_e)
                        print("refine social done")

                        # 지수이동평균
                        h = h.to(A_target.device)
                        t = t.to(A_target.device)
                        w = weight.to(A_target.device).float()
                        
                        A_target.mul_(args.alpha)                                  # A_t = α · A_{t-1}
                        A_target[h.to(device), t.to(device)] += (1.0 - args.alpha) * w.to(device)
                        
                        
                        N = A_target.size(0)
                        init_idx  = LINE_MODULE.init_nonzero_idx.to(A_target.device)   # [E0,2]
                        init_keys = init_idx[:, 0] * N + init_idx[:, 1]                # [E0]
                        cur_keys  = h * N + t                                          # [M]
                        mask_init_keep = torch.isin(cur_keys, init_keys)               # [M] bool

                        if mask_init_keep.any():
                            h_keep = h[mask_init_keep]
                            t_keep = t[mask_init_keep]
                            # 값=1로 직접 덮어쓰기(EMA 반영 후)
                            A_target.index_put_((h_keep, t_keep),
                                                torch.ones_like(h_keep, dtype=A_target.dtype))
                        
                        
                        # 1) 원본 엣지 인덱스
                        orig_idx = LINE_MODULE.nonzero_idx
                        orig_idx = orig_idx.to(h.device)
                        orig_h, orig_t = orig_idx[:, 0], orig_idx[:, 1]
                        
                        # 2) 합집합 인덱스(원본 ∪ refine)
                        h_all = torch.cat([orig_h, h], dim=0)
                        t_all = torch.cat([orig_t, t], dim=0)
                        
                        # 3) (h_all, t_all) 중복 제거
                        keys = h_all * N + t_all
                        uniq_keys = torch.unique(keys)
                        h_u = (uniq_keys // N).long()
                        t_u = (uniq_keys %  N).long()

                        # 4) EMA에서 해당 위치의 최종 가중치 읽어오기
                        h_u_dev = h_u.to(A_target.device)
                        t_u_dev = t_u.to(A_target.device)
                        w_u = A_target[h_u_dev, t_u_dev]     # EMA 반영된 실수 가중치
                        
                        social_data = sp.csr_matrix((w_u.detach().cpu().numpy(),
                                                    (h_u.detach().cpu().numpy(), t_u.detach().cpu().numpy()))
                                                    ,shape=(data.n_users, data.n_users),
                                                    dtype='float32')
                        
                        ce_prev  = mean_cross_entropy_for_ones(social_data, A_target.detach().cpu().numpy())
                        # (4-B) curriculum 로더에 새 행렬 반영
                        
                        new_idx = torch.stack([h_u_dev.long(), t_u_dev.long()], dim=1).to(LINE_MODULE.nonzero_idx.device)
                        LINE_MODULE.nonzero_idx = new_idx

                        # (4-C) SR 백본·DataLoader 갱신
                        data.train_social_h_list = h_u.detach().cpu().numpy()
                        data.train_social_t_list = t_u.detach().cpu().numpy()
                        cur_loader = build_line_curriculum_loader(social_data, epoch, ce_prev, args.line_batch)
                        new_social = np.vstack([data.train_social_h_list, data.train_social_t_list])
                        model.init_channel()
                        
        elif args.method == "original":
            if epoch >0:
                diffusion_epoch+=1
                print("Training Diffusion")
                ##Retrive user embedding from SR backbone for guidance
                with torch.no_grad():
                    all_user_embed = model.infer_embedding()[0]
                    all_embed_frozen = all_user_embed.clone() 
                    _,graph,_= data.buildSocialAdjacency()
                    social_embed = model.get_social_embed(to_tensor(graph,args),all_embed_frozen)
                    social_embed_frozen = social_embed.clone()
                    
                diffusion.train()
                diffusion_start_time = time.time()
                total_diffusion_loss = 0.0
                for  batch in diffusion_train_loader:
                    idx,rows = batch
                    idx=idx.to(device)
                    rows = rows.to(device)
                    diffusion_optimizer.zero_grad()
                    diffusion_losses = diffusion.training_losses(idx,rows,all_embed_frozen,social_embed_frozen)
                    diffusion_loss = diffusion_losses["loss"].mean()
                    total_diffusion_loss += diffusion_loss.item()
                    diffusion_loss.backward()
                    diffusion_optimizer.step()
                    torch.cuda.empty_cache()
                    gc.collect()

                print('Diffusion'+"Training Epoch{}".format(epoch)+'train loss {:.4f}'.format(total_diffusion_loss) + " costs " + time.strftime(
                                "%H: %M: %S", time.gmtime(time.time()-diffusion_start_time)))

                if (epoch-1) % 5 == 0:
                    print('refine social start')
                    refine_start_time=time.time()
                    h,t,new_score,decay,ce_buffer = refine_social(diffusion,social_data,new_score,all_embed_frozen,social_embed_frozen,args,del_threshold,flip=True)
                    if decay:
                        del_threshold = max(del_threshold*0.99,0.45)

                    social_data = sp.csr_matrix((np.ones_like(h),(h,t)), dtype='float32', shape=(data.n_users, data.n_users))
                    ce_for_1s = ce_buffer if ce_buffer is not None else mean_cross_entropy_for_ones(social_data,new_score)
                    train_social_dataset = DataDiffusionCL(social_data,epoch,ce_for_1s)
                    diffusion_train_loader = DataLoader(train_social_dataset, batch_size=args.batch_size,shuffle=False, worker_init_fn=worker_init_fn) 
                    data.train_social_h_list=h
                    data.train_social_t_list=t
                    model.init_channel()
                    print("refine social net"+"Runing Epoch {:03d} ".format(epoch)  + " costs " + time.strftime(
                                "%H: %M: %S", time.gmtime(time.time()-refine_start_time)))
                    del ce_for_1s

                    if args.wandb:
                        wandb.log({
                            "epoch": epoch,
                            "Diffusion_train_loss": total_diffusion_loss
                            })
            torch.cuda.empty_cache()
            gc.collect()
                    

        if valid_results[1][0] > best_recall: # Recall@10
            save_model(model, args.save_dir, epoch,best_epoch)
            best_recall, best_epoch = valid_results[1][0], epoch
            best_results = valid_results
            best_test_results = test_results
            print('Save model on epoch {:04d}!'.format(epoch))
            # gumbel로 만든 social 저장하기
            if args.method == "gumbel" and new_social is not None:
                np.save(args.save_dir+'new_social_gumble.npy',new_social)  
            elif args.method == "original":
                new_social=[data.train_social_h_list,data.train_social_t_list]
                np.save(args.save_dir+'new_social_list.npy',new_social)  
            
        if args.wandb:
            wandb.summary["best/epoch"] = best_epoch
            wandb.summary["best/Precision@10"] = best_test_results[0][0]
            wandb.summary["best/Recall@10"] = best_test_results[1][0]
            wandb.summary["best/NDCG@10"] = best_test_results[2][0]
            wandb.summary["best/MRR@10"] = best_test_results[3][0]
            wandb.summary["best/HR@10"] = best_test_results[4][0]

    #save diffusion model
    if args.method == "gumbel":
        model_state_file = os.path.join(args.save_dir, 'LINE_MODULE.pth')
        torch.save({'line_model_state_dict': LINE_MODULE.state_dict()}, model_state_file)
    model_state_file = os.path.join(args.save_dir, 'diffusion.pth')
    torch.save({'model_state_dict': model.state_dict()}, model_state_file)
    print('==='*18)
    print("End. Best Epoch {:03d} ".format(best_epoch))
    print_results(None, best_results, best_test_results)   
    print("End time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    ks=eval(args.topN)
    save_results(best_test_results,args.save_dir,ks)


def set_seed(seed, gpu_id=None):
    """Set seed for reproducibility"""
    import random
    import numpy as np
    import torch
    import os
    
    # Python, NumPy, PyTorch seed 설정
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 결정적 알고리즘 사용
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.set_num_threads(4)
    
    # CUDA 캐시 정리 및 지정된 GPU에만 seed 설정
    if torch.cuda.is_available() and gpu_id is not None:
        torch.cuda.empty_cache()
        # 지정된 GPU에만 seed 설정 (다른 GPU 초기화 방지)
        torch.cuda.set_device(gpu_id)
        torch.cuda.manual_seed(seed)
    
    print(f"Set seed {seed} for complete reproducibility...")


def worker_init_fn(worker_id):
    """Worker init function for DataLoader to ensure reproducibility in multi-processing"""
    import numpy as np
    import random
    import torch
    import os
    
    # 메인 프로세스의 seed를 기반으로 worker별 seed 생성
    base_seed = torch.initial_seed()
    worker_seed = (base_seed + worker_id) % 2**32
    
    # 각 worker에서 seed 설정
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    
    # PyTorch 백엔드 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # CUDA seed 설정
    if torch.cuda.is_available():
        torch.cuda.manual_seed(worker_seed)
        torch.cuda.manual_seed_all(worker_seed)
    
    print(f"Worker {worker_id} initialized with seed {worker_seed}")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='lastfm')
    parser.add_argument('--data_path', type=str, default='datasets/')
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--stopping_steps', type=int, default=100)
    parser.add_argument('--topN', type=str, default='[10,20,50]')
    parser.add_argument('--device', nargs='?', default=3,type=int)  
    parser.add_argument('--seed', type=int, default=42)  
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    
    # parameter for sr backbone
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--lambda1', nargs='?', default=0.001,type=float)
    parser.add_argument('--lambda2', nargs='?', default=0.25,type=float)
    parser.add_argument('--ssl_temp', nargs='?', default=0.01)
    parser.add_argument('--neg', nargs='?', default=1000,type=int)
    parser.add_argument('--neg_num', nargs='?', default=1) 

    # parameter for LINE
    parser.add_argument('--pretrain_epochs', type=int, default=10) # backbone의 몇 epoch이후에 정제할지
    parser.add_argument('--line_period', type=int, default=5) # 몇 epoch 마다 정제할지
    parser.add_argument('--line_epochs', type=int, default=1) # 정제 과정에서 line 모듈을 몇 번 학습시킬지, diffusion은 step이 많으니까 얘도 2이상으로 설정할 수 있게 세팅
    parser.add_argument('--cos_s', type=float, default=0.3)
    parser.add_argument('--gumbel_temp', type=float, default=0.4)
    parser.add_argument('--alpha', type=float, default=0.5) # EMA 계수
    parser.add_argument('--line_max_ep', type=int, default=5) # curriculum learning의 최대 epoch 수
    parser.add_argument('--line_init_prop', type=int, default=0.4) # curriculum learning의 초기 샘플 비율
    parser.add_argument('--line_batch', type=int, default=16) # line의 batch size
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')


    # parameter for DIFFUSION
    parser.add_argument('--time_size', type=int, default=16, help='timestep embedding size')
    parser.add_argument('--steps', type=int, default=100, help='diffusion steps')
    parser.add_argument('--hidden_units', type=int, default=128)
    parser.add_argument('--decay', type=float, default=0.6, help='moving average decay')
    parser.add_argument('--ddim', type=bool, default=True)
    parser.add_argument('--noise_scale', type=float, default=0.01, help='noise scale for noise generating')
    parser.add_argument('--noise_min', type=float, default=0.01, help='noise lower bound for noise generating')
    parser.add_argument('--noise_max', type=float, default=0.1, help='noise upper bound for noise generating')
    parser.add_argument('--threshold', type=float, default=0.6, help='threshold')
    parser.add_argument('--temp', type=float, default=0.1, help='temp')
    parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
    

    # parameter for system 
    parser.add_argument('--wandb', type=bool, default=False)
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--method', type=str, default='gumbel')

    args = parser.parse_args()
    torch.cuda.set_device(args.gpu_id)
    args.device = torch.device("cuda:{}".format(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu")

    if args.wandb and args.method == "original":
        # wandb config 덮어쓰기
        if wandb.run is not None and hasattr(wandb, "config"):
            for key, value in wandb.config.items():
                if hasattr(args, key):
                    setattr(args, key, value)
                    
        # if args.noise_max < args.noise_min:
        #     print(f"[WARNING] Invalid config: noise_max({args.noise_max}) < noise_min({args.noise_min})")
        #     wandb.init(name="INVALID_CONFIG_SKIP")
        #     quit()
        
        # diff = args.noise_max - args.noise_min
        # if args.noise_scale > diff:
        #     print(f"[WARNING] Invalid config: noise_scale({args.noise_scale}) > (noise_max - noise_min)={diff}")
        #     wandb.init(name="INVALID_CONFIG_SKIP")
        #     quit()
            
    set_seed(args.seed, args.gpu_id)

    from datetime import datetime
    now = datetime.now().strftime('%y_%m_%d_%H_%M')
    
    save_dir = 'saved_model/{}/ARD-SR-gumbel_{}/lr{}/{}'.format(
    args.dataset,args.lr,args.method,now)
    args.save_dir = save_dir

    log_path='logs/{}/ARD-SR-gumbel_{}/lr{}/{}'.format(
    args.dataset,args.lr,args.method,now)
    
    train(args,log_path)


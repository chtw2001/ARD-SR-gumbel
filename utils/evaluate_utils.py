import numpy as np
# import bottleneck as bn
import torch
import math
import pandas as pd
import os

def f1(prec, recall):
    if (prec + recall) != 0:
        return 2 * prec * recall / (prec + recall)
    else:
        return 0

def computeTopNAccuracy(GroundTruth, predictedIndices, topN):
    precision = [] 
    recall = [] 
    NDCG = [] 
    MRR = []
    HR = []
    F1 = []
    
    for index in range(len(topN)):
        sumForPrecision = 0
        sumForRecall = 0
        sumForNdcg = 0
        sumForMRR = 0
        sumForHR = 0
        for i in range(len(predictedIndices)):
            if len(GroundTruth[i]) != 0:
                mrrFlag = True
                userHit = 0
                userMRR = 0
                dcg = 0
                idcg = 0
                idcgCount = len(GroundTruth[i])
                ndcg = 0
                hit=False
                #hit = []
                for j in range(topN[index]):
                    if predictedIndices[i][j] in GroundTruth[i]:
                        # if Hit!
                        dcg += 1.0/math.log2(j + 2)
                        if mrrFlag:
                            userMRR = (1.0/(j+1.0))
                            mrrFlag = False
                        userHit += 1
                        hit=True
                
                    if idcgCount > 0:
                        idcg += 1.0/math.log2(j + 2)
                        idcgCount = idcgCount-1
                    
                            
                if(idcg != 0):
                    ndcg += (dcg/idcg)
                    
                sumForPrecision += userHit / topN[index]
                sumForRecall += userHit / len(GroundTruth[i])               
                sumForNdcg += ndcg
                sumForMRR += userMRR

                if hit:
                    sumForHR += 1  # Increment sumForHR if there was a hit
        
        precision.append(round(sumForPrecision / len(predictedIndices), 8))
        recall.append(round(sumForRecall / len(predictedIndices), 8))
        NDCG.append(round(sumForNdcg / len(predictedIndices), 8))
        MRR.append(round(sumForMRR / len(predictedIndices), 8))
        HR.append(round(sumForHR / len(predictedIndices), 8))
        F1.append(round(f1(precision[index], recall[index]), 8))
        
    return precision, recall, NDCG,MRR,HR,F1

def print_results(loss, valid_result, test_result):
    """output the evaluation results."""
    if loss is not None:
        print("[Train]: loss: {:.4f}".format(loss))
    if valid_result is not None: 
        print("[Valid]: Precision: {} Recall: {} NDCG: {} MRR: {} HR:{} F1:{}".format(
                            '-'.join([str(x) for x in valid_result[0]]), 
                            '-'.join([str(x) for x in valid_result[1]]), 
                            '-'.join([str(x) for x in valid_result[2]]), 
                            '-'.join([str(x) for x in valid_result[3]]),
                            '-'.join([str(x) for x in valid_result[4]]),
                            '-'.join([str(x) for x in valid_result[5]])))
    if test_result is not None: 
        print("[Test]: Precision: {} Recall: {} NDCG: {} MRR: {} HR:{} F1:{}".format(
                            '-'.join([str(x) for x in test_result[0]]), 
                            '-'.join([str(x) for x in test_result[1]]), 
                            '-'.join([str(x) for x in test_result[2]]), 
                            '-'.join([str(x) for x in test_result[3]]),
                            '-'.join([str(x) for x in test_result[4]]),
                            '-'.join([str(x) for x in test_result[5]])))
        
def metric_to_df(test_result,Ks):
    metric_names = ['ndcg','recall','precision','mrr','hr','f1']
    metric_col=['K']+metric_names
    ndcg=[];recall=[];precision=[];mrr=[];hr=[];f1=[]
    k=[]
    for i,k_value in enumerate(Ks):
        k.append([k_value])
        recall.append(test_result[1][i])
        ndcg.append(test_result[2][i])
        precision.append(test_result[0][i])
        mrr.append(test_result[3][i])
        hr.append(test_result[4][i])
        f1.append(test_result[5][i])
    metrics_df=pd.DataFrame([k,ndcg,recall,precision,mrr,hr,f1]).transpose()
    metrics_df.columns=metric_col
    return metrics_df

def save_results(test_result,dir,Ks,name=None):

    metric_df=metric_to_df(test_result,Ks)
    if name is None:
        metric_df.to_csv(dir + '/test_int_metrics.tsv', sep='\t', index=False)
    else:
        metric_df.to_csv(dir + name, sep='\t', index=False)

def save_model(model, model_dir, current_epoch,last_best_epoch=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_state_file = os.path.join(model_dir, 'model_epoch{}.pth'.format(current_epoch))
    torch.save({'model_state_dict': model.state_dict(), 'epoch': current_epoch}, model_state_file)

    if last_best_epoch is not None and current_epoch != last_best_epoch:
        old_model_state_file = os.path.join(model_dir, 'model_epoch{}.pth'.format(last_best_epoch))
        if os.path.exists(old_model_state_file):
           # print("exist and remove"+old_model_state_file)
            os.system('rm {}'.format(old_model_state_file))

def evaluate(model,args,data,test=False):
    u_batch_size = data.test_batch_size
    if not test:
        testDict=data.valid_user_dict
    else:
        testDict=data.test_user_dict 

    user_ids = list(testDict.keys())
    user_ids_batches = [user_ids[i: i + u_batch_size] for i in range(0, len(user_ids), u_batch_size)]
    user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]
    target_items = [ testDict[user_ids[i]] for i in range(len(user_ids))]
    topN=eval(args.topN)
    model.eval()

    predict_items = []
    with torch.no_grad():
        for batch in user_ids_batches:
            allPos = data.getUserPosItems(batch)
            batch = batch.to(args.device)
            prediction =model.getUsersRating(batch)
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            prediction[exclude_index, exclude_items] = -(1<<10)
            _, indices = torch.topk(prediction, topN[-1])
            indices = indices.cpu().numpy().tolist()
            predict_items.extend(indices) 

    test_results = computeTopNAccuracy(target_items, predict_items, topN)

    return test_results
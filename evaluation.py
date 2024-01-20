from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, precision_recall_curve,hamming_loss
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Coverage(y_pred, y_true):
    '''
    The "Coverage" rate (also called “Recall”) is to reflect the average ratio of the
    correctly predicted labels over the real labels; to measure the percentage of the
    real labels that are covered by the hits of prediction.
    '''

    num_samples, num_class = y_pred.shape
    sorce = 0
    for i in range(num_samples):
        intersection = 0
        for j in range(num_class):
            if y_pred[i,j] == y_true[i,j] == 1:
                intersection += 1
        if intersection == 0:
            continue
        sorce += intersection / sum(y_true[i])

    return sorce / num_samples

def AbsoluteFalse(y_pred, y_true):
    '''hamming loss'''
    # 将预测值转换为二进制标签（0或1）
    y_pred = np.round(y_pred)

    # 计算预测标签和真实标签的并集和交集
    union = np.logical_or(y_pred, y_true).sum(axis=1)
    intersection = np.logical_and(y_pred, y_true).sum(axis=1)

    # 计算每个样本的Hamming Loss
    hamming_loss = (union - intersection) / y_true.shape[1]

    # 返回所有样本的Hamming Loss的平均值
    return hamming_loss.mean()

def AbsoluteTrue(y_pred, y_true):
    '''
    If the predicted tag and the real tag match strictly, count+1
    '''

    num_samples,_ = y_true.shape
    count = 0
    for i in range(num_samples) :
        if list(y_pred[i]) == list(y_true[i]):
            count += 1
    return count/num_samples

def Accuracy(y_pred, y_true):
    '''
    The "Accuracy" rate is to reflect the average ratio of correctly predicted labels
    over the total labels including correctly and incorrectly predicted labels as well
    as those real labels but are missed in the prediction
    '''

    intersection = np.sum(np.logical_and(y_pred,y_true), axis=1)
    union = np.sum(np.logical_or(y_pred,y_true), axis=1)
    accuracy = np.sum(intersection / union) / y_true.shape[0]

    return accuracy

def Precision(y_pred,y_true):
    '''
    the "Precision" is to reflect the average ratio of the
    correctly predicted labels over the predicted labels; to measure the percentage
    of the predicted labels that hit the target of the real labels.
    '''

    num_samples, num_class = y_pred.shape
    sorce = 0
    for i in range(num_samples):
        intersection = 0
        for j in range(num_class):
            if y_pred[i,j] == y_true[i,j] == 1:
                intersection += 1
        if intersection == 0:
            continue
        sorce += intersection / sum(y_pred[i])
    return sorce / num_samples

from config import ArgsConfig
# args = ArgsConfig()
# graph_info = torch.ones((args.num_classes,args.num_classes))
# x = torch.diag(torch.ones(args.num_classes))
# graph_info = graph_info - x
# adj_matrix = graph_info.float().to(args.device)
# from modules import final_adj
# end_adj_matrix = final_adj(args.batch_size,adj_matrix)  # [batch_size x num_label, batch_size x num_label]
# end_adj_matrix = torch.tensor(end_adj_matrix).to(args.device)

def evaluation(data_iter, net,weight=None,loss_fn1=None,loss_fn2=None,ebv=None,is_testing=False,is_logits=False,threshold=0.5):
    all_true = []
    all_pred = []
    test_epoch_loss = []
    coverage,acc,abs_false,abs_true,precision = 0,0,0,0,0
    
    if loss_fn1 is None:
        loss_fn1 = nn.MultiLabelSoftMarginLoss(reduction='mean',weight=weight)
    
    softmax = nn.Softmax(1)
    sigmoid = nn.Sigmoid()
    for x, y in data_iter:
        x,y = x.to(device),y.to(device,dtype=torch.float)
        #outputs = net(x) 
        if is_testing:
            _,_,_,_,_,_,logits = net(x)
        else:
            _,logits = net(x) 
        
        if is_logits:
            logits = sigmoid(logits)
        # outputs = net.PredHead(x)
        if loss_fn1 is not None:
            loss = loss_fn1(logits,y)
            test_epoch_loss.append(loss.item())
        else:
            loss = 0
            test_epoch_loss.append(loss)
        # if loss_fn2 is not None and ebv is not None:
        #     soft_out = softmax(outputs)
        #     loss2 = loss_fn2(soft_out@ebv/100,y)/100
        #     loss += loss2
        # test_epoch_loss.append(loss.item())
        y_pred = (logits >= threshold).to(torch.int64)
        all_true.extend(y.cpu().detach().numpy())
        all_pred.extend(y_pred.cpu().detach().numpy())

    all_pred = np.array(all_pred)
    all_true = np.array(all_true)

    # 计算指标
    coverage = Coverage(all_pred,all_true)
    acc = Accuracy(all_pred,all_true)
    abs_true = AbsoluteTrue(all_pred,all_true)
    #abs_false = hamming_loss(all_true,all_pred) 
    abs_false = AbsoluteFalse(all_pred,all_true)
    precision = Precision(all_pred,all_true)

    metrices = dict(acc=acc, abs_true=abs_true, abs_false=abs_false, coverage=coverage, precision=precision)

    return metrices,test_epoch_loss

def final_eval(data_iter, model,model_dir,threshold=0.5,device=device):

    import os
    file_list = [] 
    for root, dirs, files in os.walk(model_dir): # 遍历文件夹下的所有子目录和文件
        for file in files: 
            file_list.append(file) 

    model_num = len(file_list)
    print("model number:",model_num)

    for i,fn in enumerate(file_list):
        model_para_dir = os.path.join(model_dir,fn)
        state = torch.load(model_para_dir)
        model.load_state_dict(state["model_state_dict"]) #,strict=False
        model.eval()

        all_pred = []
        all_true = []
        all_prob = []
        coverage,acc,abs_false,abs_true,precision = 0,0,0,0,0
        
        for x, y in data_iter:
            x,y = x.to(device),y.to(device,torch.int64)
            _,outputs = model(x)
            y_pred = (outputs >= threshold).to(torch.int64)

            all_prob.extend(outputs.cpu().detach().numpy())
            all_true.extend(y.cpu().detach().numpy())
            all_pred.extend(y_pred.cpu().detach().numpy())
    
        all_true = np.array(all_true)
        if i == 0:
            prob = np.array(all_prob)
            pred = np.array(all_pred)
        else:
            prob += prob
            pred += pred

    #final_prob = torch.tensor(prob)/model_num
    final_prob = torch.tensor(pred)
    final_pred = np.array((final_prob >= threshold*model_num).to(torch.int64))
    
    # 计算Coverage指标
    coverage = Coverage(final_pred,all_true)
    # 计算ACC指标
    acc = Accuracy(final_pred,all_true)
    # 计算AbsoluteTrue指标
    abs_true = AbsoluteTrue(final_pred,all_true)
    # 计算AbsoluteFalse指标
    abs_false = hamming_loss(all_true,final_pred) 
    # 计算Precision指标
    precision = Precision(final_pred,all_true)

    return acc,coverage,abs_true,abs_false,precision

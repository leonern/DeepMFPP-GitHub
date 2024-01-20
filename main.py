import torch,os
from data_process import SeqsData2EqlTensor,collate,data_package
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
# from model import MyModel,CNN_BiGRUModel,BertModel,MLTP,ESMMdoel,protbert,MLTP2,MLTP1
from model import ETFC,ETFC1,GnnAttModel,ETFC2,DeepMFPP
# from evaluation_etfc import evaluation
# from ESMModel.model import EsmModel 
import time
import pickle as pkl
import numpy as np
from utils import init_logger,Category_weight1,Category_weight2,Category_weight3, CW2,CW1,CosineScheduler
from LossFunction import *
from evaluation import evaluation,Accuracy
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import StepLR,MultiStepLR,CosineAnnealingWarmRestarts
from torch.optim import RMSprop,SGD,Adam,AdamW
from torch.optim import lr_scheduler
from config import ArgsConfig
from torch.utils.tensorboard import SummaryWriter

args = ArgsConfig()
args.embedding_size = 480
args.kflod = 3
args.epochs = 250
args.aa_dict = 'esm'
args.loss_fn_name = 'MLFDL'
args.weight_decay = 0
args.batch_size = 192
args.dropout = 0.62
args.exp_nums = 0.2
args.class_weight = False
args.use_ebv = False
args.training_ratio = 0
args.scale_factor = 100
args.fldl_pos_weight = 0.4
args.info = f" "  #对当前训练做的补充说明
logger = init_logger(os.path.join(args.log_dir,f'{args.model_name}.log'))
# 将当前配置打印到日志文件中
logger.info(f"{'*'*40}当前模型训练的配置参数{'*'*40}")
for key, value in args.__dict__.items():
    logger.info(f" {key} = {value}")
logger.info(f"{'*'*100}")

## 记录当前时间，并格式化当前时间戳为年-月-日 时:分:秒的形式
logger.info(f'训练开始时间:{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))}')
# 设置随机种子
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
'''## 构建数据集,随机划分train,test datsets
data,label = genData2EqlTensor(args.data_dir,args.max_len,AminoAcid_vocab=args.aa_dict)
logger.info('data shape:%s,label shape:%s',data.shape,label.shape)

dataset = Data.TensorDataset(data,label)
train_size = int(args.split_size * len(dataset))
test_size = len(dataset) - train_size
logger.info('train sample nums:%s,test sample nums:%s',train_size,test_size)
train_dataset, test_dataset = Data.random_split(dataset, [train_size, test_size])
train_sample_nums = train_size'''

# 构建训练集和测试集
train_data,train_label = SeqsData2EqlTensor(args.train_data_dir,args.max_len,AminoAcid_vocab=args.aa_dict)
logger.info('TRAIN-- data shape:%s,label shape:%s',train_data.shape,train_label.shape)
test_data,test_label = SeqsData2EqlTensor(args.test_data_dir,args.max_len,AminoAcid_vocab=args.aa_dict)
logger.info('TEST-- data shape:%s,label shape:%s',test_data.shape,test_label.shape)

train_sample_nums = train_data.shape[0]
train_dataset = Data.TensorDataset(train_data,train_label)
test_dataset = Data.TensorDataset(test_data,test_label)

train_iter = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,drop_last=True,pin_memory=True)
test_iter = Data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,drop_last=False,pin_memory=True)

# 数据增强样本
train_data_da,_ = SeqsData2EqlTensor(args.train_data_da_dir,args.max_len,AminoAcid_vocab=args.aa_dict)
logger.info('TRAIN-- DA data shape:%s,label shape:%s',train_data.shape,train_label.shape)
train_dataset_CL = Data.TensorDataset(train_data,train_data_da,train_label)
# train_iter_cont = Data.DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True,collate_fn=collate)
train_iter_cont = Data.DataLoader(train_dataset_CL, batch_size=args.batch_size,shuffle=True,drop_last=True,pin_memory=True)

# 计算学习率调度器的最大更新次数
CoslrS_max_update = train_sample_nums / args.batch_size*args.epochs
logger.info(f"CosineScheduler's max_update:{CoslrS_max_update}")
if CoslrS_max_update <= 12000:
    CoslrS_max_update = 12000
else:
    CoslrS_max_update = CoslrS_max_update*1.1
logger.info(f"CosineScheduler's max_update:{CoslrS_max_update}")

if args.training_ratio > 1:
    logger.info('NOET: DA sample and contrastive learning not used')
else:
    logger.info('NOET: Start using DA sample and contrastive learning')

d = None
if args.use_ebv:
    d = pkl.load(open(args.ebv_dir, 'rb')).data#.detach().cpu()
    d = F.normalize(d).to(args.device)
    logger.info("Note:this training use EBV.")

weights = None
if args.class_weight:
    # 计算类别权重
    weights = CW2(train_label).to(args.device)
    logger.info(f'class weight:{weights}')
else:
    logger.info("Note:this training don't use class weight.")

## 训练
## 初始化k折交叉验证存放评价指标的字典
keys = ['epoch', 'acc', 'abs_true', 'abs_false', 'coverage','precision']
Kflod_best_metrics = {key: 0 for key in keys}
for Kflod_num in range(args.kflod):
    
    model = DeepMFPP(vocab_size=21,embedding_size=args.embedding_size, encoder_layer_num=1, fan_layer_num=1, num_heads=8,output_size=args.num_classes,
                  esm_path=args.ems_path,layer_idx=args.esm_layer_idx,dropout=args.dropout,Contrastive_Learning=args.ctl).to(args.device)
    scaler = GradScaler()

    if args.continue_training:
        logger.info('Note! This training is append training.')
        logger.info(f'model paramater load path:{args.checkpoint_path}')

        state = torch.load(args.checkpoint_path)
        model.load_state_dict(state["model_state_dict"]) #
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
        if "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])
            logger.info('Loaded optimizer state for last training saved checkpoint')

        if "epoch" in state and 'best_metrics' in state:
            last_epoch = state["epoch"]
            scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma,last_epoch=last_epoch)
            logger.info(f"last training info: epoch:{last_epoch}, metrics:{state['best_metrics']}")
    else:
        optimizer = Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay) #
        scheduler = CosineScheduler(max_update=CoslrS_max_update, base_lr=args.lr, warmup_steps=500)
        # 定义学习率调度器
        #scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=10,T_mult=2)
        #scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
        #scheduler = MultiStepLR(optimizer,milestones=args.lr_milestones,gamma=args.lr_gamma) #0.0015 0.0012 0.00096 0.000768 0.0006144 0.00049152
    
    # 主损失
    # criterion_model = nn.BCELoss(reduction='mean')
    # criterion_model = nn.CrossEntropyLoss(reduction='mean',weight=weights) #,weight=weights
    # criterion_model = Poly_cross_entropy(weights)
    # criterion_model = Poly_BCE(weights,logits=True)
    # criterion_model = nn.MultiLabelSoftMarginLoss(reduction='mean',weight=weights)
    criterion_model = FocalDiceLoss(clip_pos=args.fldl_clip_pos, clip_neg=args.fldl_clip_neg, pos_weight=args.fldl_pos_weight)
    # criterion_model = FocalLoss(alpha=args.fl_alpha,gamma=args.fl_gamma,logits=True,reduction='sum')
    # criterion_model = AsymmetricLoss()
    # criterion_model = BinaryDiceLoss()

    # 对比损失
    criterion = ContrastiveLoss(args.margin)
    # criterion = NT_Xent(args.batch_size,temperature=0.1,world_size=1).to(args.device)
    # criterion = InfoNCELoss(0.1)
    
    # 创建SummaryWriter对象
    writer = SummaryWriter(log_dir=f"{args.tensorboard_log_dir}/{args.model_name}")
    '''
    tensorboard --logdir=log_dir --host=127.0.0.1
    http://127.0.0.1:6006/
    '''
    
    sigmoid = nn.Sigmoid()
    ETA,ETL,EVA,EVL = [],[],[],[]
    global_step = 0
    Version_num = Kflod_num+1
    best_metrics = {key: 0 for key in keys}
    logger.info(f"Start training K-flod:{Kflod_num+1}")
    logger.info(f"{'-'*150}")
    logger.info(f"{'-'*150}")
    for epoch in range(args.epochs):
        train_epoch_loss = []
        train_epoch_Ctloss = []
        train_epoch_acc = []
        total = len(train_dataset)
        ce_lab_count = 0
        t0 = time.time()
        model.train()

        if epoch >= args.epochs*args.training_ratio:
            training_stage = 1
            for seq,seq_da,label in train_iter_cont:  #
                seq = seq.to(args.device)
                seq_da = seq_da.to(args.device)
                label = label.to(args.device,dtype=torch.float) #
                pair_label = torch.zeros([args.batch_size,1]).to(args.device)
                
                # with autocast():
                hidden1,logits1 = model(seq) #
                hidden2,logits2 = model(seq_da) #
                # logits1,logits2 = sigmoid(logits1),sigmoid(logits2) ## 

                # 对比损失
                ctl = criterion(hidden1,hidden2,pair_label) / args.scale_factor
                # ctl = criterion(hidden1,hidden2) / (args.scale_factor)
                # ctl += ctl1

                # 主损失
                l1 = criterion_model(logits1,label) 
                l2 = criterion_model(logits2,label) #/ args.scale_factor
                # if args.use_ebv:
                #     soft_out1 = softmax(out1)
                #     soft_out2 = softmax(out2)
                #     loss_ebv1 = criterion_model(soft_out1@d.t()/100,label)/100
                #     loss_ebv2 = criterion_model(soft_out2@d.t()/100,label)/100
                #     cel1 += loss_ebv1
                #     cel2 += loss_ebv2
                loss = l1 + l2 
                
                optimizer.zero_grad() 
                loss.backward()
                optimizer.step()
                
                # scaler.scale(loss).backward()
                # scaler.step(optimizer)
                # scaler.update()
                
                train_epoch_loss.append(loss.item())
                train_epoch_Ctloss.append((ctl).item())
                writer.add_scalar('Loss', loss, global_step)

                if scheduler:
                    if scheduler.__module__ == lr_scheduler.__name__:
                        # Using PyTorch In-Built scheduler
                        scheduler.step()
                    else:
                        # Using custom defined scheduler
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = scheduler(global_step)

                global_step += 1
                ce_lab_count += torch.sum(label)
        else:
            training_stage = 0
            for seq,label in train_iter:
                seq,label = seq.to(args.device),label.to(args.device,dtype=torch.float)
            
                #with autocast():
                _,out = model(seq) # _,out
                # 主损失
                loss = criterion_model(out,label)
                if args.use_ebv:
                    softmax = nn.Softmax(1)
                    soft_out = softmax(out)
                    loss_ebv = criterion_model(soft_out@d.t()/100,label)/100 # /100 /100
                    loss += loss_ebv
                
                optimizer.zero_grad() 
                loss.backward()
                optimizer.step()
                
                train_epoch_loss.append(loss.item())
                writer.add_scalar('Loss', loss, global_step)
                
                # 更新学习率
                if scheduler:
                    if scheduler.__module__ == lr_scheduler.__name__:
                        scheduler.step()
                    else:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = scheduler(global_step)

                global_step += 1
            train_epoch_Ctloss = torch.mean(torch.tensor(train_epoch_Ctloss))
        # t1 = time.time()
        # train_acc = np.mean(train_epoch_acc)
        # 更新学习率
        # scheduler.step()
        if training_stage==1:
            #print('ce_lab_count',ce_lab_count)
            train_epoch_Ctloss = np.mean(train_epoch_Ctloss)
        
        # train_acc = np.mean(train_epoch_acc)

        model.eval() 
        with torch.no_grad(): 
            metrices,valloss = evaluation(test_iter,model,loss_fn1=criterion_model,loss_fn2=criterion_model,weight=weights) # ,is_logits=False
            train_metrices,_ = evaluation(train_iter,model,loss_fn1=criterion_model,loss_fn2=criterion_model,weight=weights)
            eval_metrics = {'epoch':epoch+1} 
            eval_metrics.update(metrices)

        # 除去abs_false,其余4个评价指标中有3个更好，就更新权重和best_metrices,因为有必定增大的epoch,所以判定条件是>=3+1
        better_count = sum([eval_metrics[k] > best_metrics[k] for k in eval_metrics.keys()])
        if better_count >= 4:
            best_metrics = eval_metrics.copy()
            state = {   
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_metrics":best_metrics
                    }
            torch.save(state,f'{args.save_para_dir}/{args.exp_nums}_{Version_num}.pth',pickle_protocol=4) # ,pickle_protocol=4

        results = f"EPOCH:[{best_metrics['epoch']}|{epoch+1}/{args.epochs}] loss:{np.mean(train_epoch_loss):.3f},Ctloss:{train_epoch_Ctloss:.3f},"
        results += f"valloss:{np.mean(valloss):.3f},tra_acc:{train_metrices['acc']:.3f}"
        for key,value in metrices.items():
            results += f" {key}:{value:.4f}"
        results += f" T:{time.time()-t0:.2f}" 
        logger.info(results)
        ETA.append(train_metrices['acc']),ETL.append(np.mean(train_epoch_loss))
        EVA.append(metrices['acc']),EVL.append(np.mean(valloss))
        
    writer.close()
    KF_better_count = sum([best_metrics[k] > Kflod_best_metrics[k] for k in best_metrics.keys()])
    if KF_better_count >= 3:
        K = Kflod_num+1
        Kflod_best_metrics = best_metrics.copy()
        BETA,BETL,BEVA,BEVL = ETA,ETL,EVA,EVL

# from visualization.plot import plot
# plotsavepath = f'./visualization/pictures/DeepMFPP/loss_acc/{args.model_name}-{args.exp_nums}-{args.loss_fn_name} \
# _{K}_{args.kflod}Flod_{args.epochs}_epochs_loss_acc曲线.png'      
# plot(args.epochs,BETL,BETA,BEVL,BEVA,plotsavepath,title='Loss and Accuracy Curve',loss_scale=False)

logger.info(f"{'*'*45}{args.kflod}-flod Best Performance{'*'*45}")
logger.info('第{best_K}Flod,Epoch:{epoch},Acc:{acc:.4f},Abs_True:{abs_true:.4f},Abs_False:{abs_false:.4f},Coverage:{coverage:.4f},Precision:{precision:.4f}' \
.format(best_K=K,**Kflod_best_metrics))
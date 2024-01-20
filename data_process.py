import numpy as np
import os
import torch
import torch.nn.utils.rnn as rnn_utils

def get_seqdata(file_path:str,file_type:str='txt'):
    
    if file_type=='txt':
        with open(file_path, 'r') as inf:
            lines = inf.read().splitlines()
        assert len(lines) % 2 == 0, "Invalid file format. Number of lines should be even."

        seqs=[]
        labels=[]
        for line in lines:
            if line[0] == '>':
                labels.append([int(i) for i in line[1:]])
            else:
                seqs.append(line)
    
    return seqs,labels

def SeqsData2EqlTensor(file_path:str,max_len:int,AminoAcid_vocab=None):
    '''
    Args:
        flie:文件路径 \n
        max_len:设定转换后的氨基酸序列最大长度 \n
        vocab_dict:esm or protbert ,默认为按顺序映射的词典
    '''
    # 只保留20种氨基酸和填充数,其余几种非常规氨基酸均用填充数代替
    # 使用 esm和portbert字典时，nn.embedding()的vocab_size = 25
    if AminoAcid_vocab =='esm':
        aa_dict = {'[PAD]': 1, 'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10, 'T': 11, 'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16, 
                   'N': 17, 'F': 18, 'Y': 19, 'M': 20, 'H': 21, 'W': 22, 'C': 23, 'X': 1, 'B': 1, 'U': 1, 'Z': 1, 'O': 1}
    elif AminoAcid_vocab == 'protbert':
        aa_dict = {'[PAD]':0,'L': 5, 'A': 6, 'G': 7, 'V': 8, 'E': 9, 'S': 10, 'I': 11, 'K': 12, 'R': 13, 'D': 14, 'T': 15, 
               'P': 16, 'N': 17, 'Q': 18, 'F': 19, 'Y': 20, 'M': 21, 'H': 22, 'C': 23, 'W': 24, 'X': 0, 'U': 0, 'B': 0, 'Z': 0, 'O': 0}
    else:
        aa_dict = {'[PAD]':0,'A':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,'M':11,'N':12,'P':13,'Q':14,'R':15,
               'S':16,'T':17,'V':18,'W':19,'Y':20,'U':0,'X':0,'J':0}
    ## Esm vocab
    ## protbert vocab
    
    padding_key = '[PAD]'
    default_padding_value = 0
    if padding_key in aa_dict:
        dict_padding_value = aa_dict.get('[PAD]')
    else:
        dict_padding_value = default_padding_value
        print(f"No padding value in the implicit dictionary, set to {default_padding_value} by default")

    with open(file_path, 'r') as inf:
        lines = inf.read().splitlines()
    assert len(lines) % 2 == 0, "Invalid file format. Number of lines should be even."
    
    long_pep_counter=0
    pep_codes=[]
    labels=[]
    for line in lines:
        if line[0] == '>':
            labels.append([int(i) for i in line[1:]])
        else:
            x = len(line)
        
            if  x < max_len:
                current_pep=[]
                for aa in line:
                    if aa.upper() in aa_dict.keys():
                        current_pep.append(aa_dict[aa.upper()])
                pep_codes.append(torch.tensor(current_pep)) #torch.tensor(current_pep)
            else:
                pep_head = line[0:int(max_len/2)]
                pep_tail = line[int(x-int(max_len/2)):int(x)]
                new_pep = pep_head+pep_tail
                current_pep=[]
                for aa in new_pep:
                    current_pep.append(aa_dict[aa])
                pep_codes.append(torch.tensor(current_pep))
                long_pep_counter += 1

    print("length > {}:{}".format(max_len,long_pep_counter))
    data = rnn_utils.pad_sequence(pep_codes,batch_first=True,padding_value=dict_padding_value)
    return data,torch.tensor(labels)

def index_alignment(batch,condition_num=0,subtraction_num1=4,subtraction_num2=1):
    '''将其他蛋白质语言模型的字典索引和默认字典索引进行对齐，保持氨基酸索引只有20个数构成，且范围在[1,20]，[PAD]=0或者1 \n
    "esm"模型，condition_num=1,subtraction_num1=3，subtraction_num2=1； \n
    "protbert"模型，condition_num=0,subtraction_num1=4

    Args:               
        batch:形状为[batch_size,seq_len]的二维张量 \n
        condition_num:字典中的[PAD]值 \n 
        subtraction_num1:对齐非[PAD]元素所需减掉的差值 \n
        subtraction_num2:对齐[PAD]元素所需减掉的差值
    
    return:
        shape:[batch_size,seq_len],dtype=tensor.
    '''
    condition = batch == condition_num
    # 创建一个张量，形状和batch相同，表示非[PAD]元素要减去的值
    subtraction = torch.full_like(batch, subtraction_num1)
    if condition_num==0:
        # 使用torch.where()函数来选择batch中为0的元素或者batch减去subtraction中的元素
        output = torch.where(condition, batch, batch - subtraction)
    elif condition_num==1:
        # 创建一个张量，形状和batch相同，表示[PAD]元素要减去的值
        subtraction_2 = torch.full_like(batch, subtraction_num2)
        output = torch.where(condition, batch-subtraction_2, batch - subtraction)
    
    return output

def data_package(dataset,train_idx,val_idx):
    '''
    将传入的dataset(Subset对象)按照train_idx和val_idx索引进行样本和标签的打包

    Args:
        dataset:Subste
        train_idx,val_idx:list
        return:TensorDataset
    '''

    import torch.utils.data as Data
    train_set_x,val_set_x = dataset.dataset.tensors[0][train_idx], dataset.dataset.tensors[0][val_idx]
    train_set_y,val_set_y = dataset.dataset.tensors[1][train_idx], dataset.dataset.tensors[1][val_idx]
    
    return Data.TensorDataset(train_set_x,train_set_y),Data.TensorDataset(val_set_x,val_set_y)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def collate(batch):
    seq1_ls=[]
    seq2_ls=[]
    label1_ls=[]
    label2_ls=[]
    label_ls=[]
    batch_size=len(batch)
    for i in range(int(batch_size/2)):
        seq1,lab1 = batch[i][0],batch[i][1]
        seq2,lab2 = batch[i+int(batch_size/2)][0],batch[i+int(batch_size/2)][1]
        label1_ls.append(lab1.unsqueeze(0))
        label2_ls.append(lab1.unsqueeze(0))
        if torch.equal(lab1,lab2):
            label = 0
        else:
            label = 1
        seq1_ls.append(seq1.unsqueeze(0))
        seq2_ls.append(seq2.unsqueeze(0))
        label_ls.append(label)
    seq1=torch.cat(seq1_ls).to(device)
    seq2=torch.cat(seq2_ls).to(device)
    label=torch.tensor(label_ls).to(device)
    label1=torch.cat(label1_ls).to(device)
    label2=torch.cat(label2_ls).to(device)

    return seq1,seq2,label,label1,label2

def Numseq2OneHot(numseq):
    '''数字化映射后的序列转为热编码形式

    Args:
        numseq:数字化后的氨基酸序列,dtype=tensor
    '''
    OneHot = []
    for seq in numseq:
        len_seq = len(seq)
        seq = seq.cpu().numpy()
        x = torch.zeros(len_seq,20)
        for i in range(len_seq):
            x[i][seq[i]-1] = 1
        OneHot.append(np.array(x))
    
    return torch.tensor(np.array(OneHot))
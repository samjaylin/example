# example
```bash
pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2
pip install transformers==4.5.1
pip install tqdm boto3 requests regex
```
```bash
import torch
from transformers import BertTokenizer
from IPython.display import clear_output

PRETRAINED_MODEL_NAME = "bert-base-chinese"  # 指定繁簡中文 BERT-BASE 預訓練模型

# 取得此預訓練模型所使用的 tokenizer
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

clear_output()
print("PyTorch 版本：", torch.__version__)
```
```bash
vocab = tokenizer.vocab
print("字典大小：", len(vocab))
```
```bash
import random
random_tokens = random.sample(list(vocab), 10)
random_ids = [vocab[t] for t in random_tokens]

print("{0:20}{1:15}".format("token", "index"))
print("-" * 25)
for t, id in zip(random_tokens, random_ids):
    print("{0:15}{1:10}".format(t, id))
```
```bash
indices = list(range(647, 657))
some_pairs = [(t, idx) for t, idx in vocab.items() if idx in indices]
for pair in some_pairs:
    print(pair)
```
```bash
text = "[CLS] 等到潮水 [MASK] 了，就知道誰沒穿褲子。"
tokens = tokenizer.tokenize(text)
ids = tokenizer.convert_tokens_to_ids(tokens)

print(text)
print(tokens[:10], '...')
print(ids[:10], '...')
```

```bash
"""
這段程式碼載入已經訓練好的 masked 語言模型並對有 [MASK] 的句子做預測
"""
from transformers import BertForMaskedLM

# 除了 tokens 以外我們還需要辨別句子的 segment ids
tokens_tensor = torch.tensor([ids])  # (1, seq_len)
segments_tensors = torch.zeros_like(tokens_tensor)  # (1, seq_len)
maskedLM_model = BertForMaskedLM.from_pretrained(PRETRAINED_MODEL_NAME)
clear_output()

# 使用 masked LM 估計 [MASK] 位置所代表的實際 token 
maskedLM_model.eval()
with torch.no_grad():
    outputs = maskedLM_model(tokens_tensor, segments_tensors)
    predictions = outputs[0]
    # (1, seq_len, num_hidden_units)
del maskedLM_model

# 將 [MASK] 位置的機率分佈取 top k 最有可能的 tokens 出來
masked_index = 5
k = 3
probs, indices = torch.topk(torch.softmax(predictions[0, masked_index], -1), k)
predicted_tokens = tokenizer.convert_ids_to_tokens(indices.tolist())

# 顯示 top k 可能的字。一般我們就是取 top 1 當作預測值
print("輸入 tokens ：", tokens[:10], '...')
print('-' * 50)
for i, (t, p) in enumerate(zip(predicted_tokens, probs), 1):
    tokens[masked_index] = t
    print("Top {} ({:2}%)：{}".format(i, int(p.item() * 100), tokens[:10]), '...')
```

```bash
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import time
import os
import re
from itertools import chain
from transformers import BertTokenizer, BertModel
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
PRETRAINED_MODEL_NAME = "bert-base-uncased" #英文pretrain(不區分大小寫)
print(torch.__version__)

```

```bash
import torch

# If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
```

```bash
import pickle
with open("data/train_data.pkl", "rb") as f:
    train_data = pickle.load(f)
with open("data/test_data.pkl", "rb") as f:
    test_data = pickle.load(f)
    
import numpy as np
percentle = np.percentile(np.array(train_data.review.apply(lambda x:len(x.split(" ")))), np.arange(10, 100, 10)) # deciles
print("TrainData Len:",len(train_data))
print("The Percentile is:",percentle)
percentle = np.percentile(np.array(test_data.review.apply(lambda x:len(x.split(" ")))), np.arange(10, 100, 10)) # deciles
print("TestData Len:",len(test_data))
print("The Percentile is:",percentle)

trainList = []
testList = []
for idx in range(len(train_data)):
    trainList.append([train_data["review"][idx],train_data["sentiment"][idx]])
for idx in range(len(test_data)):
    testList.append([test_data["review"][idx],])
```
```bash
from torch.utils.data import Dataset,random_split

#create Dataset
class MyDataset(Dataset):
    def __init__(self, mode, tokenizer):
        assert mode in ["train", "test"]  
        self.mode = mode
        if self.mode == "train":
            self.df = trainList
        if self.mode == "test":
            self.df = testList
        self.len = len(self.df)
        self.maxlen = 250 #限制文章長度(若你記憶體夠多也可以不限)
        self.tokenizer = tokenizer  # we will use BERT tokenizer
    
    # 定義回傳一筆訓練 / 測試數據的函式
    def __getitem__(self, idx):
        origin_text = self.df[idx][0]
        if self.mode == "test":
            text_a = self.df[idx][0]
            text_b = None  #for natural language inference
            #label_tensor = None #in our case, we have label
            label_tensor = None
        else:     
            text_a = self.df[idx][0]
            text_b = None  #for natural language inference
            label_id = self.df[idx][1]
            label_tensor = torch.tensor(label_id)
            
        
        # 建立第一個句子的 BERT tokens
        word_pieces = ["[CLS]"]
        tokens_a = self.tokenizer.tokenize(text_a)
        word_pieces += tokens_a[:self.maxlen] + ["[SEP]"]
        len_a = len(word_pieces)
        
        if text_b is not None:
            tokens_b = self.tokenizer.tokenize(text_b)
            word_pieces += tokens_b + ["[SEP]"]
            len_b = len(word_pieces) - len_a
               
        # 將整個 token 序列轉換成索引序列
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)
        
        # 將第一句包含 [SEP] 的 token 位置設為 0，其他為 1 表示第二句
        if text_b is None:
            segments_tensor = torch.tensor([1] * len_a,dtype=torch.long)
        elif text_b is not None:
            segments_tensor = torch.tensor([0] * len_a + [1] * len_b,dtype=torch.long)
        
        return (tokens_tensor, segments_tensor, label_tensor, origin_text)
    
    def __len__(self):
        return self.len
    
# initialize Dataset
trainset = MyDataset("train", tokenizer=tokenizer)
testset = MyDataset("test", tokenizer=tokenizer)



#split val from trainset
val_size = int(trainset.__len__()*0.2) 
trainset, valset = random_split(trainset,[trainset.__len__()-val_size,val_size])
print('trainset size:' ,trainset.__len__())
print('valset size:',valset.__len__())
print('testset size: ',testset.__len__())
```
```bash
from torch.utils.data import TensorDataset, DataLoader,RandomSampler, SequentialSampler
from torch.nn.utils.rnn import pad_sequence

""""
create_mini_batch(samples)吃上面定義的mydataset
回傳訓練 BERT 時會需要的 4 個 tensors：
- tokens_tensors  : (batch_size, max_seq_len_in_batch)
- segments_tensors: (batch_size, max_seq_len_in_batch)
- masks_tensors   : (batch_size, max_seq_len_in_batch)
- label_ids       : (batch_size)
"""
#collate_fn: 如何將多個樣本的資料連成一個batch丟進 model
#截長補短後要限制attention只注意非pad 的部分
def create_mini_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]
    
    # 訓練集有 labels
    if samples[0][2] is not None:
        label_ids = torch.stack([s[2] for s in samples])
    else:
        label_ids = None
    
    # zero pad到該batch下最長的長度
    tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)
    segments_tensors = pad_sequence(segments_tensors,batch_first=True)
    
    # attention masks，將 tokens_tensors 裡頭不為 zero padding
    # 的位置設為 1 讓 BERT 只關注這些位置的 tokens
    masks_tensors = torch.zeros(tokens_tensors.shape,dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)
    
    return tokens_tensors, segments_tensors, masks_tensors, label_ids



# 初始化一個每次回傳 batch size 個訓練樣本的 DataLoader
# 利用 'collate_fn' 將 list of samples 合併成一個 mini-batch
BATCH_SIZE = 4
train_data = RandomSampler(trainset)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE,collate_fn=create_mini_batch,shuffle=True)
valloader = DataLoader(valset, batch_size=BATCH_SIZE,collate_fn=create_mini_batch,shuffle=False)
testloader = DataLoader(testset, batch_size=BATCH_SIZE,collate_fn=create_mini_batch,shuffle=False)

data = next(iter(trainloader))
tokens_tensors, segments_tensors, masks_tensors, label_ids = data
print(tokens_tensors)
print(segments_tensors)
print(masks_tensors)
print(label_ids)

```
```bash
from transformers import BertForSequenceClassification
PRETRAINED_MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 2

model = BertForSequenceClassification.from_pretrained(
    PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)


print("""
name      module
--------------------""")

for name, module in model.named_children():
    if name == "bert":
        for n, _ in module.named_children():
            print("{:10}{}".format(name,n) )
    else:
        print("{:10} {}".format(name, module))

```
```bash
%%time
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from pytorch_transformers import AdamW, WarmupLinearSchedule

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print("device:",device)
model = model.to(device)


EPOCHS = 4
WEIGHT_DECAY = 0.01
LR = 2e-5
WARMUP_STEPS =int(0.2*len(trainloader))
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)],
     'weight_decay': WEIGHT_DECAY},
    {'params': [p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)],
     'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=LR, eps=1e-8)
scheduler = WarmupLinearSchedule(optimizer, warmup_steps=WARMUP_STEPS,
                                 t_total=len(trainloader)*EPOCHS)
best_accuracy = 0.9
best_loss = 10
for epoch in range(EPOCHS):
    correct = 0
    #total = 0
    train_loss , val_loss = 0.0 , 0.0
    train_acc, val_acc = 0, 0
    n, m = 0, 0
    model.train()
    tk = tqdm(trainloader, total=len(trainloader), position=0, leave=True)
    for step, (tokens_tensors, segments_tensors,masks_tensors, labels) in enumerate(tk):
        n += 1
        tokens_tensors, segments_tensors,masks_tensors, labels = tokens_tensors.to(device), segments_tensors.to(device),masks_tensors.to(device), labels.to(device)

        # 將參數梯度歸零
        optimizer.zero_grad()
        
        # forward pass
        outputs = model(input_ids=tokens_tensors, 
                        token_type_ids=segments_tensors, 
                        attention_mask=masks_tensors, 
                        labels=labels)
        # outputs 的順序是 "(loss), logits, (hidden_states), (attentions)"
        
        loss = outputs[0]
        loss.backward()
        scheduler.step()
        optimizer.step()
        
        
        #get prediction and calulate acc
        logits = outputs[1]
        _, pred = torch.max(logits.data, 1)
        train_acc += accuracy_score(pred.cpu().tolist() , labels.cpu().tolist())

        # 紀錄當前 batch loss
        train_loss += loss.item()
    
    #validation
    with torch.no_grad():
        model.eval()
        tk = tqdm(valloader, total=len(valloader), position=0, leave=True)
        for step, (tokens_tensors, segments_tensors,masks_tensors, labels) in enumerate(tk):
            m += 1
            tokens_tensors, segments_tensors,masks_tensors, labels =  tokens_tensors.to(device), segments_tensors.to(device),masks_tensors.to(device), labels.to(device)
            val_outputs = model(input_ids=tokens_tensors, 
                        token_type_ids=segments_tensors, 
                        attention_mask=masks_tensors, 
                        labels=labels)
            
            logits = val_outputs[1]
            _, pred = torch.max(logits.data, 1)
            val_acc += accuracy_score(pred.cpu().tolist() , labels.cpu().tolist())
            val_loss += val_outputs[0].item()
#         if val_loss < best_loss:
#             torch.save(model, "epoch" + str(epoch+1) +'model.pkl')
#             best_loss = val_loss
        if val_acc > best_accuracy:
            torch.save(model, "best_model.pkl')
            best_accuracy = val_acc
    print('[epoch %d] loss: %.4f, acc: %.4f, val loss: %4f, val acc: %4f' %
          (epoch+1, train_loss/n, train_acc/n, val_loss/m,  val_acc/m  ))

print('Done')


```
```bash
torch.cuda.empty_cache()
model = torch.load('best_model.pkl')
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print("device:",device)
model = model.to(device)
```

```bash
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
true=[]
predictions=[]

with torch.no_grad():
    model.eval()
    tk = tqdm(valloader, total=len(valloader), position=0, leave=True)
    for step, (tokens_tensors, segments_tensors,masks_tensors, labels) in enumerate(tk):
        m += 1
        tokens_tensors, segments_tensors,masks_tensors, labels =  tokens_tensors.to(device), segments_tensors.to(device),masks_tensors.to(device), labels.to(device)
        val_outputs = model(input_ids=tokens_tensors, 
                    token_type_ids=segments_tensors, 
                    attention_mask=masks_tensors, 
                    labels=labels)

        logits = val_outputs[1]
        _, pred = torch.max(logits.data, 1)
        predictions.extend(pred.cpu().tolist())
        true.extend(labels.cpu().tolist())
report = classification_report(np.array(true),predictions)
matrix = confusion_matrix(np.array(true),predictions)
print("Classification Report: \n", report)
print("Confusion Matrix: \n",matrix)  

```


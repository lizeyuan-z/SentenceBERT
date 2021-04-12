import torch
from torch import nn
from torch.optim import Adam, SGD
import numpy as np
from Model import SentenceBERT


torch.manual_seed(41)
"""
torch.cuda.current_device()
torch.cuda._initialized = True
"""
np.random.seed(41)


def train(config, is_cross, device, max_length, batch_size, learning_rate, train_path, trained_model):

    # 数据预处理
    train_set_sentence1 = []
    train_set_sentence2 = []
    train_label = []
    with open(train_path[0], 'r', encoding='utf-8') as f:
        for line in f:
            tmp = line.strip().split('\t')
            try:
                train_set_sentence1.append(tmp[0].strip())
                train_set_sentence2.append(tmp[1].strip())
                train_label.append(int(tmp[2].strip()))
            except:
                pass
    label = ['entailment', 'contradiction', 'neutral']
    with open(train_path[1], 'r', encoding='utf-8') as f:
        for line in f:
            tmp = line.strip().split('\t')
            try:
                train_set_sentence1.append(tmp[0].strip())
                train_set_sentence2.append(tmp[1].strip())
                train_label.append(int(tmp[2].strip()))
            except:
                pass
    index = [i for i in range(len(train_label))]
    np.random.shuffle(index)
    train_set_sentence1_shuffled = []
    train_set_sentence2_shuffled = []
    train_label_shuffled = []
    for i in index:
        train_set_sentence1_shuffled.append(train_set_sentence1[i])
        train_set_sentence2_shuffled.append(train_set_sentence2[i])
        train_label_shuffled.append(train_label[i])
    print('total:', len(train_label))

    # 模型初始化
    model = SentenceBERT(config, is_cross, device, max_length)
    loss_fn = nn.CrossEntropyLoss()
    model.to(device)
    loss_fn.to(device)
    # optimizer = Adam(model.parameters(), lr=learning_rate, eps=1e-8)
    optimizer = Adam(params=[{'params': model.bert.parameters(), 'lr': learning_rate},
                            {'params': model.dnn.parameters(), 'lr': learning_rate},
                            ], lr=learning_rate)
    warm_up_iter = len(train_label) // 10
    WarmUp = lambda cur_iter: cur_iter / warm_up_iter if cur_iter < warm_up_iter else 1
    NoWarmUp = lambda cur_iter: 1
    WarmUpLR = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[WarmUp, WarmUp])
    optimizer.zero_grad()

    # 模型训练
    for i in range(0, len(train_label_shuffled), batch_size):
        if i + batch_size <= len(train_label_shuffled):
            output = model(train_set_sentence1_shuffled[i: i+batch_size], train_set_sentence2_shuffled[i: i+batch_size], train_label_shuffled[i:i + batch_size]).to(device)
            loss = loss_fn(output, torch.tensor(train_label_shuffled[i:i + batch_size], dtype=torch.long).to(device)).to(device)
        else:
            output = model(train_set_sentence1_shuffled[i:], train_set_sentence2_shuffled[i:], train_label_shuffled[i:]).to(device)
            loss = loss_fn(output, torch.tensor(train_label_shuffled[i:], dtype=torch.long).to(device)).to(device)
        loss.backward()
        optimizer.step()
        WarmUpLR.step()
        optimizer.zero_grad()
        print('>>>batch:{0}     loss:{1}'.format(i // batch_size, float(loss)))
    torch.save(model.state_dict(), 'trained_model' + '/{0}_model'.format(trained_model))

    """
    for i in range(0, len(train_label_shuffled)):
        train_batch1 = [train_set_sentence1_shuffled[i], train_set_sentence2_shuffled[i]]
        train_batch2 = [train_set_sentence2_shuffled[i], train_set_sentence1_shuffled[i]]
        output = model(train_batch1, train_batch2, train_label_shuffled[i:i + batch_size]).to(device)
        loss = loss_fn(output, torch.tensor([train_label_shuffled[i], train_label_shuffled[i]], dtype=torch.long).to(device)).to(device)
        loss.backward()
        optimizer.step()
        WarmUpLR.step()
        optimizer.zero_grad()
        print('>>>batch:{0}     loss:{1}'.format(i // batch_size, float(loss)))
    """
    torch.save(model.state_dict(), 'trained_model' + '/{0}_model'.format(trained_model))


if __name__ == "__main__":
    config = 'pretrain_model'
    is_cross = False
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    max_length = 30
    batch_size = 16
    learning_rate = 2e-5
    train_path = ['data/SNLI/train.txt', 'data/MultiNLI/train.txt']
    trained_model = 'SentenceBERT_NLI'
    train(config, is_cross, device, max_length, batch_size, learning_rate, train_path, trained_model)

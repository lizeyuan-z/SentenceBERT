import torch
import numpy as np
import pandas as pd
from Model import SentenceBERT, SentenceBERTCosine, SentenceBERTQQP


def accuracy(predict_label, label):
    acc = 0
    for i, j in zip(predict_label, label):
        if i == j:
            acc += 1
    return acc / len(label)


def AUC(score_label, labels):
    sum_rank = 0
    score_label.sort()
    for i in range(len(score_label)):
        if score_label[i][1] == 1:
            sum_rank += i+1
    M = labels.count(1)
    N = labels.count(0)
    auc = (sum_rank - M * (1 + M) / 2) / (M * N)
    return auc


def Spearman(predicted_score, score):
    X1 = pd.Series(predicted_score)
    X2 = pd.Series(score)
    return X1.corr(X2, method='spearman')


def evaluate(config, is_cross, device, max_length, batch_size, dev_path, dev_index, trained_model):
    model = SentenceBERT(config, is_cross, device, max_length)
    model.to(device)
    param = torch.load(trained_model, map_location=device)
    # model.load_state_dict({k:v for k,v in param.items() if 'dnn' not in k})
    model.load_state_dict(param)
    model.eval()
    for k in range(len(dev_path)):
        dev_set_sentence1 = []
        dev_set_sentence2 = []
        dev_label = []
        label = ['entailment', 'contradiction', 'neutral', 0, 1, 2]
        with open(dev_path[k], 'r', encoding='utf-8') as f:
            for line in f:
                tmp = line.strip().split('\t')
                try:
                    dev_set_sentence1.append(tmp[dev_index[k][0]].strip())
                    dev_set_sentence2.append(tmp[dev_index[k][1]].strip())
                    dev_label.append(int(tmp[dev_index[k][2]].strip()))
                except:
                    pass
        predict_score = []
        predict_label = []
        print('{0}'.format(dev_path[k]))
        for i in range(0, len(dev_label), batch_size):
            if i + batch_size <= len(dev_label):
                output = model(dev_set_sentence1[i: i + batch_size], dev_set_sentence2[i: i + batch_size],
                               dev_label[i: i + batch_size]).to(device)
            else:
                output = model(dev_set_sentence1[i:], dev_set_sentence2[i:], dev_label[i:]).to(
                    device)

            predict_score += output[:, 1].tolist()
            score_label = list(zip(predict_score, dev_label))
            predict_label += output.max(1)[1].tolist()
            """
            predict_label += output.tolist()
            """
            print(i)
        print("preidct:{0}      dev:{1}".format(len(predict_label), len(dev_label)))
        with open('tmp_{0}.txt'.format(k), 'w', encoding='utf-8') as f:
            for i in range(len(predict_label)):
                if predict_label[i] != dev_label[i]:
                    f.write('{0}\t{1}\t{2}\n'.format(dev_label[i], dev_set_sentence1[i], dev_set_sentence2[i]))

        print('dev accuracy:', accuracy(predict_label, dev_label))
        """
        print('dev Spearman:', Spearman(predict_label, dev_label))
        """
        print('dev AUC:', AUC(score_label, dev_label))



if __name__ == '__main__':
    config = 'pretrain_model'
    trained_model = 'trained_model/SentenceBERTAlign_NLI_model'
    is_cross = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_length = 30
    batch_size = 16

    dev_path = ['data/SNLI/test.txt', 'data/MultiNLI/test.txt']
    dev_index = [(0, 1, 2), (0, 1, 2)]
    """
    dev_path = ['data/TianChi/gaiic_track3_round1_train_20210220.tsv']
    dev_index = [(0, 1, 2)]
    """
    evaluate(config, is_cross, device, max_length, batch_size, dev_path, dev_index, trained_model)

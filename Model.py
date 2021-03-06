import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer, BertModel, BertPreTrainedModel


class SentenceBERT(nn.Module):
    def __init__(self, config, is_cross, device, max_length):
        super(SentenceBERT, self).__init__()

        self.is_cross = is_cross
        self.max_length = max_length
        self.device = device
        self.dim = 768

        self.config = config
        self.config = AutoConfig.from_pretrained(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config)
        self.bert = AutoModel.from_pretrained(config).to(self.device)
        self.dnn = nn.Sequential(
            nn.Tanh(),
            nn.Linear(self.dim * 3, 3),
        ).to(device)

    def forward(self, sentence1, sentence2, labels):
        input1 = self.tokenizer(sentence1, padding='max_length', truncation=True, max_length=self.max_length)
        input1_ids = torch.tensor(input1['input_ids'], dtype=torch.long).to(self.device)
        input1_att = torch.tensor(input1['attention_mask'], dtype=torch.long).to(self.device)
        input2 = self.tokenizer(sentence2, padding='max_length', truncation=True, max_length=self.max_length)
        input2_ids = torch.tensor(input2['input_ids'], dtype=torch.long).to(self.device)
        input2_att = torch.tensor(input2['attention_mask'], dtype=torch.long).to(self.device)
        labels_ = torch.tensor(labels, dtype=torch.long).to(self.device)
        """
            n-gram+CNN
        """
        tensor1 = self.bert(input1_ids, attention_mask=input1_att)[0]
        tensor2 = self.bert(input2_ids, attention_mask=input2_att)[0]
        out1 = (tensor1 * input1_att.unsqueeze(-1).expand(tensor1.size())).sum(1) / input1_att.sum(1).unsqueeze(1)
        out2 = (tensor2 * input2_att.unsqueeze(-1).expand(tensor2.size())).sum(1) / input2_att.sum(1).unsqueeze(1)
        difference = torch.abs(out1 - out2).to(self.device)
        return self.dnn(torch.cat((out1, out2, difference), 1)).to(self.device)


class SentenceBERTCosine(nn.Module):
    def __init__(self, config, is_cross, device, max_length):
        super(SentenceBERTCosine, self).__init__()

        self.is_cross = is_cross
        self.max_length = max_length
        self.device = device
        self.dim = 768

        self.config = config
        self.config = AutoConfig.from_pretrained(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config)
        self.bert = AutoModel.from_pretrained(config).to(self.device)

    def forward(self, sentence1, sentence2, labels):
        input1 = self.tokenizer(sentence1, padding='max_length', truncation=True, max_length=self.max_length)
        input1_ids = torch.tensor(input1['input_ids'], dtype=torch.long).to(self.device)
        input1_att = torch.tensor(input1['attention_mask'], dtype=torch.long).to(self.device)
        input2 = self.tokenizer(sentence2, padding='max_length', truncation=True, max_length=self.max_length)
        input2_ids = torch.tensor(input2['input_ids'], dtype=torch.long).to(self.device)
        input2_att = torch.tensor(input2['attention_mask'], dtype=torch.long).to(self.device)
        out1 = self.bert(input1_ids, attention_mask=input1_att)[0].sum(1)/len(input1_ids[0])
        out2 = self.bert(input2_ids, attention_mask=input2_att)[0].sum(1)/len(input2_ids[0])
        return (out1 * out2).sum(1) / torch.sqrt(out1 * out1 + out2 * out2).sum(1)


class SentenceBERTQQP(nn.Module):
    def __init__(self, config, is_cross, device, max_length):
        super(SentenceBERTQQP, self).__init__()

        self.is_cross = is_cross
        self.max_length = max_length
        self.device = device
        self.dim = 768

        self.config = config
        self.config = AutoConfig.from_pretrained(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config)
        self.bert = AutoModel.from_pretrained(config).to(self.device)
        self.dnn = nn.Sequential(
            nn.Tanh(),
            nn.Linear(self.dim * 3, 2),
        ).to(device)

    def forward(self, sentence1, sentence2, labels):
        input1 = self.tokenizer(sentence1, padding='max_length', truncation=True, max_length=self.max_length)
        input1_ids = torch.tensor(input1['input_ids'], dtype=torch.long).to(self.device)
        input1_att = torch.tensor(input1['attention_mask'], dtype=torch.long).to(self.device)
        input2 = self.tokenizer(sentence2, padding='max_length', truncation=True, max_length=self.max_length)
        input2_ids = torch.tensor(input2['input_ids'], dtype=torch.long).to(self.device)
        input2_att = torch.tensor(input2['attention_mask'], dtype=torch.long).to(self.device)
        labels_ = torch.tensor(labels, dtype=torch.long).to(self.device)
        """
            n-gram+CNN
        """
        tensor1 = self.bert(input1_ids, attention_mask=input1_att)[0]
        tensor2 = self.bert(input2_ids, attention_mask=input2_att)[0]
        out1 = (tensor1.permute(0, 2, 1) @ input1_att) / input1_att.sum(1)
        out2 = (tensor2.permute(0, 2, 1) @ input2_att) / input2_att.sum(1)
        difference = torch.abs(out1 - out2).to(self.device)
        return self.dnn(torch.cat((out1, out2, difference), 1)).to(self.device)


class SentenceBERTAlign(nn.Module):
    def __init__(self, config, is_cross, device, max_length):
        super(SentenceBERTAlign, self).__init__()

        self.is_cross = is_cross
        self.max_length = max_length
        self.device = device
        self.dim = 768

        self.config = config
        self.config = AutoConfig.from_pretrained(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config)
        self.bert = AutoModel.from_pretrained(config).to(self.device)
        self.fusion = nn.Linear(self.dim * 2, self.dim * 2)
        self.dnn = nn.Sequential(
            nn.Tanh(),
            nn.Linear(self.dim * 6, 3),
        ).to(device)

    def forward(self, sentence1, sentence2, labels):
        input1 = self.tokenizer(sentence1, padding='max_length', truncation=True, max_length=self.max_length)
        input1_ids = torch.tensor(input1['input_ids'], dtype=torch.long).to(self.device)
        input1_att = torch.tensor(input1['attention_mask'], dtype=torch.long).to(self.device)
        input2 = self.tokenizer(sentence2, padding='max_length', truncation=True, max_length=self.max_length)
        input2_ids = torch.tensor(input2['input_ids'], dtype=torch.long).to(self.device)
        input2_att = torch.tensor(input2['attention_mask'], dtype=torch.long).to(self.device)
        labels_ = torch.tensor(labels, dtype=torch.long).to(self.device)
        rep_out1 = self.bert(input1_ids, attention_mask=input1_att)[0].to(self.device)
        rep_out2 = self.bert(input2_ids, attention_mask=input2_att)[0].to(self.device)
        att_matrix = rep_out1 @ rep_out2.permute(0, 2, 1).to(self.device)
        att_out1 = att_matrix.softmax(1) @ rep_out2
        att_out2 = att_matrix.permute(0, 2, 1).softmax(1) @ rep_out1
        """
            fusion???ensemble
        """
        tensor1 = torch.cat((rep_out1, att_out1), 2)
        tensor2 = torch.cat((rep_out2, att_out2), 2)
        out1 = self.fusion((tensor1 * input1_att.unsqueeze(-1).expand(tensor1.size()))).sum(1) / input1_att.sum(1).unsqueeze(1)
        out2 = self.fusion((tensor2 * input2_att.unsqueeze(-1).expand(tensor2.size()))).sum(1) / input2_att.sum(1).unsqueeze(1)
        difference = torch.abs(out1 - out2).to(self.device)
        return self.dnn(torch.cat((out1, out2, difference), 1)).to(self.device)


class SentenceBERTAlignBiCls(nn.Module):
    def __init__(self, config, is_cross, device, max_length):
        super(SentenceBERTAlignBiCls, self).__init__()

        self.is_cross = is_cross
        self.max_length = max_length
        self.device = device
        self.dim = 768

        self.config = config
        self.config = AutoConfig.from_pretrained(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config)
        self.bert = AutoModel.from_pretrained(config).to(self.device)
        self.fusion = nn.Linear(self.dim * 2, self.dim * 2)
        self.dnn = nn.Sequential(
            nn.Tanh(),
            nn.Linear(self.dim * 6, 2),
        ).to(device)

    def forward(self, sentence1, sentence2, labels):
        input1 = self.tokenizer(sentence1, padding='max_length', truncation=True, max_length=self.max_length)
        input1_ids = torch.tensor(input1['input_ids'], dtype=torch.long).to(self.device)
        input1_att = torch.tensor(input1['attention_mask'], dtype=torch.long).to(self.device)
        input2 = self.tokenizer(sentence2, padding='max_length', truncation=True, max_length=self.max_length)
        input2_ids = torch.tensor(input2['input_ids'], dtype=torch.long).to(self.device)
        input2_att = torch.tensor(input2['attention_mask'], dtype=torch.long).to(self.device)
        labels_ = torch.tensor(labels, dtype=torch.long).to(self.device)
        rep_out1 = self.bert(input1_ids, attention_mask=input1_att)[0].to(self.device)
        rep_out2 = self.bert(input2_ids, attention_mask=input2_att)[0].to(self.device)
        att_matrix = rep_out1 @ rep_out2.permute(0, 2, 1).to(self.device)
        att_out1 = att_matrix.softmax(1) @ rep_out2
        att_out2 = att_matrix.permute(0, 2, 1).softmax(1) @ rep_out1
        """
            fusion???ensemble
        """
        tensor1 = torch.cat((rep_out1, att_out1), 2)
        tensor2 = torch.cat((rep_out2, att_out2), 2)
        out1 = self.fusion((tensor1 * input1_att.unsqueeze(-1).expand(tensor1.size()))).sum(1) / input1_att.sum(1).unsqueeze(1)
        out2 = self.fusion((tensor2 * input2_att.unsqueeze(-1).expand(tensor2.size()))).sum(1) / input2_att.sum(1).unsqueeze(1)
        difference = torch.abs(out1 - out2).to(self.device)
        return self.dnn(torch.cat((out1, out2, difference), 1)).to(self.device)


class SentenceBERTAlignRepDnn(nn.Module):
    def __init__(self, config, is_cross, device, max_length):
        super(SentenceBERTAlignRepDnn, self).__init__()

        self.is_cross = is_cross
        self.max_length = max_length
        self.device = device
        self.dim = 768

        self.config = config
        self.config = AutoConfig.from_pretrained(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config)
        self.bert = AutoModel.from_pretrained(config).to(self.device)
        self.fusion = nn.Linear(self.dim * 2, self.dim * 2)
        self.repdnn = nn.Sequential(
            nn.Tanh(),
            nn.Linear(self.dim * 2, self.dim * 2),
            nn.Dropout(0.2),
            nn.Linear(self.dim * 2, self.dim * 2),
            nn.Tanh()
        )
        self.dnn = nn.Sequential(
            nn.Tanh(),
            nn.Linear(self.dim * 6, 3),
        ).to(device)

    def forward(self, sentence1, sentence2, labels):
        input1 = self.tokenizer(sentence1, padding='max_length', truncation=True, max_length=self.max_length)
        input1_ids = torch.tensor(input1['input_ids'], dtype=torch.long).to(self.device)
        input1_att = torch.tensor(input1['attention_mask'], dtype=torch.long).to(self.device)
        input2 = self.tokenizer(sentence2, padding='max_length', truncation=True, max_length=self.max_length)
        input2_ids = torch.tensor(input2['input_ids'], dtype=torch.long).to(self.device)
        input2_att = torch.tensor(input2['attention_mask'], dtype=torch.long).to(self.device)
        labels_ = torch.tensor(labels, dtype=torch.long).to(self.device)
        rep_out1 = self.bert(input1_ids, attention_mask=input1_att)[0].to(self.device)
        rep_out2 = self.bert(input2_ids, attention_mask=input2_att)[0].to(self.device)
        att_matrix = rep_out1 @ rep_out2.permute(0, 2, 1).to(self.device)
        att_out1 = att_matrix.softmax(1) @ rep_out2
        att_out2 = att_matrix.permute(0, 2, 1).softmax(1) @ rep_out1
        """
            fusion???ensemble
        """
        tensor1 = self.repdnn(torch.cat((rep_out1, att_out1), 2))
        tensor2 = self.repdnn(torch.cat((rep_out2, att_out2), 2))
        out1 = self.fusion((tensor1 * input1_att.unsqueeze(-1).expand(tensor1.size()))).sum(1) / input1_att.sum(1).unsqueeze(1)
        out2 = self.fusion((tensor2 * input2_att.unsqueeze(-1).expand(tensor2.size()))).sum(1) / input2_att.sum(1).unsqueeze(1)
        difference = torch.abs(out1 - out2).to(self.device)
        return self.dnn(torch.cat((out1, out2, difference), 1)).to(self.device)


class SentenceBERTAlignAutoEncoder(nn.Module):
    def __init__(self, config, is_cross, device, max_length):
        super(SentenceBERTAlignAutoEncoder, self).__init__()

        self.is_cross = is_cross
        self.max_length = max_length
        self.device = device
        self.dim = 768

        self.config = config
        self.config = AutoConfig.from_pretrained(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config)
        self.bert = AutoModel.from_pretrained(config).to(self.device)
        self.fusion = nn.Linear(self.dim * 2, self.dim * 2)
        self.autoencoder = nn.Sequential(
            nn.Tanh(),
            nn.Linear(self.dim * 2, self.dim),
            nn.Dropout(0.2),
            nn.Linear(self.dim, self.dim * 2),
            nn.Tanh()
        )
        self.dnn = nn.Sequential(
            nn.Tanh(),
            nn.Linear(self.dim * 6, 3),
        ).to(device)

    def forward(self, sentence1, sentence2, labels):
        input1 = self.tokenizer(sentence1, padding='max_length', truncation=True, max_length=self.max_length)
        input1_ids = torch.tensor(input1['input_ids'], dtype=torch.long).to(self.device)
        input1_att = torch.tensor(input1['attention_mask'], dtype=torch.long).to(self.device)
        input2 = self.tokenizer(sentence2, padding='max_length', truncation=True, max_length=self.max_length)
        input2_ids = torch.tensor(input2['input_ids'], dtype=torch.long).to(self.device)
        input2_att = torch.tensor(input2['attention_mask'], dtype=torch.long).to(self.device)
        labels_ = torch.tensor(labels, dtype=torch.long).to(self.device)
        rep_out1 = self.bert(input1_ids, attention_mask=input1_att)[0].to(self.device)
        rep_out2 = self.bert(input2_ids, attention_mask=input2_att)[0].to(self.device)
        att_matrix = rep_out1 @ rep_out2.permute(0, 2, 1).to(self.device)
        att_out1 = att_matrix.softmax(1) @ rep_out2
        att_out2 = att_matrix.permute(0, 2, 1).softmax(1) @ rep_out1
        """
            fusion???ensemble
        """
        tensor1 = self.autoencoder(torch.cat((rep_out1, att_out1), 2))
        tensor2 = self.autoencoder(torch.cat((rep_out2, att_out2), 2))
        out1 = self.fusion((tensor1 * input1_att.unsqueeze(-1).expand(tensor1.size()))).sum(1) / input1_att.sum(1).unsqueeze(1)
        out2 = self.fusion((tensor2 * input2_att.unsqueeze(-1).expand(tensor2.size()))).sum(1) / input2_att.sum(1).unsqueeze(1)
        difference = torch.abs(out1 - out2).to(self.device)
        return self.dnn(torch.cat((out1, out2, difference), 1)).to(self.device)


class SentenceBERTAlignFFN(nn.Module):
    def __init__(self, config, is_cross, device, max_length):
        super(SentenceBERTAlignFFN, self).__init__()

        self.is_cross = is_cross
        self.max_length = max_length
        self.device = device
        self.dim = 768

        self.config = config
        self.config = AutoConfig.from_pretrained(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config)
        self.bert = AutoModel.from_pretrained(config).to(self.device)
        self.fusion = nn.Linear(self.dim * 2, self.dim * 2)
        self.ffn = nn.Sequential(
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(self.dim * 2, self.dim),
            nn.Tanh()
        )
        self.dnn = nn.Sequential(
            nn.Tanh(),
            nn.Linear(self.dim * 6, 3),
        ).to(device)

    def forward(self, sentence1, sentence2, labels):
        input1 = self.tokenizer(sentence1, padding='max_length', truncation=True, max_length=self.max_length)
        input1_ids = torch.tensor(input1['input_ids'], dtype=torch.long).to(self.device)
        input1_att = torch.tensor(input1['attention_mask'], dtype=torch.long).to(self.device)
        input2 = self.tokenizer(sentence2, padding='max_length', truncation=True, max_length=self.max_length)
        input2_ids = torch.tensor(input2['input_ids'], dtype=torch.long).to(self.device)
        input2_att = torch.tensor(input2['attention_mask'], dtype=torch.long).to(self.device)
        labels_ = torch.tensor(labels, dtype=torch.long).to(self.device)
        rep_out1 = self.bert(input1_ids, attention_mask=input1_att)[0].to(self.device)
        rep_out2 = self.bert(input2_ids, attention_mask=input2_att)[0].to(self.device)
        att_matrix = rep_out1 @ rep_out2.permute(0, 2, 1).to(self.device)
        att_out1 = att_matrix.softmax(1) @ rep_out2
        att_out2 = att_matrix.permute(0, 2, 1).softmax(1) @ rep_out1
        """
            fusion???ensemble
        """
        tensor1 = self.autoencoder(torch.cat((rep_out1, att_out1), 2))
        tensor2 = self.autoencoder(torch.cat((rep_out2, att_out2), 2))
        out1 = self.fusion((tensor1 * input1_att.unsqueeze(-1).expand(tensor1.size()))).sum(1) / input1_att.sum(1).unsqueeze(1)
        out2 = self.fusion((tensor2 * input2_att.unsqueeze(-1).expand(tensor2.size()))).sum(1) / input2_att.sum(1).unsqueeze(1)
        difference = torch.abs(out1 - out2).to(self.device)
        return self.dnn(torch.cat((out1, out2, difference), 1)).to(self.device)


class SentenceBERTMy(nn.Module):
    def __init__(self, config, is_cross, device, max_length, featVector_dim, featAttruct_num):
        super(SentenceBERTMy, self).__init__()

        self.is_cross = is_cross
        self.max_length = max_length
        self.device = device
        self.dim = 768
        self.featVector_dim = featVector_dim
        self.featAttruct_num = featAttruct_num

        self.config = config
        self.config = AutoConfig.from_pretrained(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config)
        self.bert = AutoModel.from_pretrained(config).to(self.device)
        self.fusion = [nn.Linear(self.featVector_dim * 2, self.featVector_dim * 2) for i in range(self.featAttruct_num)]
        self.featPara = [nn.Linear(self.dim, self.featVector_dim) for i in range(self.featAttruct_num)]
        self.score = [nn.Linear(self.featVector_dim * 6, 1) for i in range(self.featAttruct_num)]
        self.dnn = nn.Sequential(
            nn.Tanh(),
            nn.Linear(self.featVector_dim * 6, 3),
        ).to(device)

    def forward(self, sentence1, sentence2, labels):
        input1 = self.tokenizer(sentence1, padding='max_length', truncation=True, max_length=self.max_length)
        input1_ids = torch.tensor(input1['input_ids'], dtype=torch.long).to(self.device)
        input1_att = torch.tensor(input1['attention_mask'], dtype=torch.long).to(self.device)
        input2 = self.tokenizer(sentence2, padding='max_length', truncation=True, max_length=self.max_length)
        input2_ids = torch.tensor(input2['input_ids'], dtype=torch.long).to(self.device)
        input2_att = torch.tensor(input2['attention_mask'], dtype=torch.long).to(self.device)
        labels_ = torch.tensor(labels, dtype=torch.long).to(self.device)

        rep_out1 = [self.featPara[i](self.bert(input1_ids, attention_mask=input1_att)[0].to(self.device)) for i in range(self.featAttruct_num)]
        rep_out2 = [self.featPara[i](self.bert(input2_ids, attention_mask=input2_att)[0].to(self.device)) for i in range(self.featAttruct_num)]

        L2 = torch.tensor([0], dtype=torch.float32)
        for i in range(self.featAttruct_num):
            for j in range(i+1, self.featAttruct_num):
                L2 += torch.norm(rep_out1[i] - rep_out1[i], p=2)
                L2 += torch.norm(rep_out2[i] - rep_out2[j], p=2)

        att_matrix = [rep_out1[i] @ rep_out2[i].permute(0, 2, 1).to(self.device) for i in range(self.featAttruct_num)]
        att_out1 = [att_matrix[i].softmax(1) @ rep_out2[i] for i in range(self.featAttruct_num)]
        att_out2 = [att_matrix[i].permute(0, 2, 1).softmax(1) @ rep_out1[i] for i in range(self.featAttruct_num)]
        tensor1 = [torch.cat((rep_out1[i], att_out1[i]), dim=2) for i in range(self.featAttruct_num)]
        tensor2 = [torch.cat((rep_out2[i], att_out2[i]), dim=2) for i in range(self.featAttruct_num)]
        vector1 = [self.fusion[i]((tensor1[i] * input1_att.unsqueeze(-1).expand(tensor1[i].size()))).sum(1) / input1_att.sum(1).unsqueeze(1) for i in range(self.featAttruct_num)]
        vector2 = [self.fusion[i]((tensor2[i] * input2_att.unsqueeze(-1).expand(tensor2[i].size()))).sum(1) / input2_att.sum(1).unsqueeze(1) for i in range(self.featAttruct_num)]
        difference = [torch.abs(vector1[i] - vector2[i]).to(self.device) for i in range(self.featAttruct_num)]
        vector = [torch.cat((vector1[i], vector2[i], difference[i]), 1) for i in range(self.featAttruct_num)]

        weight = [self.score[i](vector[i]) for i in range(self.featAttruct_num)]
        value = [self.dnn(vector[i]).to(self.device) for i in range(self.featAttruct_num)]
        out = [weight[i] * value[i] for i in range(self.featAttruct_num)]
        output = torch.zeros(out[0].shape)
        for i in range(self.featAttruct_num):
            output += out[i]
        return output, L2


class SimCSE(nn.Module):
    def __init__(self, config, is_cross, device, max_length):
        super(SimCSE, self).__init__()

        self.is_cross = is_cross
        self.max_length = max_length
        self.device = device
        self.dim = 768

        self.tokenizer = AutoTokenizer.from_pretrained(config)
        self.bert = BertModel.from_pretrained(config).to(self.device)

    def forward(self, sentence):
        input = self.tokenizer(sentence, padding='max_length', truncation=True, max_length=self.max_length)
        input_ids = torch.tensor(input['input_ids'], dtype=torch.long).to(self.device)
        input_att = torch.tensor(input['attention_mask'], dtype=torch.long).to(self.device)
        vector = torch.mean(self.bert(input_ids, attention_mask=input_att)[0], 1).to(self.device)
        return vector


class SimSwAV(nn.Module):
    def __init__(self, config, device, max_length, class_num):
        super(SimSwAV, self).__init__()

        self.dim = 768
        self.max_length = max_length
        self.device = device
        self.class_num = class_num

        self.tokenizer = AutoTokenizer.from_pretrained(config)
        self.bert = BertModel.from_pretrained(config).to(self.device)
        self.C = nn.Parameter(torch.FloatTensor(self.dim, self.class_num))

    def forward(self, sentence1, sentence2):
        input1 = self.tokenizer(sentence1, padding='max_length', truncation=True, max_length=self.max_length)
        input_ids1 = torch.tensor(input1['input_ids'], dtype=torch.long).to(self.device)
        input_att1 = torch.tensor(input1['attention_mask'], dtype=torch.long).to(self.device)
        vector1 = torch.mean(self.bert(input_ids1, attention_mask=input_att1)[0], 1).to(self.device)
        input2 = self.tokenizer(sentence2, padding='max_length', truncation=True, max_length=self.max_length)
        input_ids2 = torch.tensor(input2['input_ids'], dtype=torch.long).to(self.device)
        input_att2 = torch.tensor(input2['attention_mask'], dtype=torch.long).to(self.device)
        vector2 = torch.mean(self.bert(input_ids2, attention_mask=input_att2)[0], 1).to(self.device)
        vector1 * self.C
        vector2 * self.C
        return


class SimSaim(nn.Module):
    def __init__(self, config, device, max_length):
        super(SimSaim, self).__init__()

        self.dim = 768
        self.max_length = max_length
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(config)
        self.bert = BertModel.from_pretrained(config).to(self.device)

    def forward(self, sentence):
        input = self.tokenizer(sentence, padding='max_length', truncation=True, max_length=self.max_length)
        input_ids = torch.tensor(input['input_ids'], dtype=torch.long).to(self.device)
        input_att = torch.tensor(input['attention_mask'], dtype=torch.long).to(self.device)
        vector = torch.mean(self.bert(input_ids, attention_mask=input_att)[0], 1).to(self.device)
        return vector


class TianChiModel(nn.Module):
    def __init__(self, config, device, max_length):
        super(TianChiModel, self).__init__()

        self.max_length = max_length
        self.device = device
        self.dim = 768

        self.config = config
        self.config = AutoConfig.from_pretrained(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config)
        self.bert = AutoModel.from_pretrained(config)
        self.dnn = nn.Sequential(
            nn.Linear(self.dim * 3, 2),
            nn.Softmax(1)
        )

    def forward(self, sentence1_index, sentence2_index, labels):
        input_ids_ = []
        input_att_ = []
        for i in range(len(sentence1_index)):
            tmp1, tmp2 = self.tokenize(sentence1_index[i], sentence2_index[i])
            input_ids_.append(tmp1)
            input_att_.append(tmp2)
        input_ids = torch.tensor(input_ids_, dtype=torch.long)
        input_att = torch.tensor(input_att_, dtype=torch.long)
        labels_ = torch.tensor(labels, dtype=torch.long)
        out = self.bert(input_ids, attention_mask=input_att)[0]
        out1 = torch.zeros(len(sentence1_index), self.dim)
        out2 = torch.zeros(len(sentence2_index), self.dim)
        for i in range(len(sentence1_index)):
            out1[i, :] = out[i, 1: len(sentence1_index[i])+1].sum(0) / len(sentence1_index[i])
            out2[i, :] = out[i, len(sentence2_index[i])+2: -2].sum(0) / len(sentence2_index[i])
        difference = torch.abs(out1 - out2)
        return self.dnn(torch.cat((out1, out2, difference), 1))

    def tokenize(self, sentence1, sentence2):
        if len(sentence1) >= self.max_length:
            sentence1 = sentence1[: self.max_length+1]
        if len(sentence2) >= self.max_length:
            sentence2 = sentence2[: self.max_length+1]
        now = len(sentence1) + len(sentence2)
        pad = self.max_length * 2 - now
        if pad < 0:
            pad = 0
        return [101] + sentence1 + [102] + sentence2 + [102] + [0]*pad, [1]*(self.max_length*2+3-pad) + [0]*pad


class TianchiModelPretrain(nn.Module):
    def __init__(self, config, device, max_length):
        super(TianchiModelPretrain, self).__init__()

        self.max_length = max_length
        self.device = device
        self.dim = 768

        self.config = config
        self.config = AutoConfig.from_pretrained(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config)
        self.bert = AutoModel.from_pretrained(config)
        self.tensor2index = nn.Sequential(
            nn.Tanh(),
            nn.Linear(self.dim, self.vocabsize),
            nn.Softmax(1)
        )
        self.dnn = nn.Sequential(
            nn.Linear(self.dim * 3, 2),
            nn.Softmax(1)
        )

    def forward(self, sentence1, sentence2, masked_token):
        input = self.mask(sentence1, sentence2, masked_token)
        input_ids = torch.tensor(input['input_ids'], dtype=torch.long)
        input_att = torch.tensor(input['attention_mask'], dtype=torch.long)
        bert_tensor = self.bert(input_ids, attention_mask=input_att)[0]                 # 1*max_length*dim
        return self.tensor2index(bert_tensor[0, masked_token, :]), self.dnn(bert_tensor[:, 0, :])

    def tokenize(self, sentence1, sentence2, masked_token):
        input = self.tokenizer(sentence1, sentence2)
        input['input_ids'][0, masked_token] = 103
        return input

    def mask(self, sentence1, sentence2, masked_token):
        ids = [101] + sentence1 + [102] + sentence2 + [102]
        ids[masked_token] = 103
        att = [1] * (len(sentence1) + len(sentence2) + 3)
        input = {}
        input['input_ids'] = ids
        input['attention_mask'] = att
        return input

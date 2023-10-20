import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
from LossFunction import LossFunction
from SPPLayer2 import SPPLayer

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


class TrainingData(Dataset):
    """
    数据集构造
    """

    def __init__(self, work_path, save_path, SOH_path):
        super(TrainingData, self).__init__()
        self.workPath = work_path
        self.savePath = save_path
        self.SOHPath = SOH_path
        cell_files = [item for item in os.listdir(self.workPath) if item.endswith('cell_Features.csv')]
        temperature_files = [item for item in os.listdir(self.workPath) if item.endswith('temperature_Features.csv')]
        soh_files = [item for item in os.listdir(self.SOHPath) if item.endswith('_Capacity_svm.csv')]
        files_num = len(cell_files)
        cells = [f'cell{i}' for i in range(1, 97)]
        probes = [f'temp{i}' for i in range(1, 49)]
        self.cells = []
        self.temperatures = []
        self.SOHs = []
        self.groups_soh = []
        for i in range(files_num):
            f_cells = pd.read_csv(os.path.join(self.workPath, cell_files[i]), header=[0])
            f_temperatures = pd.read_csv(os.path.join(self.workPath, temperature_files[i]), header=[0])
            groups_cell = f_cells.groupby('group').apply(lambda x: x.loc[x.index.min(), 'group']).values
            num_fragments = len(groups_cell)
            # 看一下这里的group是不是齐全的
            features_cells = torch.Tensor(f_cells[cells].values.astype(np.float32)). \
                reshape(num_fragments, -1, 96).transpose(1, 2)
            features_temperatures = torch.Tensor(f_temperatures[probes].values.astype(np.float32)). \
                reshape(num_fragments, -1, 48).transpose(1, 2)
            self.cells.append(features_cells)
            self.temperatures.append(features_temperatures)
            f_soh = pd.read_csv(os.path.join(self.SOHPath, soh_files[i]), header=[0])
            paires = {item[0]: item[1] for item in f_soh[['group', '平均容量']].values}
            # 以-1填补没有soh的group
            soh = [paires[item] if item in paires.keys() else -1 for item in groups_cell]
            self.SOHs.append(torch.Tensor(soh).reshape(-1, 1))

        self.len = len(self.SOHs)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.cells[item], self.temperatures[item], self.SOHs[item]


class PositionEncoding(nn.Module):
    def __init__(self, d_model, drop=0.1, max_len=10):
        super(PositionEncoding, self).__init__()
        self.dropout = nn.Dropout(p=drop)

        pe = torch.zeros(max_len, d_model)
        # pe:(max_len,d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # position:(max_len,1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # div_term:(d_model//2+1,1)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe:(max_len,1,d_model)
        self.register_buffer('pe', pe)
        # save pe as 'pe'

    def forward(self, x):
        # (batch,4,64)
        x = (x.transpose(0, 1) + self.pe[:x.size(1), :]).transpose(0, 1)
        # x:(batch,4,64)

        return self.dropout(x)


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q, features = seq_q.size()
    batch_size, len_k, features = seq_k.size()

    pad_attn_mask = torch.eq(torch.sum(torch.eq(seq_k, 0), dim=2), 0).unsqueeze(1)
    # 判断各位置是否为0，插入一个维度
    # pad_attn_mask:(batch_size,1,len_k)
    return pad_attn_mask.expand(batch_size, len_q, len_k)
    # 在第二个维度上复制len_q次
    # pad_attn_mask:(batch_size,len_q,len_k)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        # attn_mask:(batch_size,num_heads,seq_len,seq_len)
        # Q:(batch_size,num_heads,1,d_k)
        # K:(batch_size,num_heads,len_k,d_k)
        # V:(batch_size,num_heads,len_k,d_v)

        d_k = K.size(-1)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask[:, :, -1, :].unsqueeze(2), -1e9)
        # 对scores中0的部分填充

        # scores = nn.Softmax(dim=-1)(scores)
        # 最后一层做softmax

        context = torch.matmul(scores, V)
        # context:(batch_size,num_heads,1,d_v)

        return context, scores


class PosWiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden_size, bias=False):
        super(PosWiseFeedForward, self).__init__()
        self.d_model = d_model

        self.dense1 = nn.Linear(self.d_model, hidden_size, bias=bias)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(hidden_size, self.d_model, bias=bias)

    def forward(self, inputs):
        # (batch_size, len_q, d_model)
        residual = inputs
        output = self.dense2(self.relu(self.dense1(inputs)))
        return output + residual


class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, num_heads, d_model, drop_out, bias=False):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.dotProduct = ScaledDotProductAttention()

        self.W_q = nn.Linear(self.d_model, self.d_k * self.n_heads, bias=bias)
        self.W_k = nn.Linear(self.d_model, self.d_k * self.n_heads, bias=bias)
        self.W_v = nn.Linear(self.d_model, self.d_v * self.n_heads, bias=bias)

        self.fc = nn.Linear(self.n_heads * self.d_v, self.d_model, bias=bias)

    def forward(self, inputs, attn_mask):
        # input_Q:(batch_size,len_q,d_model)

        residual, batch_size = inputs, inputs.size(0)

        Q = self.W_q(inputs[:, -1, :].unsqueeze(1)).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(inputs).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(inputs).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        # Q:(batch_size,num_heads,1,d_k)
        # K:(batch_size,num_heads,len_k,d_k)
        # V:(batch_size,num_heads,len_k,d_v)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        # attn_mask:(batch_size,seq_len,seq_len) --> (batch_size,num_heads,seq_len,seq_len)

        context, attn = self.dotProduct(Q, K, V, attn_mask)
        # context:(batch_size,num_heads,len_q,d_v)

        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        # 把各头数据拼接
        # context:(batch_size,len_q,num_heads*d_v)

        output = self.fc(context)
        # output:(batch_size,len_q,d_model)

        return output + residual[0][-1][:]


class EncoderLayer(nn.Module):
    def __init__(self, d_k, d_v, num_heads, d_model, hidden_size, drop_out):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_k, d_v, num_heads, d_model, drop_out, bias=True)
        self.pos_ffn = PosWiseFeedForward(d_model, hidden_size)

    def forward(self, enc_inputs, enc_self_attn):
        # enc_inputs:(batch_size,src_len,d_model)
        # enc_self_attn:(batch_size,src_len,src_len)

        enc_outputs = self.enc_self_attn(enc_inputs, enc_self_attn)
        # enc_outputs:(batch_size,len_q,d_model)
        # attn:(batch_size,num_heads,src_len,src_len)

        enc_outputs = self.pos_ffn(enc_outputs)
        # enc_outputs:(batch_size,len_q,d_model)

        return enc_outputs


class Encoder(nn.Module):
    def __init__(self, d_k, d_v, num_heads, d_model, encoder_hidden_size, drop_out):
        super(Encoder, self).__init__()
        self.encoder_layer = EncoderLayer(d_k, d_v, num_heads, d_model, encoder_hidden_size, drop_out)
        self.enc_pos_emb = PositionEncoding(d_model)

    def forward(self, enc_inputs):
        # enc_inputs:(batch_size,src_len,feature_size)
        # (seq_len-3,4,64)

        enc_outputs = self.enc_pos_emb(enc_inputs)

        enc_self_attn_mask = get_attn_pad_mask(enc_outputs, enc_outputs).to(device)
        enc_outputs = self.encoder_layer(enc_outputs, enc_self_attn_mask)
        # enc_outputs:(batch_size,1,d_model)
        enc_outputs = enc_outputs.view(enc_outputs.shape[0], -1)
        # enc_outputs:(batch_size,d_model)

        return enc_outputs


class ResNet_LSTM(nn.Module):
    def __init__(self):
        super(ResNet_LSTM, self).__init__()
        self.branch1 = nn.Linear(in_features=32, out_features=4, bias=True)
        self.branch2_1 = nn.Linear(in_features=32, out_features=16, bias=True)
        self.branch2_2 = nn.Linear(in_features=16, out_features=4, bias=True)
        self.activation = nn.ReLU()
        self.activation2 = nn.Tanh()
        self.linear = nn.Linear(in_features=4, out_features=1, bias=True)

    def forward(self, x):
        # (seq_len-3,64)
        x1 = self.activation(self.branch1(x))

        x2 = self.activation(self.branch2_1(x))
        x2 = self.activation(self.branch2_2(x2))

        x = self.activation(self.linear(x1 + x2))
        return x


class LSTMModel(nn.Module):
    def __init__(self, d_k, d_v, num_heads, d_model, encoder_hidden_size, lstm_hidden_size, num_layers, drop_out=0.1):
        super(LSTMModel, self).__init__()
        self.encoder = Encoder(d_k, d_v, num_heads, d_model, encoder_hidden_size, drop_out)
        self.num_layers = num_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm = nn.LSTM(input_size=32, hidden_size=self.lstm_hidden_size, num_layers=self.num_layers,
                            batch_first=True, bidirectional=True)
        self.resnet = ResNet_LSTM()

    def forward(self, x):
        # (batch,seq_len,64)
        # Encoder提取四个矩阵的相关性，再接入LSTM
        batch_size, seq_len, features = x.shape[0], x.shape[1], x.shape[2]
        hidden = (
            torch.zeros(self.num_layers * 2, batch_size, self.lstm_hidden_size).to(device),
            torch.zeros(self.num_layers * 2, batch_size, self.lstm_hidden_size).to(device))
        index = [i for i in range(seq_len)]
        index = [item for item in zip(index, index[1:], index[2:], index[3:],
                                      index[4:], index[5:], index[6:]) if len(item) == 7]
        x = [x[0, item].unsqueeze(0) for item in index]
        x = torch.cat(x, dim=0)
        # (seq_len-6,7,64)
        x = self.encoder(x)
        # (seq_len-6,224)
        x = x.unsqueeze(0)
        x, _ = self.lstm(x, hidden)
        # (1,seq_len-6,64)
        x = self.resnet(x.reshape(seq_len - 6, -1))
        # (seq_len-6,1)
        return x.unsqueeze(0)


class InceptionA(nn.Module):
    def __init__(self):
        super(InceptionA, self).__init__()
        self.branch1 = nn.Conv2d(in_channels=20, out_channels=10, kernel_size=3, stride=2, padding=1)
        self.branch2 = nn.Conv2d(in_channels=20, out_channels=10, kernel_size=5, stride=2, padding=2)
        self.branch3 = nn.ModuleList(
            [nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3, stride=2, padding=1),
             nn.Conv2d(in_channels=40, out_channels=10, kernel_size=5, stride=1, padding=2)])

        self.activation = nn.ReLU()

    def forward(self, x):
        # (batch*seq_len,10,55,96)
        x1 = self.activation(self.branch1(x))
        x2 = self.activation(self.branch2(x))
        x3 = self.activation(self.branch3[1](self.activation(self.branch3[0](x))))
        # (batch*seq_len,5,28,48)
        return x1 + x2 + x3


class InceptionB(nn.Module):
    def __init__(self):
        super(InceptionB, self).__init__()
        self.branch1 = nn.Conv2d(in_channels=20, out_channels=10, kernel_size=3, stride=2, padding=1)
        self.branch2 = nn.Conv2d(in_channels=20, out_channels=10, kernel_size=5, stride=2, padding=2)
        self.branch3 = nn.ModuleList(
            [nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3, stride=2, padding=1),
             nn.Conv2d(in_channels=40, out_channels=10, kernel_size=5, stride=1, padding=2)])

        self.activation = nn.ReLU()

    def forward(self, x):
        # (batch*seq_len,10,55,36)
        x1 = self.activation(self.branch1(x))
        x2 = self.activation(self.branch2(x))
        x3 = self.activation(self.branch3[1](self.activation(self.branch3[0](x))))
        # (batch*seq_len,10,28,18)
        return x1 + x2 + x3


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.extend_cov = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 5), stride=(1, 1), padding=(1, 2))
        self.conv1 = nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(5, 7), stride=(2, 3), padding=(2, 3))
        self.bn1 = nn.BatchNorm2d(num_features=10)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=5, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(num_features=5)
        self.shortcut = nn.Conv2d(in_channels=20, out_channels=5, kernel_size=(3, 5), stride=(4, 6), padding=(1, 2))
        self.conv3 = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.activation = nn.ReLU()

    def forward(self, x):
        # (batch*seq_len,10,28,96)
        x = self.extend_cov(x)
        # (batch*seq_len,20,28,96)
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation(out)
        # (batch*seq_len,5,7,16)
        out = self.activation(self.conv3(out))
        # (batch*seq_len,1,7,16)

        return out.squeeze(1)


class ExtendLayer1(nn.Module):
    def __init__(self, features):
        super(ExtendLayer1, self).__init__()
        self.linear = nn.Linear(in_features=features, out_features=128, bias=True)
        self.extend_cnn = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(5, 3), stride=(2, 1),
                                    padding=(2, 1))
        self.activation = nn.ReLU()

    def forward(self, x):
        # (1,96,17)
        x = x.unsqueeze(0)
        x = self.activation(self.linear(x))
        x = self.activation(self.extend_cnn(x))
        # (1,20,48,128)
        return x


class Extend1(nn.Module):
    def __init__(self):
        super(Extend1, self).__init__()
        self.layer1 = ExtendLayer1(features=12)
        self.layer2 = ExtendLayer1(features=15)

    def forward(self, x):
        # batch*seq_len,1,96,17
        l = x.shape[0]
        outputs = []
        for num_fragment in range(l):
            if torch.all(torch.eq(x[num_fragment, 0, :, -4], x[num_fragment, 0, 0, -4])):
                # 最后一列相等，则为充电片段
                outputs.append(self.layer1(x[num_fragment, :, :, :-3]))
            else:
                outputs.append(self.layer2(x[num_fragment]))
        # (1,20,48,128)
        return torch.cat(outputs, dim=0)


class ExtendLayer2(nn.Module):
    def __init__(self, features):
        super(ExtendLayer2, self).__init__()
        self.linear = nn.Linear(in_features=features, out_features=64, bias=True)
        self.extend_cnn = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(5, 3), stride=(2, 1),
                                    padding=(2, 1))
        self.activation = nn.ReLU()

    def forward(self, x):
        # (1,96,8)
        x = x.unsqueeze(0)
        x = self.activation(self.linear(x))
        x = self.activation(self.extend_cnn(x))
        # (1,20,48,64)
        return x


class Extend2(nn.Module):
    def __init__(self):
        super(Extend2, self).__init__()
        self.layer1 = ExtendLayer2(features=6)
        self.layer2 = ExtendLayer2(features=7)

    def forward(self, x):
        # batch*seq_len,1,96,8
        l = x.shape[0]
        outputs = []
        for num_fragment in range(l):
            if torch.all(torch.eq(x[num_fragment, 0, :, -1], x[num_fragment, 0, 0, -1])):
                # 最后一列相等，则为充电片段
                outputs.append(self.layer1(x[num_fragment, :, :, :-1]))
            else:
                outputs.append(self.layer2(x[num_fragment]))
        return torch.cat(outputs, dim=0)


class FeedForward(nn.Module):
    def __init__(self):
        super(FeedForward, self).__init__()

        self.dense1 = nn.Linear(in_features=96, out_features=128, bias=True)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(in_features=128, out_features=32, bias=True)
        self.dense3 = nn.Linear(in_features=96, out_features=32, bias=True)

    def forward(self, inputs):
        # (batch,112)
        output = self.dense2(self.relu(self.dense1(inputs)))
        output = self.relu(self.dense3(inputs) + output)
        # (batch,64)
        return output


class FeatureModel(nn.Module):
    def __init__(self):
        super(FeatureModel, self).__init__()
        self.bn_cell = nn.BatchNorm1d(num_features=15)
        self.bn_temperature = nn.BatchNorm1d(num_features=7)
        self.block1 = InceptionA()
        self.block2 = InceptionB()
        self.resnet = ResNet()
        self.extendLayer_cell = Extend1()
        self.extendLayer_temperature = Extend2()
        self.spp = SPPLayer(num_levels=10, pool_type='avg_pool')
        self.fc = FeedForward()

    def forward(self, cell_features, temperature_features):
        # cell_features:(batch,seq_len,96,17)
        # temperature_features:(batch,seq_len,48,8)
        batch, seq_len, num_cells, num_cell_features = cell_features.shape[0], cell_features.shape[1], \
            cell_features.shape[2], cell_features.shape[3]
        num_probes, num_temperature_features = temperature_features.shape[2], temperature_features.shape[3]

        cell_features = cell_features.reshape(-1, num_cells, num_cell_features)
        temperature_features = temperature_features.reshape(-1, num_probes, num_temperature_features)
        # 归一化
        cell_features = self.bn_cell(cell_features.transpose(1, 2)).transpose(1, 2)
        temperature_features = self.bn_temperature(temperature_features.transpose(1, 2)).transpose(1, 2)
        # (batch*seq_len,96,17),(batch*seq_len,48,8)

        cell_features = cell_features.unsqueeze(1)
        temperature_features = temperature_features.unsqueeze(1).repeat(1, 1, 2, 1)
        # (batch*seq_len,1,96,17),(batch*seq_len,1,96,8)

        cell_features = self.extendLayer_cell(cell_features)
        temperature_features = self.extendLayer_temperature(temperature_features)
        # (batch*seq_len,20,48,128),(batch*seq_len,20,48,64)

        # cell_features = self.spp(cell_features)
        # temperature_features = self.spp(temperature_features)
        # (batch*seq_len,10,55,96),(batch*seq_len,10,55,36)

        cell_features = self.block1(cell_features)
        temperature_features = self.block2(temperature_features)
        # (batch*seq_len,10,28,64),(batch*seq_len,10,28,32)

        x = torch.cat((cell_features, temperature_features), dim=-1)
        # (batch*seq_len,10,28,96)
        x = self.resnet(x)
        # (batch*seq_len,7,16)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        # (batch*seq_len,64)
        x = x.reshape(batch, seq_len, -1)
        # (batch,seq_len,64)
        return x


class SOHEstimation(nn.Module):
    def __init__(self, d_k, d_v, num_heads, d_model, encoder_hidden_size, lstm_hidden_size, num_layers, drop_out=0.01):
        super(SOHEstimation, self).__init__()
        self.feature_layer = FeatureModel()
        self.lstm_layer = LSTMModel(d_k, d_v, num_heads, d_model, encoder_hidden_size, lstm_hidden_size, num_layers,
                                    drop_out)

    def forward(self, cell_features, temperature_features):
        # cell_features:(batch,seq_len,96,17)
        # temperature_features:(batch,seq_len,48,8)
        x = self.feature_layer(cell_features, temperature_features)
        # (batch,seq_len,64)
        x = self.lstm_layer(x)
        return x


batch_size = 1
d_k = 6
d_v = 6
num_heads = 3
d_model = 32
encoder_hidden_size = 8
lstm_hidden_size = 16

workPath = 'E:\\pycharm\\DigitalCarRace\\SOHFeatureFile'
savePath = 'E:\\pycharm\\DigitalCarRace\\SOHEstimation'
SOHPath = 'E:\\pycharm\\DigitalCarRace\\RedefineSOH'
trainData = TrainingData(workPath, savePath, SOHPath)
train_loader = DataLoader(trainData, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
model = SOHEstimation(d_k, d_v, num_heads, d_model, encoder_hidden_size, lstm_hidden_size, num_layers=3).to(device)
criterion = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)


def train(epoch):
    running_loss = 0
    for batch_idx, data in enumerate(train_loader, 0):
        cells, temperatures, SOHs = data
        SOHs = SOHs[:, 6:]
        cells, temperatures, SOHs = cells.to(device), temperatures.to(device), SOHs.to(device)
        pred = model(cells, temperatures)
        pred = pred.reshape(-1)
        SOHs = SOHs.reshape(-1)
        for idx in torch.eq(SOHs, -1).nonzero():
            SOHs[idx] = pred[idx]
        # pred[torch.eq(SOHs, -1).nonzero()] = -1
        loss = criterion(pred, SOHs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss


def valid(epoch):
    with torch.no_grad():
        for data in train_loader:
            cells, temperatures, SOHs = data
            SOHs = SOHs[:, 6:]
            cells, temperatures, SOHs = cells.to(device), temperatures.to(device), SOHs.to(device)
            outputs = model(cells, temperatures)
            outputs = outputs.view(-1, 1)
            outputs = outputs.to(torch.device('cpu'))
            SOHs = SOHs.view(-1, 1).to(torch.device('cpu'))
            comp = np.array(torch.concat([outputs, SOHs], dim=1))
            pd.DataFrame(comp, columns=['pred', 'target']).to_csv(
                f'E:/pycharm/DigitalCarRace/SOHmodel/SOHModel{epoch}.csv', index=False)


if __name__ == '__main__':
    loss = []
    epoch = 0
    while True:
        subloss = train(epoch)
        epoch += 1
        print(epoch, subloss)
        loss += [subloss]
        if epoch % 10 == 0:
            torch.save(model, os.path.join(savePath, f'SOHEstimationModel{epoch}.pkl'))
            df_loss = pd.DataFrame(loss, columns=['loss'])
            df_loss.to_csv(os.path.join(savePath, f'SOHEstimationModel_loss.csv'), index=False)
            valid(epoch)

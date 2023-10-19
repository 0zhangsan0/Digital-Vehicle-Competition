import os

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from SPPLayer2 import SPPLayer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# device=torch.device('cpu')


class TrainTestSplit:
    def __init__(self, work_path, save_path, file_name1, SOHPath):
        self.workPath = work_path
        self.savePath = save_path
        self.fileName1 = file_name1
        f1_1 = pd.read_csv(os.path.join(self.workPath, self.fileName1[0]), header=[0])
        f1_2 = pd.read_csv(os.path.join(self.workPath, self.fileName1[1]), header=[0])
        f2 = pd.read_csv(SOHPath, header=[0])
        f2['SOH3'] = f2['SOH3'] / f2['SOH3'].max()
        self.groups = f2['group'].values
        self.SOHs = f2['SOH3'].values

        def assignmentSOH(x):
            group = x.loc[x.index.min(), 'group']
            if group in self.groups:
                x['SOH'] = f2.loc[f2['group'] == group, 'SOH3'].values[0]
                return x

        self.f1 = f1_1.groupby('group', group_keys=False).apply(assignmentSOH)
        self.f2 = f1_2.groupby('group', group_keys=False).apply(assignmentSOH)

    def splitDataset(self):
        groupTrain, groupTest, SOHTrain, SOHTest = train_test_split(self.groups, self.SOHs, test_size=0.2,
                                                                    random_state=42)
        dataToTrain_cell = self.f1.groupby('group', group_keys=False).apply(
            lambda x: x if x.loc[x.index.min(), 'group'] in groupTrain else None)
        dataToTrain_temperature = self.f2.groupby('group', group_keys=False).apply(
            lambda x: x if x.loc[x.index.min(), 'group'] in groupTrain else None)
        dataToTest_cell = self.f1.groupby('group', group_keys=False).apply(
            lambda x: x if x.loc[x.index.min(), 'group'] in groupTest else None)
        dataToTest_temperature = self.f2.groupby('group', group_keys=False).apply(
            lambda x: x if x.loc[x.index.min(), 'group'] in groupTest else None)
        return dataToTrain_cell, dataToTrain_temperature, dataToTest_cell, dataToTest_temperature


class TrainingData(Dataset):
    """
    数据集构造
    """

    def __init__(self, work_path, save_path, df_cell, df_temperature):
        super(TrainingData, self).__init__()
        self.workPath = work_path
        self.savePath = save_path
        cells = [f'cell{i}' for i in range(1, 97)]
        probes = [f'temp{i}' for i in range(1, 49)]
        self.SOHs = df_cell.groupby('group').apply(lambda x: x.loc[x.index.min(), 'SOH']).values
        self.SOHs = torch.Tensor(self.SOHs).reshape(-1, 1)
        num = len(self.SOHs)
        data_cell = df_cell[cells].values
        data_temperature = df_temperature[probes].values
        self.data_cell = torch.Tensor(data_cell).reshape(num, -1, 96).transpose(1, 2)
        self.data_temperature = torch.Tensor(data_temperature).reshape(num, -1, 48).transpose(1, 2)
        self.len = num

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.data_cell[item], self.data_temperature[item], self.SOHs[item]


class InceptionA(nn.Module):
    def __init__(self):
        super(InceptionA, self).__init__()
        self.branch1 = nn.Conv2d(in_channels=10, out_channels=5, kernel_size=3, stride=2, padding=1)
        self.branch2 = nn.Conv2d(in_channels=10, out_channels=5, kernel_size=5, stride=2, padding=2)
        self.branch3 = nn.ModuleList(
            [nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=2, padding=1),
             nn.Conv2d(in_channels=20, out_channels=5, kernel_size=5, stride=1, padding=2)])

        self.activation = nn.ReLU()

    def forward(self, x):
        # (batch,10,55,96)
        x1 = self.activation(self.branch1(x))
        x2 = self.activation(self.branch2(x))
        x3 = self.activation(self.branch3[1](self.activation(self.branch3[0](x))))
        # (batch,5,28,48)
        return x1 + x2 + x3


class InceptionB(nn.Module):
    def __init__(self):
        super(InceptionB, self).__init__()
        self.branch1 = nn.Conv2d(in_channels=10, out_channels=5, kernel_size=3, stride=2, padding=1)
        self.branch2 = nn.Conv2d(in_channels=10, out_channels=5, kernel_size=5, stride=2, padding=2)
        self.branch3 = nn.ModuleList(
            [nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=2, padding=1),
             nn.Conv2d(in_channels=20, out_channels=5, kernel_size=5, stride=1, padding=2)])

        self.activation = nn.ReLU()

    def forward(self, x):
        # (batch,10,55,36)
        x1 = self.activation(self.branch1(x))
        x2 = self.activation(self.branch2(x))
        x3 = self.activation(self.branch3[1](self.activation(self.branch3[0](x))))
        # (batch,10,28,18)
        return x1 + x2 + x3


# class PositionEncoding(nn.Module):
#     def __init__(self, d_model):
#         super(PositionEncoding, self).__init__()
#         self.d_model = d_model
#
#         # position:(max_len,1)
#         self.div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).to(device)
#         # div_term:(d_model//2,1)
#
#     def forward(self, x, positionInfo):
#         # x:(batch_size,src_len,d_model)
#         # positionInfo:(batch,src_len,1)
#         positionInfo = positionInfo.transpose(1, 2).repeat(1, 96, 1).view(positionInfo.size(0) * 96, -1, 1)
#         pe = torch.zeros(x.shape).to(device)
#         pe[:, :, 0::2] = torch.sin(positionInfo * self.div_term)
#         pe[:, :, 1::2] = torch.cos(positionInfo * self.div_term)
#         x = x + pe
#         # (batch_size,src_len,d_model)
#         return x
#
#
# class ScaledDotProductAttention(nn.Module):
#     def __init__(self):
#         super(ScaledDotProductAttention, self).__init__()
#
#     def forward(self, Q, K, V):
#         # attn_mask:(batch_size,num_heads,seq_len,seq_len)
#         # Q:(batch_size,num_heads,1,d_k)
#         # K:(batch_size,num_heads,len_k,d_k)
#         # V:(batch_size,num_heads,len_k,d_v)
#
#         d_k = K.size(-1)
#         scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
#
#         attn = nn.Softmax(dim=-1)(scores)
#         # 最后一层做softmax
#
#         context = torch.matmul(attn, V)
#         # context:(batch_size,num_heads,1,d_v)
#
#         return context, attn
#
#
# class PosWiseFeedForward(nn.Module):
#     def __init__(self, d_model, hidden_size, bias=False):
#         super(PosWiseFeedForward, self).__init__()
#         self.d_model = d_model
#
#         self.dense1 = nn.Linear(self.d_model, hidden_size, bias=bias)
#         self.relu = nn.ReLU()
#         self.dense2 = nn.Linear(hidden_size, self.d_model, bias=bias)
#
#     def forward(self, inputs):
#         # (batch_size, len_q, d_model)
#         residual = inputs
#         output = self.dense2(self.relu(self.dense1(inputs)))
#         return output + residual
#
#
# class AddNorm(nn.Module):
#     # Residual、Normalized
#     def __init__(self, normalized_shape, drop_out):
#         super(AddNorm, self).__init__()
#         self.dropout = nn.Dropout(drop_out)
#         self.ln = nn.LayerNorm(normalized_shape)
#
#     def forward(self, X, Y):
#         # X经过MultiHeadAttention
#         return self.ln(self.dropout(Y) + X)
#
#
# class MultiHeadAttention(nn.Module):
#     def __init__(self, d_k, d_v, num_heads, d_model, drop_out, bias=False):
#         super(MultiHeadAttention, self).__init__()
#         self.n_heads = num_heads
#         self.d_k = d_k
#         self.d_v = d_v
#         self.d_model = d_model
#         self.normLayer = AddNorm(self.d_model, drop_out)
#         self.dotProduct = ScaledDotProductAttention()
#
#         self.W_q = nn.Linear(self.d_model, self.d_k * self.n_heads, bias=bias)
#         self.W_k = nn.Linear(self.d_model, self.d_k * self.n_heads, bias=bias)
#         self.W_v = nn.Linear(self.d_model, self.d_v * self.n_heads, bias=bias)
#
#         self.fc = nn.Linear(self.n_heads * self.d_v, self.d_model, bias=bias)
#
#     def forward(self, input_Q, input_K, input_V):
#         # input_Q:(batch_size,len_q,d_model)
#         # input_K:(batch_size,len_k,d_model)
#         # input_V:(batch_size,len_k,d_model)
#
#         residual, batch_size = input_Q, input_Q.size(0)
#
#         Q = self.W_q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
#         K = self.W_k(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
#         V = self.W_v(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
#         # Q:(batch_size,num_heads,len_q,d_k)
#         # K:(batch_size,num_heads,len_k,d_k)
#         # V:(batch_size,num_heads,len_k,d_v)
#
#         context, attn = self.dotProduct(Q, K, V)
#         # context:(batch_size,num_heads,len_q,d_v)
#
#         context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
#         # 把各头数据拼接
#         # context:(batch_size,len_q,num_heads*d_v)
#
#         output = self.fc(context)
#         # output:(batch_size,len_q,d_model)
#
#         return self.normLayer.to(device)(output, residual), attn
#
#
# class EncoderLayer(nn.Module):
#     def __init__(self, d_k, d_v, num_heads, d_model, hidden_size, drop_out):
#         super(EncoderLayer, self).__init__()
#         self.enc_self_attn = MultiHeadAttention(d_k, d_v, num_heads, d_model, drop_out, bias=True)
#         self.pos_ffn = PosWiseFeedForward(d_model, hidden_size)
#
#     def forward(self, enc_inputs):
#         # enc_inputs:(batch_size,src_len,d_model)
#         # enc_self_attn:(batch_size,src_len,src_len)
#
#         enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)
#         # enc_outputs:(batch_size,len_q,d_model)
#         # attn:(batch_size,num_heads,src_len,src_len)
#
#         enc_outputs = self.pos_ffn(enc_outputs)
#         # enc_outputs:(batch_size,len_q,d_model)
#
#         return enc_outputs, attn
#
#
# class Encoder(nn.Module):
#     def __init__(self, d_k, d_v, num_heads, d_model, encoder_hidden_size, drop_out):
#         super(Encoder, self).__init__()
#         self.encoder_layer = EncoderLayer(d_k, d_v, num_heads, d_model, encoder_hidden_size, drop_out)
#         self.enc_pos_emb = PositionEncoding(d_model)
#
#     def forward(self, enc_inputs):
#         # enc_inputs:(batch_size,src_len,feature_size)
#
#         enc_outputs = self.enc_pos_emb(enc_inputs).to(device)
#
#         enc_outputs, enc_self_attn = self.encoder_layer(enc_outputs)
#         # enc_outputs:(batch_size,src_len,d_model)
#         enc_outputs = enc_outputs.view(batch_size, -1, enc_outputs.shape[1], enc_outputs.shape[2]).transpose(1, 2)
#         # enc_outputs:(batch_size,src_len,96,d_model)
#         return enc_outputs


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.extend_cov = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=(3, 5), stride=(1, 1), padding=(1, 2))
        self.conv1 = nn.Conv2d(in_channels=10, out_channels=5, kernel_size=(5, 7), stride=(2, 3), padding=(2, 3))
        self.bn1 = nn.BatchNorm2d(num_features=5)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 0))
        self.bn2 = nn.BatchNorm2d(num_features=2)
        self.shortcut = nn.Conv2d(in_channels=10, out_channels=2, kernel_size=(3, 5), stride=(4, 7), padding=(1, 2))
        self.conv3 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.activation = nn.ReLU()

    def forward(self, x):
        # (batch,5,28,66)
        x = self.extend_cov(x)
        # (batch,10,28,66)
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation(out)
        # (batch,2,7,10)
        out = self.activation(self.conv3(out))
        # (batch,1,7,10)

        return out.squeeze(1)


class ExtendLayer1(nn.Module):
    def __init__(self):
        super(ExtendLayer1, self).__init__()
        self.linear = nn.Linear(in_features=12, out_features=96, bias=True)
        self.extend_cnn = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(5, 3), stride=(2, 1),
                                    padding=(2, 1))
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.linear(x))
        x = self.activation(self.extend_cnn(x))
        return x


class ExtendLayer2(nn.Module):
    def __init__(self):
        super(ExtendLayer2, self).__init__()
        self.linear = nn.Linear(in_features=6, out_features=36, bias=True)
        self.extend_cnn = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(5, 3), stride=(2, 1),
                                    padding=(2, 1))
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.linear(x))
        x = self.activation(self.extend_cnn(x))
        return x


class FeedForward(nn.Module):
    def __init__(self):
        super(FeedForward, self).__init__()

        self.dense1 = nn.Linear(in_features=25, out_features=16, bias=True)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(in_features=16, out_features=8, bias=True)
        self.dense3 = nn.Linear(in_features=25, out_features=8, bias=True)

    def forward(self, inputs):
        # (batch,25)
        output = self.dense2(self.relu(self.dense1(inputs)))
        output = self.relu(self.dense3(inputs) + output)
        # (batch,8)
        return output


class CapacityEstimation(nn.Module):
    def __init__(self):
        super(CapacityEstimation, self).__init__()
        self.bn_cell = nn.BatchNorm1d(num_features=96)
        self.bn_temperature = nn.BatchNorm1d(num_features=48)
        self.block1 = InceptionA()
        self.block2 = InceptionB()
        self.resnet = ResNet()
        self.extendLayer_cell = ExtendLayer1()
        self.extendLayer_temperature = ExtendLayer2()
        self.spp = SPPLayer(num_levels=10, pool_type='avg_pool')
        self.conv = nn.Conv1d(in_channels=7, out_channels=5, kernel_size=3, stride=2, padding=1)
        self.fc = FeedForward()
        self.linear = nn.Linear(in_features=8, out_features=1, bias=True)

    def forward(self, cell_features, temperature_features):
        # 归一化
        cell_features = self.bn_cell(cell_features)
        temperature_features = self.bn_temperature(temperature_features)

        cell_features = cell_features.unsqueeze(1)
        temperature_features = temperature_features.unsqueeze(1).repeat(1, 1, 2, 1)
        # (batch,1,96,12),(batch,1,96,6)

        cell_features = self.extendLayer_cell(cell_features)
        temperature_features = self.extendLayer_temperature(temperature_features)
        # (batch,10,48,96),(batch,10,48,36)

        cell_features = self.spp(cell_features)
        temperature_features = self.spp(temperature_features)
        # (batch,10,55,96),(batch,10,55,36)

        cell_features = self.block1(cell_features)
        temperature_features = self.block2(temperature_features)
        # (batch,5,28,48),(batch,5,28,18)

        x = torch.cat((cell_features, temperature_features), dim=-1)
        # (batch,5,28,66)
        x = self.resnet(x)
        # (batch,7,10)
        x = self.conv(x)
        # (batch,5,5)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        x = self.linear(x)
        return x


workPath = 'E:\\pycharm\\DigitalCarRace\\CapacityFeatureFile'
savePath = r'E:\pycharm\DigitalCarRace\CapacityModel1_1'
fileName1 = ['LFPHC7PE0K1A07972_cell_Features2.csv', 'LFPHC7PE0K1A07972_temperature_Features2.csv']
SOHpath = r'E:\pycharm\DigitalCarRace\chargeSet\LFPHC7PE0K1A07972_redefineCapacity.csv'
model1 = TrainTestSplit(workPath, savePath, fileName1, SOHpath)
trainData_cell, trainData_temperature, testData_cell, testData_temperature = model1.splitDataset()
trainData = TrainingData(workPath, savePath, trainData_cell, trainData_temperature)
train_loader = DataLoader(trainData, batch_size=6, shuffle=True, num_workers=10, drop_last=True)
testData = TrainingData(workPath, savePath, testData_cell, testData_temperature)
test_loader = DataLoader(testData, batch_size=1, shuffle=False, num_workers=10, drop_last=True)


def train(epoch):
    runningLoss = 0
    for batch_idx, data in enumerate(train_loader, 0):
        data_cell, data_temperature, SOHs = data
        data_cell, data_temperature, SOHs = \
            data_cell.to(device), data_temperature.to(device), SOHs.to(device)
        pred = model2(data_cell, data_temperature)
        loss = criterion(pred, SOHs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        runningLoss += loss.item()
    return runningLoss


def valid():
    runningLoss = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader, 0):
            data_cell, data_temperature, SOHs = data
            data_cell, data_temperature, SOHs = \
                data_cell.to(device), data_temperature.to(device), SOHs.to(device)
            pred = model2(data_cell, data_temperature)
            loss = criterion(pred, SOHs)

            runningLoss += loss.item()
    return runningLoss


if __name__ == '__main__':
    epoch = 0
    model2 = CapacityEstimation().to(device)
    criterion = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(params=model2.parameters(), lr=0.001)
    loss1 = []
    loss2 = []
    while True:
        subloss = train(epoch)
        epoch += 1
        print(epoch, subloss)
        loss1 += [subloss]
        if epoch % 50 == 0:
            torch.save(model2, os.path.join(savePath, f'CapacityEstimationModel{epoch}.pkl'))
            df_loss = pd.DataFrame(loss1, columns=['loss'])
            df_loss.to_csv(os.path.join(savePath, f'CapacityEstimationModel_loss.csv'), index=False)
            loss2 += [valid()]
            pd.DataFrame(loss2, columns=['loss']).to_csv(
                os.path.join(savePath, f'CapacityEstimationModel_valid_loss.csv'), index=False)

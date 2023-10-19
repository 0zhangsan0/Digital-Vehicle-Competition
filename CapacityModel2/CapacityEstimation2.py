import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
from sklearn.model_selection import train_test_split
from SPPLayer2 import SPPLayer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class TrainTestSplit:
    def __init__(self, work_path, save_path, file_name1, file_name2):
        self.workPath = work_path
        self.savePath = save_path
        self.fileName1 = file_name1
        self.fileName2 = file_name2
        f1 = pd.read_csv(os.path.join(self.workPath, self.fileName1), header=[0])
        f2 = pd.read_csv(os.path.join(self.workPath, self.fileName2), header=[0])
        self.groups = f2['group'].values
        self.SOHs = f2['安时容量'].values

        def assignmentSOH(x):
            group = x.loc[x.index.min(), 'group']
            if group in self.groups:
                x['SOH'] = float(f2.loc[f2['group'] == group, '安时容量'])
                return x

        self.f = f1.groupby('group', group_keys=False).apply(assignmentSOH)

    def splitDataset(self):
        groupTrain, groupTest, SOHTrain, SOHTest = train_test_split(self.groups, self.SOHs, test_size=0.2,
                                                                    random_state=42)
        dataToTrain = self.f.groupby('group', group_keys=False).apply(
            lambda x: x if x.loc[x.index.min(), 'group'] in groupTrain else None)
        dataToTest = self.f.groupby('group', group_keys=False).apply(
            lambda x: x if x.loc[x.index.min(), 'group'] in groupTest else None)
        return dataToTrain, dataToTest


class TrainingData(Dataset):
    """
    数据集构造
    """

    def __init__(self, work_path, save_path, df):
        super(TrainingData, self).__init__()
        self.workPath = work_path
        self.savePath = save_path
        df['数据采集时间'] = pd.to_datetime(df['数据采集时间'])
        cells = [f'cell{i}' for i in range(1, 97)]
        probes = [f'temp{i}' for i in range(1, 49)]
        self.SOHs = []
        self.voltages = []
        self.currents = []
        self.temps = []
        self.T = []
        for index, group in df.groupby('group'):
            t0 = group.loc[group.index.min(), '数据采集时间']
            self.T.append(
                np.array([[(t - t0).total_seconds()] for t in group['数据采集时间'].values]).astype(np.float32))
            self.SOHs.append([group.loc[group.index.min(), 'SOH'].astype(np.float32)])
            self.voltages.append(group[cells].values.astype(np.float32))
            self.currents.append(group[['总电流']].values.astype(np.float32))
            self.temps.append(group[probes].values.astype(np.float32))
        self.len = len(self.SOHs)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return torch.Tensor(self.T[item]), torch.Tensor(self.voltages[item]), torch.Tensor(self.temps[item]), \
            torch.Tensor(self.currents[item]), torch.Tensor(self.SOHs[item])


class InceptionA(nn.Module):
    def __init__(self):
        super(InceptionA, self).__init__()
        self.spp = SPPLayer(num_levels=13, pool_type='avg_pool')
        self.branch1 = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0).to(device)
        self.branch2 = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=5, stride=1, padding=2).to(device)
        self.branch3 = nn.ModuleList(
            [nn.Conv1d(in_channels=3, out_channels=5, kernel_size=3, stride=1, padding=1).to(device),
             nn.Conv1d(in_channels=5, out_channels=1, kernel_size=5, stride=1, padding=2).to(device)])

        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.spp(x)
        # (N,91,3)
        x = x.transpose(1, 2)
        # (N,3,91)
        x1 = self.activation(self.branch1(x)).transpose(1, 2)
        x2 = self.activation(self.branch2(x)).transpose(1, 2)
        x3 = x
        for layer in self.branch3:
            x3 = self.activation(layer(x3))
        x3 = x3.transpose(1, 2)
        # (N,91,3)
        return torch.cat((x1, x2, x3), dim=2)


class InceptionB(nn.Module):
    def __init__(self):
        super(InceptionB, self).__init__()
        self.linear1_1 = nn.Linear(in_features=96, out_features=48, bias=True).to(device)
        self.linear2_1 = nn.Linear(in_features=48, out_features=16, bias=True).to(device)
        self.linear3_1 = nn.Linear(in_features=16, out_features=4, bias=True).to(device)
        self.linear4 = nn.Linear(in_features=4, out_features=1, bias=True).to(device)
        self.linear1_2 = nn.Linear(in_features=48, out_features=48, bias=True).to(device)
        self.linear2_2 = nn.Linear(in_features=16, out_features=16, bias=True).to(device)
        self.linear3_2 = nn.Linear(in_features=4, out_features=4, bias=True).to(device)
        self.activation = nn.ReLU()

    def forward(self, x):
        # (batch,96)
        x = self.activation(self.linear1_1(x))
        x = self.activation(self.activation(self.linear1_2(x)) + x)
        x = self.activation(self.linear2_1(x))
        x = self.activation(self.activation(self.linear2_2(x)) + x)
        x = self.activation(self.linear3_1(x))
        x = self.activation(self.activation(self.linear3_2(x)) + x)
        x = self.activation(self.linear4(x))
        # (batch,1)
        return x


class FeedForward(nn.Module):
    def __init__(self):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(in_features=91, out_features=64).to(device)
        self.linears = nn.ModuleList([nn.Linear(in_features=64, out_features=16, bias=True).to(device),
                                      nn.Linear(in_features=16, out_features=4, bias=True).to(device),
                                      nn.Linear(in_features=4, out_features=1, bias=True).to(device)])
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=64, out_channels=16, kernel_size=1, stride=1, padding=0).to(device),
             nn.Conv1d(in_channels=16, out_channels=4, kernel_size=1, stride=1, padding=0).to(device),
             nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1, stride=1, padding=0).to(device)])
        self.activation = nn.ReLU()

    def forward(self, x):
        # (N,1,64)
        x = self.activation(self.linear1(x))
        for i in range(3):
            layer1 = self.linears[i]
            layer2 = self.convs[i]
            x1 = self.activation(layer1(x))
            x2 = layer2(x.transpose(1, 2)).transpose(1, 2)
            x = self.activation(x1 + x2)
        # (N,1,1)
        return x


class CapacityEstimation(nn.Module):
    def __init__(self):
        super(CapacityEstimation, self).__init__()
        self.fw = FeedForward()
        self.block1 = InceptionA()
        self.block2 = InceptionB()
        self.cnn1 = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0).to(device)
        self.cnn2 = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0).to(device)
        self.spp = SPPLayer(num_levels=13, pool_type='avg_pool')
        self.activation = nn.ReLU()

    def forward(self, T, voltages, temps, currents):
        T = T.unsqueeze(1)
        T = T.repeat(1, 96, 1, 1)
        voltages = voltages.transpose(1, 2)
        voltages = voltages.unsqueeze(3)
        currents = currents.unsqueeze(1)
        currents = currents.repeat(1, 96, 1, 1)
        temps = temps.transpose(1, 2)
        temps = temps.unsqueeze(3)
        temps = temps.repeat(1, 2, 1, 1)
        x = torch.cat((T, voltages, currents, temps), dim=3)
        # (batch,96,seq_len,4)
        batch_size, cell_num, seq_len, featrue_size = x.shape
        x1 = x[:, :, :, :featrue_size - 1]
        x2 = x[:, :, :, [-1]]
        x1 = x1.reshape(-1, seq_len, featrue_size - 1)
        x1 = self.block1(x1)
        x1 = x1.transpose(1, 2)
        # (N,91,1)
        x1 = self.activation(self.cnn1(x1)).transpose(1, 2)
        x2 = x2.reshape(-1, seq_len, 1)
        # (N,91,1)
        x2 = self.spp(x2)
        # (N,91,2)
        x = torch.cat((x1, x2), dim=2)
        x = x.transpose(1, 2)
        # (N,1,91)
        x = self.activation(self.cnn2(x))
        x = self.fw(x)
        # (batch,cell_num)
        x = x.reshape(batch_size, cell_num)
        x = self.block2(x)
        return x


workPath = 'E:\\pycharm\\DigitalCarRace\\chargeSet'
savePath = 'E:\\pycharm\\DigitalCarRace\\CapacityModel'
fileName1 = 'LFPHC7PE0K1A07972_charge.csv'
fileName2 = 'LFPHC7PE0K1A07972_redefineCapacity.csv'
model1 = TrainTestSplit(workPath, savePath, fileName1, fileName2)
trainData, testData = model1.splitDataset()
trainData = TrainingData(workPath, savePath, trainData)
train_loader = DataLoader(trainData, batch_size=1, shuffle=True, num_workers=10, drop_last=True)

model2 = CapacityEstimation()
criterion = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(params=model2.parameters(), lr=0.001)


def train(epoch):
    runningLoss = 0
    for batch_idx, data in enumerate(train_loader, 0):
        T, voltages, temps, currents, SOHs = data
        T, voltages, temps, currents, SOHs = \
            T.to(device), voltages.to(device), temps.to(device), currents.to(device), SOHs.to(device)
        pred = model2(T, voltages, temps, currents)
        loss = criterion(pred, SOHs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        runningLoss += loss.item()
    return runningLoss


if __name__ == '__main__':
    epoch = 0
    loss = []
    while True:
        subloss = train(epoch)
        epoch += 1
        print(epoch, subloss)
        loss += [subloss]
        if epoch % 50 == 0:
            torch.save(model2, os.path.join(savePath, f'CapacityEstimationModel{epoch}.pkl'))
            df_loss = pd.DataFrame(loss, columns=['loss'])
            df_loss.to_csv(os.path.join(savePath, f'CapacityEstimationModel_loss.csv'), index=False)

"""分别训练，再微调"""

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
    def __init__(self, feature_path, save_path, soh_Path):
        self.featurePath = feature_path
        self.savePath = save_path
        self.sohPath = soh_Path
        featureFiles_cell = [item for item in os.listdir(self.featurePath) if item.endswith('cell_Features2.csv')]
        featureFiles_temeprature = [item for item in os.listdir(self.featurePath) if
                                    item.endswith('temperature_Features2.csv')]
        sohFiles = [item for item in os.listdir(self.sohPath) if item.endswith('_Capacity_svm.csv')]
        num_files = len(featureFiles_cell)
        self.f1 = pd.DataFrame()
        self.f2 = pd.DataFrame()

        def assignmentSOH(x):
            group = x.loc[x.index.min(), 'group']
            if group in self.groups:
                x['SOH'] = f2.loc[f2['group'] == group, '平均容量'].values[0]
                return x

        for i in range(num_files):
            f1_1 = pd.read_csv(os.path.join(self.featurePath, featureFiles_cell[i]), header=[0])
            f1_2 = pd.read_csv(os.path.join(self.featurePath, featureFiles_temeprature[i]), header=[0])
            f2 = pd.read_csv(os.path.join(self.sohPath, sohFiles[i]), header=[0])
            self.groups = f2['group'].values
            f_cells = f1_1.groupby('group', group_keys=False).apply(assignmentSOH)
            f_cells['vin'] = featureFiles_cell[i][:17]
            self.f1 = pd.concat([self.f1, f_cells], axis=0, ignore_index=True)
            f_temperatures = f1_2.groupby('group', group_keys=False).apply(assignmentSOH)
            f_temperatures['vin'] = featureFiles_cell[i][:17]
            self.f2 = pd.concat([self.f2, f_temperatures], axis=0, ignore_index=True)

    def splitDataset(self):

        def split(group):
            groups = list(set(group['group'].values))
            SOHs = group.groupby('group').apply(lambda x: x.loc[x.index.min(), 'SOH']).values
            groupTrain, groupTest, SOHTrain, SOHTest = train_test_split(groups, SOHs, test_size=0.2, random_state=42)
            return groupTrain, groupTest

        f_split = self.f1.groupby('vin', group_keys=True).apply(split)
        groupTrain = {item[1]: item[0][0] for item in zip(f_split.values, f_split.index)}
        groupTest = {item[1]: item[0][1] for item in zip(f_split.values, f_split.index)}
        dataToTrain_cell = self.f1.groupby(['group', 'vin'], group_keys=False).apply(
            lambda x: x if x.loc[x.index.min(), 'group'] in groupTrain[x.loc[x.index.min(), 'vin']] else None)
        dataToTrain_temperature = self.f2.groupby(['group', 'vin'], group_keys=False).apply(
            lambda x: x if x.loc[x.index.min(), 'group'] in groupTrain[x.loc[x.index.min(), 'vin']] else None)
        dataToTest_cell = self.f1.groupby(['group', 'vin'], group_keys=False).apply(
            lambda x: x if x.loc[x.index.min(), 'group'] in groupTest[x.loc[x.index.min(), 'vin']] else None)
        dataToTest_temperature = self.f2.groupby(['group', 'vin'], group_keys=False).apply(
            lambda x: x if x.loc[x.index.min(), 'group'] in groupTest[x.loc[x.index.min(), 'vin']] else None)
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
        self.SOHs = df_cell.groupby(['group', 'vin']).apply(lambda x: x.loc[x.index.min(), 'SOH']).values
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
        self.branch1 = nn.Conv2d(in_channels=20, out_channels=10, kernel_size=3, stride=2, padding=1)
        self.branch2 = nn.Conv2d(in_channels=20, out_channels=10, kernel_size=5, stride=2, padding=2)
        self.branch3 = nn.ModuleList(
            [nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3, stride=2, padding=1),
             nn.Conv2d(in_channels=40, out_channels=10, kernel_size=5, stride=1, padding=2)])

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
        self.branch1 = nn.Conv2d(in_channels=20, out_channels=10, kernel_size=3, stride=2, padding=1)
        self.branch2 = nn.Conv2d(in_channels=20, out_channels=10, kernel_size=5, stride=2, padding=2)
        self.branch3 = nn.ModuleList(
            [nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3, stride=2, padding=1),
             nn.Conv2d(in_channels=40, out_channels=10, kernel_size=5, stride=1, padding=2)])

        self.activation = nn.ReLU()

    def forward(self, x):
        # (batch,10,55,36)
        x1 = self.activation(self.branch1(x))
        x2 = self.activation(self.branch2(x))
        x3 = self.activation(self.branch3[1](self.activation(self.branch3[0](x))))
        # (batch,10,28,18)
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
        # (batch,10,28,96)
        x = self.extend_cov(x)
        # (batch,20,28,96)
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation(out)
        # (batch,5,7,16)
        out = self.activation(self.conv3(out))
        # (batch,1,7,16)

        return out.squeeze(1)


class ExtendLayer1(nn.Module):
    def __init__(self):
        super(ExtendLayer1, self).__init__()
        self.linear = nn.Linear(in_features=8, out_features=64, bias=True)
        self.extend_cnn = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(5, 3), stride=(2, 1),
                                    padding=(2, 1))
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.linear(x))
        x = self.activation(self.extend_cnn(x))
        return x


class ExtendLayer2(nn.Module):
    def __init__(self):
        super(ExtendLayer2, self).__init__()
        self.linear = nn.Linear(in_features=5, out_features=16, bias=True)
        self.extend_cnn = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(5, 3), stride=(2, 1),
                                    padding=(2, 1))
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.linear(x))
        x = self.activation(self.extend_cnn(x))
        return x


class FeedForward(nn.Module):
    def __init__(self):
        super(FeedForward, self).__init__()

        self.dense1 = nn.Linear(in_features=20, out_features=32, bias=True)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(in_features=32, out_features=16, bias=True)
        self.dense3 = nn.Linear(in_features=20, out_features=16, bias=True)

    def forward(self, inputs):
        # (batch,40)
        output = self.dense2(self.relu(self.dense1(inputs)))
        output = self.relu(self.dense3(inputs) + output)
        # (batch,16)
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
        self.linear = nn.Linear(in_features=16, out_features=1, bias=True)
        self.activation=nn.Sigmoid()

    def forward(self, cell_features, temperature_features):
        # 归一化
        # cell_features = self.bn_cell(cell_features)
        # temperature_features = self.bn_temperature(temperature_features)

        cell_features = cell_features.unsqueeze(1)
        temperature_features = temperature_features.unsqueeze(1).repeat(1, 1, 2, 1)
        # (batch,1,96,12),(batch,1,96,6)

        temperature_features = self.extendLayer_temperature(temperature_features)
        cell_features = self.extendLayer_cell(cell_features)
        # (batch,20,48,128),(batch,20,48,64)

        cell_features = self.spp(cell_features)
        temperature_features = self.spp(temperature_features)
        # (batch,10,55,96),(batch,10,55,36)

        cell_features = self.block1(cell_features)
        temperature_features = self.block2(temperature_features)
        # (batch,10,28,64),(batch,10,28,32)

        x = torch.cat((cell_features, temperature_features), dim=-1)
        # (batch,5,28,96)
        x = self.resnet(x)
        # (batch,7,16)
        x = self.conv(x)
        # (batch,5,8)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        x = self.linear(x)
        return x


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
    featurePath = 'E:\\pycharm\\DigitalCarRace\\CapacityFeatureFile2'
    sohPath = r'E:\pycharm\DigitalCarRace\RedefineSOH'
    savePath = r'E:\pycharm\DigitalCarRace\CapacityModel3'
    model1 = TrainTestSplit(featurePath, savePath, sohPath)
    trainData_cell, trainData_temperature, testData_cell, testData_temperature = model1.splitDataset()
    trainData = TrainingData(featurePath, savePath, trainData_cell, trainData_temperature)
    train_loader = DataLoader(trainData, batch_size=100, shuffle=True, num_workers=10, drop_last=True)
    testData = TrainingData(featurePath, savePath, testData_cell, testData_temperature)
    test_loader = DataLoader(testData, batch_size=1, shuffle=False, num_workers=10, drop_last=True)


    epoch = 0
    model2 = CapacityEstimation().to(device)
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(params=model2.parameters(), lr=0.001)
    loss1 = []
    loss2 = []
    while True:
        subloss = train(epoch)
        epoch += 1
        print(epoch, subloss)
        loss1 += [subloss]
        if epoch % 10 == 0:
            torch.save(model2, os.path.join(savePath, f'CapacityEstimationModel{epoch}.pkl'))
            df_loss = pd.DataFrame(loss1, columns=['loss'])
            df_loss.to_csv(os.path.join(savePath, f'CapacityEstimationModel_loss.csv'), index=False)
            loss2 += [valid()]
            pd.DataFrame(loss2, columns=['loss']).to_csv(
                os.path.join(savePath, f'CapacityEstimationModel_valid_loss.csv'), index=False)

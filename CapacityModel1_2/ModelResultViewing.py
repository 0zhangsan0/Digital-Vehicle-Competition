import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from CapacityModel1_2.CapacityEstimation1_2 import *
from torch.utils.data import DataLoader


class TrainingData2(TrainingData):
    def __init__(self, work_path, save_path, df_cell, df_temperature):
        super(TrainingData2, self).__init__(work_path, save_path, df_cell, df_temperature)
        self.groups = df_cell.groupby('group').apply(lambda x: x.loc[x.index.min(), 'group']).values
        self.groups = torch.Tensor(self.groups).reshape(-1, 1)

    def __getitem__(self, item):
        return self.data_cell[item], self.data_temperature[item], self.SOHs[item], self.groups


workPath = 'E:\\pycharm\\DigitalCarRace\\CapacityFeatureFile'
savePath = 'E:\\pycharm\\DigitalCarRace\\CapacityFeatureFile'
fileName1 = ['LFPHC7PE0K1A07972_cell_Features2.csv', 'LFPHC7PE0K1A07972_temperature_Features2.csv']
SOHpath = r'E:\pycharm\DigitalCarRace\chargeSet\LFPHC7PE0K1A07972_redefineCapacity.csv'
model1 = TrainTestSplit(workPath, savePath, fileName1, SOHpath)
trainData_cell, trainData_temperature, testData_cell, testData_temperature = model1.splitDataset()
trainData = TrainingData2(workPath, savePath, trainData_cell, trainData_temperature)
train_loader = DataLoader(trainData, batch_size=1, shuffle=False, num_workers=10, drop_last=True)
testData = TrainingData2(workPath, savePath, testData_cell, testData_temperature)
test_loader = DataLoader(testData, batch_size=1, shuffle=False, num_workers=10, drop_last=True)


def train(epoch, loader):
    pred_all = np.array([])
    SOH_all = np.array([])
    groups_all = np.array([])
    with torch.no_grad():
        for batch_idx, data in enumerate(loader, 0):
            data_cell, data_temperature, SOHs, groups = data
            data_cell, data_temperature, SOHs = \
                data_cell.to(device), data_temperature.to(device), SOHs.to(device)
            pred = model2(data_cell, data_temperature)

            pred, SOHs = pred.to(torch.device('cpu')), SOHs.to(torch.device('cpu'))
            pred_all = np.append(pred_all, np.array(pred))
            SOH_all = np.append(SOH_all, np.array(SOHs))
            groups_all = np.append(groups_all, np.array(groups))
    return pred_all, SOH_all, groups_all


if __name__ == '__main__':
    f = pd.read_csv(r'E:\pycharm\DigitalCarRace\chargeSet\LFPHC7PE0K1A07972_redefineCapacity.csv', header=[0])
    model2 = torch.load(r'E:\pycharm\DigitalCarRace\CapacityModel1_2\CapacityEstimationModel5000.pkl')
    criterion = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(params=model2.parameters(), lr=0.001)
    pred_all, SOH_all, groups_all = train(1, train_loader)
    mileage = np.array([f.loc[f['group'] == item, '累计里程'].values[0] for item in groups_all])
    f_train = pd.DataFrame([item for item in zip(SOH_all, pred_all, mileage)], columns=['SOH', 'Pred', '累计里程'])
    f_train.sort_values(['累计里程'],inplace=True)
    pred_all, SOH_all, groups_all = train(1, test_loader)
    mileage = np.array([f.loc[f['group'] == item, '累计里程'].values[0] for item in groups_all])
    f_test = pd.DataFrame([item for item in zip(SOH_all, pred_all, mileage)], columns=['SOH', 'Pred', '累计里程'])
    f_test.sort_values(['累计里程'], inplace=True)
    plt.plot(f_train['累计里程'], f_train['Pred'], 'r*')
    plt.plot(f_train['累计里程'], f_train['SOH'], 'k*')
    plt.plot(f_test['累计里程'], f_test['Pred'], 'r*')
    plt.plot(f_test['累计里程'], f_test['SOH'], 'k*')
    plt.show()

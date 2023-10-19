"""减少特征"""

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from CapacityModel1_2_3.CapacityEstimation1_2_3 import *
from torch.utils.data import DataLoader

plt.rc("font", family='Microsoft YaHei')


class TrainingData2(Dataset):
    def __init__(self, work_path, save_path, SOH_path, file_cell, file_temperature, file_SOH):
        super(TrainingData2, self).__init__()
        f_cell = pd.read_csv(os.path.join(work_path, file_cell), header=[0])
        f_temperature = pd.read_csv(os.path.join(work_path, file_temperature), header=[0])
        self.SOHs = torch.Tensor(
            f_cell.groupby('group').apply(lambda x: x.loc[x.index.min(), 'cellCapacity']).values.astype(
                np.float32)).reshape(-1, 1)
        self.group = torch.Tensor(
            f_cell.groupby('group').apply(lambda x: x.loc[x.index.min(), 'group']).values.astype(
                np.float32)).reshape(-1, 1)
        self.mileage = torch.Tensor(
            f_cell.groupby('group').apply(lambda x: x.loc[x.index.min(), '累计里程']).values.astype(
                np.float32)).reshape(-1, 1)
        num = len(self.group)
        cells = [f'cell{i}' for i in range(1, 97)]
        probes = [f'temp{i}' for i in range(1, 49)]
        self.data_cell = torch.Tensor(f_cell[cells].values.astype(np.float32)).reshape(num, -1, 96).transpose(1, 2)
        self.data_temperature = torch.Tensor(f_temperature[probes].values.astype(np.float32)).reshape(num, -1,
                                                                                                      48).transpose(1,
                                                                                                                    2)
        self.len = num

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.data_cell[item], self.data_temperature[item], self.group[item], self.mileage[item]


workPath = 'E:\\pycharm\\DigitalCarRace\\CapacityFeatureFile2'
savePath = r'E:\pycharm\DigitalCarRace\CapacityModel3\3139'
cell_files = [item for item in os.listdir(workPath) if item.endswith('_cell_Features2.csv')]
temperature_files = [item for item in os.listdir(workPath) if item.endswith('_temperature_Features2.csv')]
SOHpath = r'E:\pycharm\DigitalCarRace\RedefineSOH'
SOH_files = [item for item in os.listdir(SOHpath) if item.endswith('_Capacity_svm.csv')]
file_nums = len(cell_files)


def train(epoch, loader):
    pred_all = np.array([])
    group_all2 = np.array([])
    mileage_all2 = np.array([])
    with torch.no_grad():
        for batch_idx, data in enumerate(loader, 0):
            data_cell, data_temperature, group, mileage = data
            data_cell, data_temperature = data_cell.to(device), data_temperature.to(device)
            pred = model2(data_cell, data_temperature)

            pred = pred.to(torch.device('cpu'))
            pred_all = np.append(pred_all, np.array(pred))
            group_all2 = np.append(group_all2, np.array(group))
            mileage_all2 = np.append(mileage_all2, np.array(mileage))
    return pred_all, group_all2, mileage_all2


if __name__ == '__main__':
    model2 = torch.load(r'E:\pycharm\DigitalCarRace\CapacityModel3\3139\CapacityEstimationModel4000.pkl')
    criterion = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(params=model2.parameters(), lr=0.001)
    num_model = 1
    for i in range(num_model, num_model + 1):
        # for i in range(file_nums):
        vin = cell_files[i][:17]
        data = TrainingData2(workPath, savePath, SOHpath, cell_files[i], temperature_files[i], SOH_files[i])
        data_loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=10, drop_last=True)
        pred_all, group_all2, mileage_all2 = train(1, data_loader)
        f1 = pd.DataFrame([item for item in zip(pred_all, group_all2, mileage_all2)],
                          columns=['Pred', 'group', '累计里程_Pred'])
        f2 = pd.read_csv(os.path.join(SOHpath, SOH_files[i]), header=[0])
        f3 = f1.merge(f2, on='group',how='outer')
        f3.sort_values(['累计里程_Pred'], inplace=True)
        for i in f3.index:
            mileage = f3.loc[i, '累计里程_Pred']
            ave_SOH = f3.loc[(f3['累计里程_Pred'] <= mileage + 2500) &
                             (f3['累计里程_Pred'] >= mileage - 2500), 'Pred'].mean()
            f3.loc[i, '平均容量_pred'] = ave_SOH
        plt.plot(f3.dropna(subset='累计里程',axis=0)['累计里程'], f3.dropna(subset='累计里程',axis=0)['安时容量'], 'y*')
        plt.plot(f3.dropna(subset='累计里程',axis=0)['累计里程'], f3.dropna(subset='累计里程',axis=0)['平均容量'], 'k*-')
        plt.plot(f3['累计里程_Pred'], f3['Pred'], 'r*')
        plt.plot(f3['累计里程_Pred'], f3['平均容量_pred'], '-*')
        plt.legend(['实际值','实际值平均', '估计值', '估计值平均'], fontsize=20, loc='upper right')
        plt.xlabel('累计里程', fontsize=20)
        plt.ylabel('容量', fontsize=20)
        plt.title(vin, fontsize=20)
        plt.show()
        f3.to_csv(os.path.join(savePath, 'predict.csv'), index=False)

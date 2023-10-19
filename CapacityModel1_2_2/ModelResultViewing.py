import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from CapacityModel1_2_2.CapacityEstimation1_2_2 import *
from torch.utils.data import DataLoader


class TrainingData2(Dataset):
    def __init__(self, work_path, save_path, SOH_path, file_cell, file_temperature, file_SOH):
        super(TrainingData2, self).__init__()
        f_cell = pd.read_csv(os.path.join(work_path, file_cell), header=[0])
        f_temperature = pd.read_csv(os.path.join(work_path, file_temperature), header=[0])
        f_SOH = pd.read_csv(os.path.join(SOH_path, file_SOH), header=[0])
        self.SOHs = torch.Tensor(f_SOH['安时容量'].values.astype(np.float32)).reshape(-1, 1)
        self.mileages = torch.Tensor(f_SOH['累计里程'].values.astype(np.float32)).reshape(-1, 1)
        groups = f_SOH['group'].values
        num = len(self.SOHs)
        cells = [f'cell{i}' for i in range(1, 97)]
        probes = [f'temp{i}' for i in range(1, 49)]
        f_cell = f_cell.groupby('group').apply(lambda x: x if x.loc[x.index.min(), 'group'] in groups else None)
        f_temperature = f_temperature.groupby('group').apply(
            lambda x: x if x.loc[x.index.min(), 'group'] in groups else None)
        self.data_cell = torch.Tensor(f_cell[cells].values.astype(np.float32)).reshape(num, -1, 96).transpose(1,2)
        self.data_temperature = torch.Tensor(f_temperature[probes].values.astype(np.float32)).reshape(num, -1, 48).transpose(1,2)
        self.len = num

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.data_cell[item], self.data_temperature[item], self.SOHs[item], self.mileages[item]


workPath = 'E:\\pycharm\\DigitalCarRace\\CapacityFeatureFile'
savePath = 'E:\\pycharm\\DigitalCarRace\\CapacityFeatureFile'
cell_files = [item for item in os.listdir(workPath) if item.endswith('_cell_Features2.csv')]
temperature_files = [item for item in os.listdir(workPath) if item.endswith('_temperature_Features2.csv')]
SOHpath = r'E:\pycharm\DigitalCarRace\RedefineSOH'
SOH_files = [item for item in os.listdir(SOHpath) if item.endswith('_Capacity_svm.csv')]
file_nums = len(cell_files)


def train(epoch, loader):
    pred_all = np.array([])
    SOH_all = np.array([])
    mileages_all = np.array([])
    with torch.no_grad():
        for batch_idx, data in enumerate(loader, 0):
            data_cell, data_temperature, SOHs, mileages = data
            data_cell, data_temperature, SOHs = \
                data_cell.to(device), data_temperature.to(device), SOHs.to(device)
            pred = model2(data_cell, data_temperature)

            pred, SOHs = pred.to(torch.device('cpu')), SOHs.to(torch.device('cpu'))
            pred_all = np.append(pred_all, np.array(pred))
            SOH_all = np.append(SOH_all, np.array(SOHs))
            mileages_all = np.append(mileages_all, np.array(mileages))
    return pred_all, SOH_all, mileages_all


if __name__ == '__main__':
    model2 = torch.load(r'E:\pycharm\DigitalCarRace\CapacityModel1_2_2\CapacityEstimationModel3000.pkl')
    criterion = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(params=model2.parameters(), lr=0.001)
    for i in range(file_nums):
        data = TrainingData2(workPath, savePath, SOHpath, cell_files[i], temperature_files[i], SOH_files[i])
        data_loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=10, drop_last=True)
        pred_all, SOH_all, mileages_all = train(1, data_loader)
        f = pd.DataFrame([item for item in zip(SOH_all, pred_all, mileages_all)],
                         columns=['安时容量', 'Pred', '累计里程'])
        f.sort_values(['累计里程'], inplace=True)
        plt.plot(f['累计里程'], f['Pred'], 'r*')
        plt.plot(f['累计里程'], f['安时容量'], 'k*')
        plt.show()

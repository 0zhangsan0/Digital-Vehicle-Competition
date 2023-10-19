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
        self.mileages_soh = torch.Tensor(
            f_cell.groupby('group').apply(lambda x: x.loc[x.index.min(), '累计里程']).values.astype(
                np.float32)).reshape(-1, 1)
        num = len(self.mileages_soh)
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
        return self.data_cell[item], self.data_temperature[item],self.mileages_soh[item]


workPath = 'E:\\pycharm\\DigitalCarRace\\CapacityFeatureFile'
savePath = 'E:\\pycharm\\DigitalCarRace\\CapacityFeatureFile'
cell_files = [item for item in os.listdir(workPath) if item.endswith('_cell_Features2.csv')]
temperature_files = [item for item in os.listdir(workPath) if item.endswith('_temperature_Features2.csv')]
SOHpath = r'E:\pycharm\DigitalCarRace\RedefineSOH'
SOH_files = [item for item in os.listdir(SOHpath) if item.endswith('_IQR.csv')]
file_nums = len(cell_files)


def train(epoch, loader):
    pred_all = np.array([])
    mileages_all2=np.array([])
    with torch.no_grad():
        for batch_idx, data in enumerate(loader, 0):
            data_cell, data_temperature, mileages_soh = data
            data_cell, data_temperature = data_cell.to(device), data_temperature.to(device)
            pred = model2(data_cell, data_temperature)

            pred= pred.to(torch.device('cpu'))
            pred_all = np.append(pred_all, np.array(pred))
            mileages_all2 = np.append(mileages_all2, np.array(mileages_soh))
    return pred_all, mileages_all2


if __name__ == '__main__':
    model2 = torch.load(r'E:\pycharm\DigitalCarRace\CapacityModel1_2_4\CapacityEstimationModel1200.pkl')
    criterion = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(params=model2.parameters(), lr=0.001)
    for i in range(file_nums):
        vin = cell_files[i][:17]
        data = TrainingData2(workPath, savePath, SOHpath, cell_files[i], temperature_files[i], SOH_files[i])
        data_loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=10, drop_last=True)
        pred_all,mileages_all2 = train(1, data_loader)
        f1 = pd.DataFrame([item for item in zip(pred_all, mileages_all2)],
                         columns=['Pred', '累计里程_soh'])
        f2 = pd.read_csv(os.path.join(SOHpath, SOH_files[i]), header=[0])
        f1.sort_values(['累计里程_soh'], inplace=True)
        plt.plot(f1['累计里程_soh'], f1['Pred'], 'r*')
        plt.plot(f2['累计里程'], f2['SOH'], 'k*-')
        plt.legend(['估计值', '实际值'], fontsize=20, loc='upper right')
        plt.xlabel('累计里程', fontsize=20)
        plt.ylabel('SOH', fontsize=20)
        plt.title(vin, fontsize=20)
        plt.show()

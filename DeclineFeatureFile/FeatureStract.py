import numpy as np
import pandas as pd
import os
import bisect
from joblib import Parallel, delayed

num_model = 0


class FeatureExtraction:
    def __init__(self, dataset_path, chargedata_path, save_path, capacity_path):
        self.dataPath = dataset_path
        self.chargePath = chargedata_path
        self.savePath = save_path
        self.capacityPath = capacity_path
        self.cells = [f'cell{i}' for i in range(1, 97)]
        self.probes = [f'temp{i}' for i in range(1, 49)]
        self.files_data = [item for item in os.listdir(self.dataPath) if item.endswith('.csv')]
        self.files_charge = [item for item in os.listdir(self.chargePath) if item.endswith('_charge.csv')]
        self.files_capacity = [item for item in os.listdir(self.capacityPath) if item.endswith('_decline.csv')]
        self.num = len(self.files_data)

    def splitDischarge(self, save=True):
        for i in range(num_model, num_model + 1):
            vin = self.files_data[i][:-4]
            print(vin)
            self.f_data = pd.read_csv(os.path.join(self.dataPath, self.files_data[i]), header=[0])
            self.f_data['数据采集时间'] = pd.to_datetime(self.f_data['数据采集时间'])
            self.f_charge = pd.read_csv(os.path.join(self.chargePath, self.files_charge[i]), header=[0])[
                ['数据采集时间', 'group']]
            self.f_charge['数据采集时间'] = pd.to_datetime(self.f_charge['数据采集时间'])
            self.f_data.drop(['group'], axis=1, inplace=True)
            self.f = self.f_data.merge(right=self.f_charge, on='数据采集时间', how='outer')
            self.f.loc[self.f['group'].isna(), 'group'] = -1
            self.f['group2'] = (self.f['group'] != self.f['group'].shift(1)).cumsum()
            self.f['deltT'] = (self.f['数据采集时间'] - self.f['数据采集时间'].shift(1)).apply(
                lambda x: x.total_seconds())
            self.f = self.f.loc[self.f['group'] == -1]
            self.f['group2'] = self.f['group2'].rank(method='dense') - 1
            self.f = self.f.loc[self.f['group2'] != 0]
            self.f = self.f.loc[self.f['group2'] != self.f['group2'].max()]
            self.f['group2'] = self.f['group2'].astype('int') - 1
            self.select_Effective_Fragment()
            self.f_capacity = pd.read_csv(os.path.join(self.capacityPath, self.files_capacity[i]), header=[0])
            self.extraction()
            if save:
                self.save(vin)

    def select_Effective_Fragment(self):
        self.f = self.f.loc[self.f['累计里程'] != 0]
        # self.f = self.f.groupby('group2', group_keys=False).apply(
        #     lambda x: None if x.shape[0] <= 1 else x)
        # self.f['group2'] = self.f['group2'].rank(method='dense') - 1
        # self.f['group2'] = self.f['group2'].astype(int)
        self.f.drop(['group'], axis=1, inplace=True)
        self.f.reset_index(drop=True, inplace=True)

    def extraction(self):

        def extractionFeatures(group):
            min_index = group.index.min()
            max_index = group.index.max()

            deltTs = group['deltT'].values
            time_rest = sum([item for item in deltTs if item > 15]) / 60
            group['deltT'] = [item if item <= 15 else 0 for item in deltTs]
            group['avg_current'] = group['总电流'].rolling(2).mean()
            group['capacityBetweenFrame'] = abs(group['deltT'] * group['avg_current'] / 3600)

            num = group.loc[min_index, 'group2']
            cells_features = [[] for _ in range(96)]
            # 描述基本情况
            end_current = group.loc[max_index, '总电流']
            avg_current = group['总电流'].mean()
            var_current = group['总电流'].var()
            discharge_time = (group.loc[max_index, '数据采集时间'] - group.loc[
                min_index, '数据采集时间']).total_seconds() / 60
            SOCs = group['SOC'].values
            start_SOC = SOCs[0]
            end_SOC = SOCs[-1]
            SOC_change = end_SOC - start_SOC
            avg_SOC = np.mean(SOCs)
            mileages = group['累计里程'].values
            start_mileage = mileages.min()
            end_mileage = mileages.max()
            change_mileage = end_mileage - start_mileage
            if discharge_time == 0:
                avg_speed = 0
            else:
                avg_speed = (end_mileage - start_mileage) / (discharge_time / 60)
            last_capacity = self.f_capacity.loc[self.f_capacity['group'] == num, '平均容量_pred'].values[0]
            cells_features = [item + [avg_current, discharge_time, start_SOC, end_SOC, SOC_change, avg_SOC,
                                      change_mileage, time_rest, avg_speed, last_capacity]
                              for item in cells_features]
            start_voltage = group[self.cells].values[0]
            end_voltage = group[self.cells].values[-1]
            cells_features = [item1 + [item2] + [item3] + [item2 - item3] for item1, item2, item3 in
                              zip(cells_features, start_voltage, end_voltage)]
            # 描述每个探针的温度情况
            temperatures = [group[item].values for item in self.probes]
            avg_temperatures = [item.mean() for item in temperatures]
            max_temperatures = [item.max() for item in temperatures]
            min_temperatures = [item.min() for item in temperatures]
            temperatures_change = [item.max() - item.min() for item in temperatures]
            if discharge_time == 0:
                temperature_rise_rates = [0 for item in temperatures_change]
            else:
                temperature_rise_rates = [item / discharge_time for item in temperatures_change]
            temperature_rest = self.f.loc[min_index, self.probes].values

            # 温度矩阵
            temperature_matrix = [item for item in zip(avg_temperatures, max_temperatures, min_temperatures,
                                                       temperatures_change, temperature_rise_rates,
                                                       temperature_rest)]

            cells_features = np.array(cells_features).T
            temperature_matrix = np.array(temperature_matrix).T
            cell_features = pd.DataFrame(cells_features, columns=self.cells)
            temperature_features = pd.DataFrame(temperature_matrix, columns=self.probes)
            cell_features['group'] = num
            temperature_features['group'] = num
            cell_features['use_state'] = -1
            temperature_features['use_state'] = -1
            return cell_features, temperature_features

        # features=self.applyParallel(self.f.groupby('group'), extractionFromCharge)
        features = self.f.groupby('group2', group_keys=True).apply(extractionFeatures)
        features = features.dropna()
        cell_features = [item[0] for item in features.values]
        temperature_features = [item[1] for item in features.values]
        cell_features = pd.concat(cell_features, ignore_index=True)
        temperature_features = pd.concat(temperature_features, ignore_index=True)
        cell_features.dropna(axis=0, inplace=True)
        temperature_features.dropna(axis=0, inplace=True)
        self.cell_features = cell_features
        self.temperature_features = temperature_features

    def save(self, vin):
        self.cell_features.to_csv(os.path.join(savePath, vin + '_cell_Features.csv'), index=False)
        self.temperature_features.to_csv(os.path.join(savePath, vin + '_temperature_Features.csv'), index=False)


if __name__ == '__main__':
    dataPath = r'E:\pycharm\DigitalCarRace\dataSet'
    chargedataPath = r'E:\pycharm\DigitalCarRace\chargeSet'
    savePath = r'E:\pycharm\DigitalCarRace\DeclineFeatureFile'
    capacityPath = r'E:\pycharm\DigitalCarRace\DeclineCapacity'
    model = FeatureExtraction(dataset_path=dataPath, chargedata_path=chargedataPath, save_path=savePath,
                              capacity_path=capacityPath)
    model.splitDischarge()

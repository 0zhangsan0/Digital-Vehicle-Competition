from datetime import timedelta

import numpy as np
import pandas as pd
import os
import bisect
from joblib import Parallel, delayed


class FeatureExtraction:
    def __init__(self, work_path, save_path, file_name):
        self.workPath = work_path
        self.savePath = save_path
        self.fileName = file_name
        self.cells = [f'cell{i}' for i in range(1, 97)]
        self.probes = [f'temp{i}' for i in range(1, 49)]
        f = pd.read_csv(os.path.join(self.workPath, self.fileName), header=[0])
        f['数据采集时间'] = pd.to_datetime(f['数据采集时间'])
        self.f = f

    def select_Effective_Fragment(self):
        self.f = self.f.groupby('group', group_keys=False).apply(
            lambda x: None if x.shape[0] <= 1 else x)
        self.f['group'] = (self.f['state'] != self.f['state'].shift(1)).cumsum()
        self.f['group'] = self.f['group'].rank(method='dense') - 1
        self.f['group'] = self.f['group'].astype(int)
        self.f.reset_index(drop=True, inplace=True)

    def extraction(self):

        def extractionFeatures(group):
            min_index = group.index.min()
            max_index = group.index.max()
            group['deltT'] = (group['数据采集时间'] - group['数据采集时间'].shift(1)).apply(lambda x: x.total_seconds())
            group['avg_current'] = group['总电流'].rolling(2).mean()
            group['capacityBetweenFrame'] = abs(group['deltT'] * group['avg_current'] / 3600)
            if group.loc[min_index, 'state'] == 1:
                num = group.loc[min_index, 'group']
                cells_features = [[] for _ in range(96)]
                # 描述基本情况
                end_current = group.loc[max_index, '总电流']
                avg_current = group['总电流'].mean()
                var_current = group['总电流'].var()
                charge_capacity = group['capacityBetweenFrame'].sum()
                charge_time = (group.loc[max_index, '数据采集时间'] - group.loc[
                    min_index, '数据采集时间']).total_seconds() / 60
                SOCs = group['SOC'].values
                start_SOC = SOCs[0]
                end_SOC = SOCs[-1]
                SOC_change = start_SOC - end_SOC
                avg_SOC = np.mean(SOCs)
                mileage = group['累计里程'].values[0]
                cells_features = [item + [end_current, avg_current, var_current, charge_capacity, charge_time,
                                          start_SOC, end_SOC, SOC_change, avg_SOC] + [0, 0, 0]
                                  for item in cells_features]
                start_voltage = group[self.cells].values[0]
                end_voltage = group[self.cells].values[-1]

                # 单体矩阵
                cells_features = [item1 + [item2] + [item3] + [item3 - item2] for item1, item2, item3 in
                                  zip(cells_features, start_voltage, end_voltage)]

                # 描述每个探针的温度情况
                temperatures = [group[item].values for item in self.probes]
                avg_temperatures = [item.mean() for item in temperatures]
                max_temperatures = [item.max() for item in temperatures]
                min_temperatures = [item.min() for item in temperatures]
                temperatures_change = [item.max() - item.min() for item in temperatures]
                temperature_rise_rates = [item / charge_time for item in temperatures_change]
                temperature_vars = [np.var(item) for item in temperatures]

                # 温度矩阵
                temperature_matrix = [list(item) + [0] for item in zip(avg_temperatures, max_temperatures,
                                                                       min_temperatures, temperatures_change,
                                                                       temperature_rise_rates, temperature_vars)]

                cells_features = np.array(cells_features).T
                temperature_matrix = np.array(temperature_matrix).T
                cell_features = pd.DataFrame(cells_features, columns=self.cells)
                temperature_features = pd.DataFrame(temperature_matrix, columns=self.probes)
                cell_features['group'] = num
                temperature_features['group'] = num
                cell_features['use_state'] = 1
                temperature_features['use_state'] = 1
            else:
                if min_index == 0 or max_index == self.f.shape[0] - 1:
                    cell_features = None
                    temperature_features = None
                else:
                    num = group.loc[min_index, 'group']
                    cells_features = [[] for _ in range(96)]
                    # 描述基本情况
                    end_current = group.loc[max_index, '总电流']
                    avg_current = group['总电流'].mean()
                    var_current = group['总电流'].var()
                    discharge_capacity = group['capacityBetweenFrame'].sum()
                    discharge_time = (group.loc[max_index, '数据采集时间'] - group.loc[
                        min_index, '数据采集时间']).total_seconds() / 60
                    SOCs = group['SOC'].values
                    start_SOC = SOCs[0]
                    end_SOC = SOCs[-1]
                    SOC_change = start_SOC - end_SOC
                    avg_SOC = np.mean(SOCs)
                    time_before_start = (self.f.loc[min_index, '数据采集时间'] - self.f.loc[
                        min_index - 1, '数据采集时间']).total_seconds() / 60
                    time_after_end = (self.f.loc[max_index + 1, '数据采集时间'] - self.f.loc[
                        max_index, '数据采集时间']).total_seconds() / 60
                    mileages = group['累计里程'].values
                    start_mileage = mileages.min()
                    end_mileage = mileages.max()
                    change_mileage = end_mileage - start_mileage
                    avg_speed = (end_mileage - start_mileage) / (discharge_time / 60)
                    cells_features = [item + [end_current, avg_current, var_current, discharge_capacity, discharge_time,
                                              start_SOC, end_SOC, SOC_change, avg_SOC, change_mileage,
                                              time_before_start, avg_speed]
                                      for item in cells_features]
                    start_voltage = group[self.cells].values[0]
                    end_voltage = group[self.cells].values[-1]
                    cells_features = [item1 + [item2] + [item3] + [item3 - item2] for item1, item2, item3 in
                                      zip(cells_features, start_voltage, end_voltage)]
                    # 描述每个探针的温度情况
                    temperatures = [group[item].values for item in self.probes]
                    avg_temperatures = [item.mean() for item in temperatures]
                    max_temperatures = [item.max() for item in temperatures]
                    min_temperatures = [item.min() for item in temperatures]
                    temperatures_change = [item.max() - item.min() for item in temperatures]
                    temperature_rise_rates = [item / discharge_time for item in temperatures_change]
                    temperature_vars = [np.var(item) for item in temperatures]
                    temperature_before_start = self.f.loc[min_index, self.probes].values
                    temperature_after_end = self.f.loc[max_index + 1, self.probes].values

                    # 温度矩阵
                    temperature_matrix = [item for item in zip(avg_temperatures, max_temperatures, min_temperatures,
                                                               temperatures_change, temperature_rise_rates,
                                                               temperature_vars, temperature_before_start)]

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
        features = self.f.groupby('group', group_keys=True).apply(extractionFeatures)
        features = features.dropna()
        cell_features = [item[0] for item in features.values]
        temperature_features = [item[1] for item in features.values]
        cell_features = pd.concat(cell_features, ignore_index=True)
        temperature_features = pd.concat(temperature_features, ignore_index=True)
        cell_features.dropna(axis=0, inplace=True)
        temperature_features.dropna(axis=0, inplace=True)
        return cell_features, temperature_features

    def save(self):
        self.select_Effective_Fragment()
        cell_features, temperature_features = self.extraction()
        cell_features.to_csv(os.path.join(savePath, self.fileName[:-4] + 'cell_Features.csv'), index=False)
        temperature_features.to_csv(os.path.join(savePath, self.fileName[:-4] + 'temperature_Features.csv'),
                                    index=False)

    @staticmethod
    def applyParallel(dfgrouped, func):
        res = Parallel(n_jobs=6)(delayed(func)(group) for index, group in dfgrouped)

        return pd.concat(res)


if __name__ == '__main__':
    workPath = r'E:\pycharm\DigitalCarRace\dataSet'
    savePath = r'E:\pycharm\DigitalCarRace\SOHFeatureFile'
    files = os.listdir(workPath)
    for file in files:
        model = FeatureExtraction(workPath, savePath, file)
        model.save()

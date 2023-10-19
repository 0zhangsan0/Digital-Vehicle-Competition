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

    def extraction(self, isCharge=True):

        def extractionFromCharge(group):
            min_index = group.index.min()
            max_index = group.index.max()
            startVoltages = group.loc[min_index, self.cells].values
            endVoltages = group.loc[max_index, self.cells].values
            maxStartVoltage = startVoltages.max()
            minEndVoltage = endVoltages.min()
            if maxStartVoltage > 4.0 or minEndVoltage < 4.12:
                return None
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
            cells_features = [item + [end_current, avg_current, var_current, charge_capacity, charge_time,
                                      start_SOC, end_SOC, SOC_change, avg_SOC]
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
            temperature_rise_rates = [item / charge_time for item in temperatures_change]
            temperature_vars = [np.var(item) for item in temperatures]

            # 温度矩阵
            temperature_matrix = [item for item in zip(avg_temperatures, max_temperatures, min_temperatures,
                                                       temperatures_change, temperature_rise_rates, temperature_vars)]

            # 描述增量容量情况\
            voltages = group[self.cells].values.T
            capacity_between_frame = group['capacityBetweenFrame'].values
            for i in range(96):
                cell_voltages = list(voltages[i])
                need_voltages = [i / 1000 for i in
                                 range(int(cell_voltages[0] * 1000), int(cell_voltages[-1] * 1000) + 1, 15)]
                voltageIndex = [cell_voltages.index(voltage) for voltage in need_voltages if voltage in cell_voltages]
                voltageIndex = voltageIndex[1:]
                IC_Values = [
                    sum(capacity_between_frame[index1:index2 + 1]) / (cell_voltages[index2] - cell_voltages[index1])
                    for index1, index2 in zip(voltageIndex, voltageIndex[1:])]
                cell_voltages = np.array(cell_voltages)
                valuePairs = [i for i in zip(cell_voltages[voltageIndex[1:]], IC_Values)]
                limit_index = bisect.bisect_right([i[0] for i in valuePairs], 4.0)
                IC_slopes = [(IC2[1] - IC1[1]) / (IC2[0] - IC1[0]) for IC1, IC2 in
                             zip(valuePairs[limit_index - 1:], valuePairs[limit_index:])]
                IC_slopes.insert(0, 0)
                voltages_in_value_pairs = [item[1] for item in valuePairs[limit_index:]]
                max_IC = max(voltages_in_value_pairs)
                max_IC_location = voltages_in_value_pairs[voltages_in_value_pairs.index(max(voltages_in_value_pairs))]
                max_slope = max(IC_slopes)
                min_slope = min(IC_slopes)
                cells_features[i].extend([max_IC, max_IC_location, max_slope, min_slope])
            cells_features = np.array(cells_features).T
            temperature_matrix = np.array(temperature_matrix).T
            cell_features = pd.DataFrame(cells_features, columns=self.cells)
            temperature_features = pd.DataFrame(temperature_matrix, columns=self.probes)
            cell_features['group'] = num
            temperature_features['group'] = num
            cell_features['use_state'] = 1
            temperature_features['use_state'] = 1
            return cell_features, temperature_features

        def extractionFromDischarge(group):
            min_index=group.index.min()
            max_index=group.index.max()
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
            before_rest_time = group.loc[min_index, 'before_rest']
            after_rest_time = group.loc[max_index, 'after_rest']
            mileages = group['累计里程'].values
            start_mileage = mileages.min()
            end_mileage = mileages.max()
            avg_speed = (end_mileage - start_mileage) / (discharge_time / 3600)
            cells_features = [item + [end_current, avg_current, var_current, discharge_capacity, discharge_time,
                                      start_SOC, end_SOC, SOC_change, avg_SOC, before_rest_time, after_rest_time,
                                      start_mileage, end_mileage, avg_speed]
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

            # 温度矩阵
            temperature_matrix = [item for item in zip(avg_temperatures, max_temperatures, min_temperatures,
                                                       temperatures_change, temperature_rise_rates, temperature_vars)]

            cells_features = np.array(cells_features).T
            temperature_matrix = np.array(temperature_matrix).T
            cell_features = pd.DataFrame(cells_features, columns=self.cells)
            temperature_features = pd.DataFrame(temperature_matrix, columns=self.probes)
            cell_features['group'] = num
            temperature_features['group'] = num
            cell_features['use_state'] = -1
            temperature_features['use_state'] = -1
            return cell_features, temperature_features

        if isCharge:
            # features=self.applyParallel(self.f.groupby('group'), extractionFromCharge)
            features = self.f.groupby('group', group_keys=True).apply(extractionFromCharge)
            features = features.dropna()
            cell_features = [item[0] for item in features.values]
            temperature_features = [item[1] for item in features.values]
            cell_features = pd.concat(cell_features, ignore_index=True)
            temperature_features = pd.concat(temperature_features, ignore_index=True)
        else:
            # cell_features, temperature_features = self.applyParallel(self.f.groupby('group'), extractionFromDischarge)
            features = self.f.groupby('group', group_keys=True).apply(extractionFromDischarge)
            features = features.dropna()
            cell_features = [item[0] for item in features.values]
            temperature_features = [item[1] for item in features.values]
            cell_features = pd.concat(cell_features, ignore_index=True)
            temperature_features = pd.concat(temperature_features, ignore_index=True)
        return cell_features, temperature_features

    def save(self, isCharge=True):
        if isCharge:
            cell_features, temperature_features = self.extraction(isCharge)
            cell_features.to_csv(os.path.join(savePath, self.fileName[:-10] + 'cell_Features1.csv'), index=False)
            temperature_features.to_csv(os.path.join(savePath, self.fileName[:-10] + 'temperature_Features1.csv'),
                                        index=False)
        else:
            cell_features, temperature_features = self.extraction(isCharge)
            cell_features.to_csv(os.path.join(savePath, self.fileName[:-10] + 'cell_Features2.csv'), index=False)
            temperature_features.to_csv(os.path.join(savePath, self.fileName[:-10] + 'temperature_Features2.csv'),
                                        index=False)

    @staticmethod
    def applyParallel(dfgrouped, func):
        res = Parallel(n_jobs=6)(delayed(func)(group) for index, group in dfgrouped)

        return pd.concat(res)


if __name__ == '__main__':
    workPath = r'E:\pycharm\DigitalCarRace\chargeSet'
    savePath = r'E:\pycharm\DigitalCarRace\CapacityFeatureFile'
    fileName = 'LFPHC7PE0K1A07972_charge.csv'
    model = FeatureExtraction(workPath, savePath, fileName)
    model.save(isCharge=True)

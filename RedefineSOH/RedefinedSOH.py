"""
重定义SOH
"""
import math
import os
from sko.PSO import PSO
import pandas as pd
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import random
import numpy as np
from sklearn import svm

plt.rc("font", family='YouYuan')
random.seed(42)


class RedefineSOH:
    def __init__(self, work_path, save_path, file_name):
        self.workPath = work_path
        self.savePath = save_path
        self.fileName = file_name
        readPath = os.path.join(self.workPath, self.fileName)
        self.chargePartial = pd.read_csv(readPath, header=[0])
        self.chargePartial['数据采集时间'] = pd.to_datetime(self.chargePartial['数据采集时间'])
        self.cells = [f'cell{i}' for i in range(1, 97)]
        minMileage = self.chargePartial['累计里程'].min()
        maxMileage = self.chargePartial['累计里程'].max()
        lowerLimit = math.floor(minMileage / 10000)
        upperLimit = math.ceil(maxMileage / 10000)
        self.mileageList = [10000 * i for i in range(lowerLimit, upperLimit + 1)]

    def SOHCalculation(self, sVoltage=4.00, eVoltage=4.12):
        def calculate(group):
            group.reset_index(drop=True, inplace=True)
            startVoltages = group.loc[0, self.cells].values
            endVoltages = group.loc[group.shape[0] - 1, self.cells].values
            maxStartVoltage = startVoltages.max()
            minEndVoltage = endVoltages.min()
            chargeCapacityOfCell = []
            if maxStartVoltage <= sVoltage and minEndVoltage >= eVoltage:
                for cell in self.cells:
                    voltages = list(group[cell].values)
                    try:
                        startIdx = np.argmin(np.abs(np.array(voltages) - sVoltage))
                        endIdx = np.argmin(np.abs(np.array(voltages) - eVoltage))
                    except ValueError as e:
                        continue
                    else:
                        chargeCapacityOfCell.append(group.loc[startIdx:endIdx, 'capacityBetweenFrame'].sum())

            if len(chargeCapacityOfCell) != 0:
                chargeCapacityOfPartial = sum(chargeCapacityOfCell) / len(chargeCapacityOfCell)
            else:
                chargeCapacityOfPartial = None
            res = pd.DataFrame({'group': [group['group'].values[0]],
                                '数据采集时间': [group.loc[0, '数据采集时间']],
                                '部分区域容量': [chargeCapacityOfPartial],
                                '累计里程': [group.loc[0, '累计里程']],
                                '安时容量': [group.loc[0, 'cellCapacity']]})
            return res

        # self.SOHLabel = self.applyParallel(self.chargePartial.groupby('group', group_keys=False), calculate)
        self.SOHLabel = self.chargePartial.groupby('group', group_keys=True).apply(calculate)
        drop_index = np.where(np.isinf(self.SOHLabel['安时容量'].values))[0]
        self.SOHLabel.drop(drop_index, axis=0, inplace=True)
        self.SOHLabel.dropna(axis=0, subset=['部分区域容量'], inplace=True)

        result_IQR = []
        data_capacity = self.SOHLabel['部分区域容量'].values
        total_Q1 = np.percentile(data_capacity, 25)
        total_Q3 = np.percentile(data_capacity, 75)
        total_IQR = total_Q3 - total_Q1
        total_limit_lower = total_Q1 - 0.5 * total_IQR
        total_limit_upper = total_Q3 + 0.5 * total_IQR
        for mileage1, mileage2 in zip(self.mileageList, self.mileageList[1:]):
            data_capacity = self.SOHLabel.loc[
                (self.SOHLabel['累计里程'] >= mileage1) & (self.SOHLabel['累计里程'] < mileage2), '部分区域容量'].values
            if len(data_capacity) <= 5:
                result_IQR.extend(map(lambda x: x if total_limit_upper >= x >= total_limit_lower else 0, data_capacity))
            else:
                Q1 = np.percentile(data_capacity, 25)
                Q3 = np.percentile(data_capacity, 75)
                IQR = Q3 - Q1
                limit_lower = Q1 - 0.5 * IQR
                limit_upper = Q3 + 0.5 * IQR
                result_IQR.extend(map(lambda x: x if limit_upper >= x >= limit_lower else 0, data_capacity))
        self.SOHLabel['部分区域容量'] = result_IQR

        result_IQR = []
        data_capacity = self.SOHLabel['安时容量'].values
        total_Q1 = np.percentile(data_capacity, 25)
        total_Q3 = np.percentile(data_capacity, 75)
        total_IQR = total_Q3 - total_Q1
        total_limit_lower = total_Q1 - 0.5 * total_IQR
        total_limit_upper = total_Q3 + 0.5 * total_IQR
        for mileage1, mileage2 in zip(self.mileageList, self.mileageList[1:]):
            data_capacity = self.SOHLabel.loc[
                (self.SOHLabel['累计里程'] >= mileage1) & (self.SOHLabel['累计里程'] < mileage2), '安时容量'].values
            if len(data_capacity) <= 5:
                result_IQR.extend(map(lambda x: x if total_limit_upper >= x >= total_limit_lower else 0, data_capacity))
            else:
                Q1 = np.percentile(data_capacity, 25)
                Q3 = np.percentile(data_capacity, 75)
                IQR = Q3 - Q1
                limit_lower = Q1 - 0.5 * IQR
                limit_upper = Q3 + 0.5 * IQR
                result_IQR.extend(map(lambda x: x if limit_upper >= x >= limit_lower else 0, data_capacity))
        self.SOHLabel['安时容量'] = result_IQR

        self.SOHLabel['temp'] = self.SOHLabel['部分区域容量'] * self.SOHLabel['安时容量']
        self.SOHLabel = self.SOHLabel.loc[self.SOHLabel['temp'] != 0]
        self.SOHLabel.drop(['temp'], axis=1, inplace=True)
        self.Cap1 = self.SOHLabel['安时容量'].max()
        self.Cap2 = self.SOHLabel['部分区域容量'].max()
        self.SOHLabel['SOH1'] = self.SOHLabel['安时容量'] / self.Cap1
        self.SOHLabel['SOH2'] = self.SOHLabel['部分区域容量'] / self.Cap2

    def minimizeVarOfSOH3(self, params):
        # 0.87336437
        weight = [params, 1 - params]
        self.SOHLabel['SOH'] = (weight[0] * self.SOHLabel['SOH1'] + weight[1] * self.SOHLabel['SOH2'])
        # 计算各窗口内的方差之和
        varOfSOH3 = 0
        for mileage1, mileage2 in zip(self.mileageList, self.mileageList[1:]):
            subdf = self.SOHLabel.loc[(self.SOHLabel['累计里程'] >= mileage1) & (self.SOHLabel['累计里程'] < mileage2)]
            if subdf.shape[0] == 0 or subdf.shape[0] == 1:
                continue
            else:
                varOfSOH3 += subdf['SOH'].values.var()
        return varOfSOH3

    def optimizeParms(self):
        pso = PSO(func=self.minimizeVarOfSOH3, n_dim=1, pop=20, lb=[0], ub=[1], max_iter=10,
                  verbose=True)
        pso.run()
        self.SOHLabel['加权容量'] = self.SOHLabel['SOH'] * (self.SOHLabel['安时容量'] + self.SOHLabel['部分区域容量'])
        return pso.gbest_x

    def move_AVE(self):
        for i in self.SOHLabel.index:
            mileage = self.SOHLabel.loc[i,'累计里程']
            ave_SOH = self.SOHLabel.loc[(self.SOHLabel['累计里程'] <= mileage + 2500) &
                                        (self.SOHLabel['累计里程'] >= mileage - 2500), '加权容量'].mean()
            self.SOHLabel.loc[i,'平均容量']=ave_SOH

    def plotSOH(self):
        plt.plot(self.SOHLabel['累计里程'], self.SOHLabel['SOH1'], 'r-*')
        plt.plot(self.SOHLabel['累计里程'], self.SOHLabel['SOH2'], 'b-*')
        plt.plot(self.SOHLabel['累计里程'], self.SOHLabel['SOH'], 'k-*')
        plt.xlabel('累计里程', fontsize=20)
        plt.ylabel('SOH', fontsize=20)
        plt.legend(['安时容量', '区域容量', '加权容量'], loc='upper right', fontsize=20)
        plt.show()
        plt.plot(self.SOHLabel['累计里程'], self.SOHLabel['平均容量'], 'k-*')
        plt.show()

    def save(self):
        path = os.path.join(self.savePath, self.fileName[:-10] + 'redefineCapacity_IQR.csv')
        self.SOHLabel.to_csv(path, index=False)

    @staticmethod
    def applyParallel(dfgrouped, func):
        res = Parallel(n_jobs=6)(delayed(func)(index, group) for index, group in dfgrouped)
        return pd.concat(res)


if __name__ == '__main__':
    workPath = r'E:\pycharm\DigitalCarRace\chargeSet'
    savePath = r'E:\pycharm\DigitalCarRace\RedefineSOH'
    files = [file for file in os.listdir(r'E:\pycharm\DigitalCarRace\chargeSet') if file.endswith('charge.csv')]
    for file in files:
        model = RedefineSOH(workPath, savePath, file)
        model.SOHCalculation()
        model.optimizeParms()
        model.move_AVE()
        model.plotSOH()
        model.save()

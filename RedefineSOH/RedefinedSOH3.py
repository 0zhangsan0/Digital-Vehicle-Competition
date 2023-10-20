"""
箱型图方法
移动平均
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

    def SOHCalculation(self):
        def calculate(group):
            group.reset_index(drop=True, inplace=True)
            res = pd.DataFrame({'group': [group['group'].values[0]],
                                '数据采集时间': [group.loc[0, '数据采集时间']],
                                '累计里程': [group.loc[0, '累计里程']],
                                '安时容量': [group.loc[0, 'cellCapacity']]})
            return res

        self.SOHLabel = self.chargePartial.groupby('group', group_keys=True).apply(calculate)
        drop_index = np.where(np.isinf(self.SOHLabel['安时容量'].values))[0]
        self.SOHLabel.drop(drop_index, axis=0, inplace=True)
        self.SOHLabel.reset_index(drop=True, inplace=True)
        min_mileage = self.SOHLabel['累计里程'].min()
        max_mileage = self.SOHLabel['累计里程'].max()
        mileage_list = [item * 10000 for item in range(int(min_mileage // 10000), int(max_mileage // 10000 + 2))]
        data_capacity = self.SOHLabel['安时容量'].values
        total_Q1 = np.percentile(data_capacity, 25)
        total_Q3 = np.percentile(data_capacity, 75)
        total_IQR = total_Q3 - total_Q1
        total_limit_lower = total_Q1 - 0.5 * total_IQR
        total_limit_upper = total_Q3 + 0.5 * total_IQR
        result_IQR = []
        for mileage1, mileage2 in zip(mileage_list, mileage_list[1:]):
            data_capacity = self.SOHLabel.loc[
                (self.SOHLabel['累计里程'] >= mileage1) & (self.SOHLabel['累计里程'] < mileage2), '安时容量'].values
            if len(data_capacity) <= 5:
                result_IQR.extend(map(lambda x: 1 if total_limit_upper >= x >= total_limit_lower else 0, data_capacity))
            else:
                Q1 = np.percentile(data_capacity, 25)
                Q3 = np.percentile(data_capacity, 75)
                IQR = Q3 - Q1
                limit_lower = Q1 - 0.5 * IQR
                limit_upper = Q3 + 0.5 * IQR
                result_IQR.extend(map(lambda x: 1 if limit_upper >= x >= limit_lower else 0, data_capacity))
        drop_index = np.where(np.array(result_IQR) == 0)[0]
        self.SOHLabel.drop(drop_index, axis=0, inplace=True)

    def move_AVE(self):
        for i in self.SOHLabel.index:
            mileage = self.SOHLabel.loc[i, '累计里程']
            ave_SOH = self.SOHLabel.loc[(self.SOHLabel['累计里程'] <= mileage + 2500) &
                                        (self.SOHLabel['累计里程'] >= mileage - 2500), '安时容量'].mean()
            self.SOHLabel.loc[i, '平均容量'] = ave_SOH

    def plotSOH(self):
        plt.plot(self.SOHLabel['累计里程'], self.SOHLabel['安时容量'], 'r-*')
        plt.plot(self.SOHLabel['累计里程'], self.SOHLabel['平均容量'], 'k-*')
        plt.xlabel('累计里程', fontsize=20)
        plt.ylabel('SOH', fontsize=20)
        plt.legend(['安时容量', '平均容量'], fontsize=20, loc='upper right')
        plt.show()

    def save(self):
        path = os.path.join(self.savePath, self.fileName[:-10] + 'Capacity_svm.csv')
        self.SOHLabel.to_csv(path, index=False)


if __name__ == '__main__':
    workPath = r'E:\pycharm\DigitalCarRace\chargeSet'
    savePath = r'E:\pycharm\DigitalCarRace\RedefineSOH'
    files = [file for file in os.listdir(r'E:\pycharm\DigitalCarRace\chargeSet') if file.endswith('charge.csv')]
    for file in files:
        print(file)
        model = RedefineSOH(workPath, savePath, file)
        model.SOHCalculation()
        model.move_AVE()
        model.plotSOH()
        model.save()

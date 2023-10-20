"""
重定义SOH
SVM方法
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
        df_for_svm = self.SOHLabel[['数据采集时间', '安时容量']]
        data = df_for_svm['安时容量'].values.reshape(-1, 1)
        clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
        clf.fit(data)
        df_for_svm.insert(2, 'result_svm_Ah', clf.predict(data))
        self.SOHLabel = pd.merge(left=self.SOHLabel, right=df_for_svm[['数据采集时间', 'result_svm_Ah']],
                                 on='数据采集时间',
                                 how='left')
        self.SOHLabel['安时容量'] = [item[1] if item[0] == 1 else None
                                     for item in zip(self.SOHLabel['result_svm_Ah'], self.SOHLabel['安时容量'])]
        self.SOHLabel.dropna(subset=['安时容量'], axis=0, inplace=True)
        self.SOHLabel.drop('result_svm_Ah', axis=1, inplace=True)

    def plotSOH(self):
        plt.plot(self.SOHLabel['累计里程'], self.SOHLabel['安时容量'], 'r-*')
        plt.xlabel('累计里程', fontsize=20)
        plt.ylabel('SOH', fontsize=20)
        plt.show()

    def save(self):
        path = os.path.join(self.savePath, self.fileName[:-10] + 'Capacity_svm.csv')
        self.SOHLabel.to_csv(path, index=False)


if __name__ == '__main__':
    workPath = r'E:\pycharm\DigitalCarRace\chargeSet'
    savePath = r'E:\pycharm\DigitalCarRace\RedefineSOH'
    files = [file for file in os.listdir(r'E:\pycharm\DigitalCarRace\chargeSet') if file.endswith('charge.csv')]
    for file in files:
        model = RedefineSOH(workPath, savePath, file)
        model.SOHCalculation()
        # model.plotSOH()
        model.save()

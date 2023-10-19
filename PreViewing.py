"""
数据清洗，数据集分离
"""

import os
from joblib import Parallel, delayed
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


class SelectUsefulCol:
    def __init__(self, read_path, work_path, file_name):
        self.readPath = read_path
        self.workPath = work_path
        self.fileName = file_name

    def read(self):
        f = pd.read_csv(os.path.join(self.readPath, self.fileName), header=[0])
        f = f['数据采集时间,车辆状态,充电状态,运行模式,车速,累计里程,总电压,总电流,SOC,单体电池(可充电储能子系统)总数,' \
              '单体电池包总数(电压上传),单体电池电压值列表,' \
              '单体电池(可充电储能子系统)温度探针总数,单体电池包总数(温度上传),单体电池温度值列表'.split(',')]
        f['numOfCells'] = f['单体电池(可充电储能子系统)总数'].apply(lambda x: int(x[x.index(':') + 1:]))
        f['numOfProbes'] = f['单体电池(可充电储能子系统)温度探针总数'].apply(lambda x: int(x[x.index(':') + 1:]))

        def stringSplit(x):
            idx = x.index(':')
            datas = x[idx + 1:].split('_')
            return [float(data) for data in datas]

        self.cells = [f'cell{i}' for i in range(1, 1 + f.loc[0, 'numOfCells'])]
        temp = [stringSplit(item) for item in f['单体电池电压值列表'].values]
        temp = pd.DataFrame(temp, columns=self.cells)
        f = pd.concat([f, temp], axis=1)
        self.probes = [f'temp{i}' for i in range(1, 1 + f.loc[0, 'numOfProbes'])]
        temp = [stringSplit(item) for item in f['单体电池温度值列表'].values]
        temp = pd.DataFrame(temp, columns=self.probes)
        f = pd.concat([f, temp], axis=1)
        print(f[['单体电池包总数(电压上传)', '单体电池包总数(温度上传)']].value_counts())
        f.drop(['单体电池(可充电储能子系统)总数', '单体电池电压值列表', '单体电池(可充电储能子系统)温度探针总数',
                '单体电池温度值列表', '单体电池包总数(电压上传)', '单体电池包总数(温度上传)'], axis=1,
               inplace=True)
        print(f.isna().value_counts())
        self.f = f
        self.f['数据采集时间'] = pd.to_datetime(self.f['数据采集时间'])
        # 排序并删除时间空值、里程异常值、充电状态异常值
        self.f.sort_values(['数据采集时间'], inplace=True)
        self.f.drop_duplicates(subset=['数据采集时间'], keep='first',inplace=True)
        temp = zip(self.f.index, self.f['累计里程'].values)
        temp = [item[0] for item in temp if item[1] > 500000]
        self.f.drop(temp, axis=0, inplace=True)
        temp = zip(self.f.index, self.f['充电状态'].values)
        temp = [item[0] for item in temp if item[1] > 4]
        self.f.drop(temp, axis=0, inplace=True)
        self.f.reset_index(drop=True, inplace=True)
        self.splitChargeAndDisCharge()

    def splitChargeAndDisCharge(self):
        self.f['state'] = self.f['充电状态'].apply(lambda x: 1 if x == 1 else (0 if x in [2, 3, 4] else 2))
        self.f['group'] = (self.f['state'] != self.f['state'].shift(1)).cumsum()
        self.chargePartial = self.f.loc[self.f['state'] == 1]
        self.dischargePartial = self.f.loc[self.f['state'] == 0]
        # 仅保留记录数量大于100条的充电片段
        self.chargePartial = self.chargePartial.groupby(['group'], group_keys=False).apply(
            lambda x: x if x.shape[0] >= 100 else None)
        self.chargePartial['group'] = self.chargePartial['group'].rank(method='dense') - 1
        self.chargePartial['group'] = self.chargePartial['group'].astype(int)
        self.dischargePartial['group'] = self.dischargePartial['group'].rank(method='dense') - 1
        self.dischargePartial['group'] = self.dischargePartial['group'].astype(int)

    def calculateCapacity(self):
        def calculateFunc(group):
            judgeMileage = group['累计里程'].value_counts()
            if judgeMileage.shape[0] > 1:
                print(judgeMileage)
            group['deltT'] = (group['数据采集时间'] - group['数据采集时间'].shift(1)).apply(lambda x: x.total_seconds())
            group['AVEcurrent'] = group['总电流'].rolling(2).mean()
            group['capacityBetweenFrame'] = -group['deltT'] * group['AVEcurrent'] / 3600
            minSOC = group.loc[group.index.min(), 'SOC']
            maxSOC = group.loc[group.index.max(), 'SOC']
            changeSOC = maxSOC - minSOC
            if changeSOC == 0:
                return None
            group['cellCapacity'] = 100 * group['capacityBetweenFrame'].sum() / changeSOC
            # group['chargeSpeed'] = 'quick' if group['总电流'].mean() < -30 else 'slow'
            return group

        self.chargePartial = self.chargePartial.groupby(['group']).apply(calculateFunc)
        # self.chargePartial = self.applyParallel(self.chargePartial.groupby(['group'], group_keys=True), calculateFunc)
        # self.quickChargePartial = self.chargePartial.loc[self.chargePartial['chargeSpeed'] == 'quick']
        # self.slowChargePartial = self.chargePartial.loc[self.chargePartial['chargeSpeed'] == 'slow']

    def save(self):
        self.f.to_csv(os.path.join(self.workPath, 'dataSet', self.fileName), index=False)
        self.chargePartial.to_csv(os.path.join(self.workPath, 'chargeSet', self.fileName[:-4] + '_charge.csv'),
                                  index=False)
        self.dischargePartial.to_csv(os.path.join(self.workPath, 'dischargeSet',
                                                  self.fileName[:-4] + '_discharge.csv'), index=False)

    @staticmethod
    def applyParallel(dfgrouped, func):
        res = Parallel(n_jobs=6)(delayed(func)(index, group) for index, group in dfgrouped)
        return pd.concat(res)


if __name__ == '__main__':
    read_path = 'E:\\2023年数字汽车大赛创新组赛题一数据\\2023年数字汽车大赛创新组赛题一数据'
    work_path = 'E:\\pycharm\\DigitalCarRace'
    files = [file for file in os.listdir(read_path) if file.endswith('.csv')]
    for file in files:
        model = SelectUsefulCol(read_path, work_path, file)
        model.read()
        model.calculateCapacity()
        model.save()

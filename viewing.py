"""
主处理的结果查看程序
"""

import collections

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

plt.rc("font", family='Microsoft YaHei')

# PreProcess
"""清洗数据，划分充放电片段"""


def viewingSOC():
    f = pd.read_csv(r'E:\pycharm\DigitalCarRace\dataSet\LFPHC7PE0K1A07972.csv', header=[0])
    f['数据采集时间'] = pd.to_datetime(f['数据采集时间'])
    fig = plt.figure()
    a1 = fig.add_subplot(111)
    plt.plot(f['数据采集时间'], f['总电压'], 'r--')
    plt.ylabel('总电压', fontsize=20)
    plt.xlabel('数据采集时间', fontsize=20)
    plt.legend(['总电压'], loc='upper left', fontsize=20)
    a2 = a1.twinx()
    plt.plot(f['数据采集时间'], f['总电流'], 'y-.')
    plt.ylabel('总电流', fontsize=20)
    plt.legend(['总电流'], loc='upper right', fontsize=20)
    plt.show()
    plt.plot(f['数据采集时间'], f['SOC'], 'y-')
    plt.ylabel('SOC', fontsize=20)
    plt.xlabel('数据采集时间', fontsize=20)
    plt.show()


"""异常里程记录查看"""


def viewingMileage():
    f = pd.read_csv('E:\\2023年数字汽车大赛创新组赛题一数据\\2023年数字汽车大赛创新组赛题一数据\\LFPHC7PE0K1A07972.csv',
                    header=[0])
    # f = pd.read_csv(r'E:\pycharm\DigitalCarRace\dataSet\LFPHC7PE0K1A07972.csv', header=[0])
    f['数据采集时间'] = pd.to_datetime(f['数据采集时间'])
    f.sort_values(['数据采集时间'])
    plt.plot(f['数据采集时间'], f['累计里程'], '-*')
    plt.xlabel('数据采集时间', fontsize=20)
    plt.ylabel('累计里程', fontsize=20)
    plt.show()


"""观察电压电流的采样精度"""


def viewingSamplingAccuracy():
    f = pd.read_csv(r'E:\pycharm\DigitalCarRace\dataSet\LFPHC7PE0K1A07972.csv', header=[0])
    f['数据采集时间'] = pd.to_datetime(f['数据采集时间'])
    voltages = f['cell1'].values
    currents = f['总电流'].values
    plt.plot(voltages, 'r.')
    plt.xlabel('数据帧索引')
    plt.ylabel('数值')
    plt.show()
    plt.plot(currents, 'r.')
    plt.xlabel('数据帧索引')
    plt.ylabel('数值')
    plt.show()

"""观察采样时间"""


def viewingSamplingTime():
    f = pd.read_csv(r'E:\pycharm\DigitalCarRace\dataSet\LFPHC7PE0K1A07972.csv', header=[0])
    f['数据采集时间'] = pd.to_datetime(f['数据采集时间'])
    f['deltT']=(f['数据采集时间']-f['数据采集时间'].shift(1)).apply(lambda x: x.total_seconds())
    deltTs=f['deltT'].value_counts()
    deltTs.to_csv('./a.csv')
    plt.bar(deltTs.index,deltTs.values)
    plt.xlabel('采样间隔')
    plt.ylabel('数量')
    plt.show()


"""查看充电片段和放电片段"""


def viewingCharge_Discharge():
    f1 = pd.read_csv(r'E:\pycharm\DigitalCarRace\chargeSet\LFPHC7PE0K1A07972_charge.csv', header=[0])
    f1['数据采集时间'] = pd.to_datetime(f1['数据采集时间'])
    fig = plt.figure()
    a1 = fig.add_subplot(111)
    plt.plot(f1['数据采集时间'], f1['总电压'], 'r--')
    plt.ylabel('总电压', fontsize=20)
    plt.xlabel('数据采集时间', fontsize=20)
    plt.legend(['总电压'], loc='upper left')
    a2 = a1.twinx()
    plt.plot(f1['数据采集时间'], f1['总电流'], 'y-.')
    plt.ylabel('总电流', fontsize=20)
    plt.legend(['总电流'], loc='upper right', fontsize=20)
    plt.show()
    f2 = pd.read_csv(r'E:\pycharm\DigitalCarRace\dischargeSet\LFPHC7PE0K1A07972_discharge.csv', header=[0])
    f2['数据采集时间'] = pd.to_datetime(f2['数据采集时间'])
    fig = plt.figure()
    a1 = fig.add_subplot(111)
    plt.plot(f2['数据采集时间'], f2['总电压'], 'r--')
    plt.ylabel('总电压', fontsize=20)
    plt.xlabel('数据采集时间', fontsize=20)
    plt.legend(['总电压'], loc='upper left')
    a2 = a1.twinx()
    plt.plot(f2['数据采集时间'], f2['总电流'], 'y-.')
    plt.ylabel('总电流', fontsize=20)
    plt.legend(['总电流'], loc='upper right', fontsize=20)
    plt.show()


"""通过安时积分法计算容量，观察充电片段的容量变化"""


def viewingCapacity():
    f = pd.read_csv(r'E:\pycharm\DigitalCarRace\chargeSet\LFPHC7PE0K1A07972_charge.csv', header=[0])
    f['数据采集时间'] = pd.to_datetime(f['数据采集时间'])
    fig = plt.figure()
    a1 = fig.add_subplot(121)
    plt.plot(f['累计里程'], f['cellCapacity'], 'k-*')
    plt.xlabel('累计里程（km）', fontsize=20)
    plt.ylabel('容量（Ah）', fontsize=20)
    a2 = fig.add_subplot(122)
    plt.plot(f['数据采集时间'], f['cellCapacity'], 'k-*')
    plt.xlabel('数据采集时间', fontsize=20)
    plt.ylabel('容量（Ah）', fontsize=20)
    plt.show()


"""查看箱型图筛选后的容量"""


def viewingCapacityAfterIQR():
    f = pd.read_csv(r'E:\pycharm\DigitalCarRace\RedefineSOH\LFPHC7PE0K1A07972_redefineCapacity_IQR.csv', header=[0])
    f['数据采集时间'] = pd.to_datetime(f['数据采集时间'])
    plt.plot(f['累计里程'], f['安时容量'], 'k-*')
    plt.xlabel('累计里程（km）', fontsize=20)
    plt.ylabel('容量（Ah）', fontsize=20)
    plt.show()


"""观察IC曲线,IC峰不明显，且常用特征的数值杂乱"""


def viewingIC():
    f = pd.read_csv(r'E:\pycharm\DigitalCarRace\chargeSet\LFPHC7PE0K1A07972_ICA5.csv', header=[0])
    f['数据采集时间'] = pd.to_datetime(f['数据采集时间'])
    legend = []

    def plotICA(group):
        # 绘制对应数据的变化图
        if group['voltage'].min() <= 4.0 and group['voltage'].max() >= 4.1:
            if group.loc[group.index.min(), 'group'] % 10 == 1:
                nonlocal legend
                legend.append(group.loc[group.index.min(), 'group'])
                # plt.plot(group.loc[(group['voltage'] > 4.0) & (group['voltage'] < 4.1), 'voltage'],
                #          group.loc[(group['voltage'] > 4.0) & (group['voltage'] < 4.1), 'dQ-dV'])
                plt.plot(group['voltage'], group['dQ-dV'])

    f.loc[f['cellName'] == 'cell1'].groupby(['group'], group_keys=True).apply(plotICA)
    plt.xlabel('电压/V', fontsize=20)
    plt.ylabel('dQ/dV', fontsize=20)
    # plt.legend(legend)
    plt.show()


"""查看DV曲线"""


def viewingDV():
    f = pd.read_csv(r'E:\pycharm\DigitalCarRace\chargeSet\LFPHC7PE0K1A07972_DVA.csv', header=[0])
    f['数据采集时间'] = pd.to_datetime(f['数据采集时间'])
    legend = []

    def plotDVA(group):
        # 绘制对应数据的变化图
        nonlocal legend
        legend.append(group.loc[group.index.min(), 'group'])
        # plt.plot(group.loc[(group['voltage'] > 4.0) & (group['voltage'] < 4.1), 'voltage'],
        #          group.loc[(group['voltage'] > 4.0) & (group['voltage'] < 4.1), 'dQ-dV'])
        plt.plot(group['SOC'], group['dV-dQ'])

    # f.loc[f['cellName'] == 'cell1'].groupby(['group'], group_keys=True).apply(plotDVA)
    f.groupby(['group', 'cellName']).apply(plotDVA)
    plt.xlabel('SOC', fontsize=20)
    plt.ylabel('dV/dQ', fontsize=20)
    # plt.legend(legend)
    plt.show()


"""原始容量未知，开路电压曲线未知，通过SOC计算的容量标签不准确，重新评估电池组的SOH"""


# 查看电池的所有电压数据分布
def viewingVoltages():
    f = pd.read_csv(r'E:\pycharm\DigitalCarRace\chargeSet\LFPHC7PE0K1A07972_charge.csv', header=[0])
    f['数据采集时间'] = pd.to_datetime(f['数据采集时间'])
    cells = [f'cell{i}' for i in range(1, 97)]
    voltages = f[cells].values.flatten()
    d = collections.Counter(voltages)
    plt.bar(d.keys(), d.values())
    plt.xlabel('单体电压/V')
    plt.ylabel('数量')
    plt.show()


# 不是很合理，改为查看每段充电片段的起始结束电压分布
def viewingVoltagesDistribution():
    f = pd.read_csv(r'E:\pycharm\DigitalCarRace\chargeSet\LFPHC7PE0K1A07972_charge.csv', header=[0])
    f['数据采集时间'] = pd.to_datetime(f['数据采集时间'])
    cells = [f'cell{i}' for i in range(1, 97)]

    def func(group):
        group.reset_index(drop=True, inplace=True)
        startVoltages = list(group.loc[0, cells].values)
        endVoltages = list(group.loc[group.shape[0] - 1, cells].values)
        df = pd.DataFrame({'startVoltages': startVoltages, 'endVoltages': endVoltages})
        return df

    voltagesfDistribute = f.groupby('group', group_keys=True).apply(func)
    startVoltagesDistribute = collections.Counter(voltagesfDistribute['startVoltages'])
    endVoltagesDistribute = collections.Counter(voltagesfDistribute['endVoltages'])
    print(np.percentile(voltagesfDistribute['startVoltages'], 80))
    print(np.percentile(voltagesfDistribute['endVoltages'], 20))
    fig = plt.figure()
    # a1 = fig.add_subplot(121)
    plt.bar(startVoltagesDistribute.keys(), startVoltagesDistribute.values())
    plt.xlabel('电压/V', fontsize=20)
    plt.ylabel('数量', fontsize=20)
    plt.title('充电起始电压分布', fontsize=20)
    plt.show()
    # a2 = fig.add_subplot(122)
    plt.bar(endVoltagesDistribute.keys(), endVoltagesDistribute.values())
    plt.xlabel('电压/V', fontsize=20)
    plt.ylabel('数量', fontsize=20)
    plt.title('充电结束电压分布', fontsize=20)
    plt.show()


"""从柱状图中可以看出，起始电压和结束电压均有集中片段，3.774V为起始电压的80%分位点，4.012V为结束电压的20%分位点，选择该电压范围重新定义SOH"""


# RedefineSOH

def viewingSOH():
    f = pd.read_csv(r'E:\pycharm\DigitalCarRace\RedefineSOH\LFPHC7PE0K1A07972_redefineCapacity_IQR.csv', header=[0])
    f['数据采集时间'] = pd.to_datetime(f['数据采集时间'])
    plt.plot(f['累计里程'], f['SOH1'], 'r*')
    plt.plot(f['累计里程'], f['SOH2'], 'b*')
    plt.plot(f['累计里程'], f['SOH'], 'k-*')
    plt.legend(['SOH1', 'SOH2', 'SOH'], fontsize=20, loc='upper right')
    plt.xlabel('累计里程', fontsize=20)
    plt.ylabel('SOH', fontsize=20)
    plt.show()


# 里程时间错乱数据过多
def viewingMileageError():
    f = pd.read_csv(r'E:\2023年数字汽车大赛创新组赛题一数据\2023年数字汽车大赛创新组赛题一数据\LFPHC7PE0K1B16707.csv',
                    header=[0])
    f['数据采集时间'] = pd.to_datetime(f['数据采集时间'])
    plt.plot(f['数据采集时间'], f['累计里程'], 'k*')
    plt.xlabel('数据采集时间', fontsize=20)
    plt.ylabel('累计里程', fontsize=20)
    plt.show()


# 样本过少
def viewingFewData():
    f = pd.read_csv(r'E:\2023年数字汽车大赛创新组赛题一数据\2023年数字汽车大赛创新组赛题一数据\LFPHC7PE5K1A24931.csv',
                    header=[0])
    f['数据采集时间'] = pd.to_datetime(f['数据采集时间'])
    plt.plot(f['数据采集时间'], f['累计里程'], 'k*')
    plt.xlabel('数据采集时间', fontsize=20)
    plt.ylabel('累计里程', fontsize=20)
    plt.show()


# 算法改变

def viewingAlgro_change():
    f = pd.read_csv(r'E:\pycharm\DigitalCarRace\chargeSet\LFPHC7PEXK1B02846_charge.csv',
                    header=[0])
    f['数据采集时间'] = pd.to_datetime(f['数据采集时间'])
    plt.plot(f['累计里程'],f['cellCapacity'], 'k*')
    plt.xlabel('累计里程', fontsize=20)
    plt.ylabel('安时容量', fontsize=20)
    plt.show()


"""查看电池组温度变化、温度方差变化，比较最大方差处的温度情况"""


def viewingTemperature():
    f = pd.read_csv(r'E:\pycharm\DigitalCarRace\chargeSet\LFPHC7PE0K1A07972_charge.csv', header=[0])
    f['数据采集时间'] = pd.to_datetime(f['数据采集时间'])
    tempProbes = [f'temp{i}' for i in range(1, 49)]
    for probe in tempProbes:
        plt.plot(f['数据采集时间'], f[probe], '-*')
    plt.xlabel('数据采集时间', fontsize=20)
    plt.ylabel('温度/℃', fontsize=20)
    plt.show()
    var_temp = [i.var() for i in f[tempProbes].values]
    plt.plot(f['数据采集时间'], var_temp, '-*')
    plt.xlabel('数据采集时间', fontsize=20)
    plt.ylabel('温度方差', fontsize=20)
    plt.show()
    print(max)


def viewingLoss():
    f1 = pd.read_csv(r'E:\pycharm\DigitalCarRace\CapacityModel1_2_3\CapacityEstimationModel_loss.csv')
    f2 = pd.read_csv(r'E:\pycharm\DigitalCarRace\CapacityModel1_2_3\CapacityEstimationModel_valid_loss.csv')
    plt.plot(f1['loss'], 'k*-')
    plt.plot([100 * i for i in f2.index], f2['loss'], 'r*')
    plt.legend(['训练集损失', '验证集损失'])
    plt.xlabel('迭代次数', fontsize=20)
    plt.ylabel('loss', fontsize=20)
    plt.show()


if __name__ == '__main__':
    viewingSamplingTime()

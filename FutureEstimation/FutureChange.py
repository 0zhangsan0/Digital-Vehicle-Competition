import os

import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np
import bisect

plt.rc("font", family='Microsoft YaHei')

random_seed = 43
random.seed(random_seed)
np.random.seed(random_seed)
move_frequency = pd.read_csv(r'E:\pycharm\DigitalCarRace\FutureSampling\move_frequency.csv', header=[0])
move_frequency['frequency'] = move_frequency['数量'] / move_frequency['数量'].sum()
moves = move_frequency['移动量'].values
temp_f = pd.read_csv(r'E:\pycharm\DigitalCarRace\FutureSampling\mileageDistribution.csv', header=[0])


def changeValue(move):
    if 0 <= move < 100:
        miu, sigma = 0.004446, 0.066678
    elif 100 <= move < 200:
        miu, sigma = 0.001704, 0.059242
    elif 200 <= move < 300:
        miu, sigma = 0.005591, 0.072391
    else:
        miu, sigma = -0.01577, 0.052023
    change = np.random.normal(miu, sigma, 1)[0]
    return change


def sampling():
    while True:
        move = random.uniform(0, 400)
        a = random.uniform(0, 1)
        idx = bisect.bisect_right(moves, move)
        if a <= move_frequency.loc[idx - 1, 'frequency']:
            change = changeValue(move)
            break
    return move, change


class Forecast:
    def __init__(self):
        self.f = pd.read_csv(r'E:\pycharm\DigitalCarRace\DeclineCapacity\LFPHC7PE0K1A07972_decline.csv', header=[0])

    def short_forecast(self):
        for i in range(679, 851):
            move, change = sampling()
            if i == 679:
                self.f.loc[i, 'capacity_guess'] = self.f.loc[i - 1, '平均容量'] + change
                self.f.loc[i, '累计里程_guess'] = self.f.loc[i - 1, '累计里程'] + move
            else:
                self.f.loc[i, 'capacity_guess'] = self.f.loc[i - 1, 'capacity_guess'] + change
                self.f.loc[i, '累计里程_guess'] = self.f.loc[i - 1, '累计里程_guess'] + move
        plt.plot(self.f['累计里程'].dropna(axis=0).values, self.f['平均容量'].dropna(axis=0).values, 'k*-')
        plt.plot(self.f['累计里程_guess'], self.f['capacity_guess'], 'r*-')
        plt.xlabel('累计里程', fontsize=20)
        plt.ylabel('容量', fontsize=20)
        plt.legend(['实际容量', '预测容量'], fontsize=20)
        plt.show()
        self.save_short()

    def save_short(self):
        self.f.to_csv(os.path.join(r'E:\pycharm\DigitalCarRace\FutrueEstimation', 'Capacity_short_forecast.csv'),
                      index=False)

    def long_forecast(self):
        move = sum(moves) / len(moves)
        change = temp_f.loc[
            (move - 5 <= temp_f['mileage_change']) & (temp_f['mileage_change'] <= move + 5), 'decline_pred'].mean()
        for i in range(679, 851):
            if i == 679:
                self.f.loc[i, 'capacity_guess'] = self.f.loc[i - 1, '平均容量'] + change
                self.f.loc[i, '累计里程_guess'] = self.f.loc[i - 1, '累计里程'] + move
            else:
                self.f.loc[i, 'capacity_guess'] = self.f.loc[i - 1, 'capacity_guess'] + change
                self.f.loc[i, '累计里程_guess'] = self.f.loc[i - 1, '累计里程_guess'] + move
        plt.plot(self.f['累计里程'].dropna(axis=0).values, self.f['平均容量'].dropna(axis=0).values, 'k*-')
        plt.plot(self.f['累计里程_guess'], self.f['capacity_guess'], 'r*-')
        plt.xlabel('累计里程', fontsize=20)
        plt.ylabel('容量', fontsize=20)
        plt.legend(['实际容量', '预测容量'], fontsize=20)
        plt.show()
        self.save_long()

    def save_long(self):
        self.f.to_csv(os.path.join(r'E:\pycharm\DigitalCarRace\FutrueEstimation', 'Capacity_long_forecast.csv'),
                      index=False)


if __name__ == '__main__':
    model = Forecast()
    # model.short_forecast()
    model.long_forecast()

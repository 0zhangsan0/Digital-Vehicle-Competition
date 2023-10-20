import math
import os

import matplotlib.pyplot as plt
import pandas as pd

plt.rc("font", family='Microsoft YaHei')


class ConditionDistribution:
    def __init__(self, decline_file, estimation_file, save_path):
        self.savePath = save_path
        declineFile = pd.read_csv(decline_file, header=[0])
        estimationFile = pd.read_csv(estimation_file, header=[0])
        declineFile.loc[2:, 'decline_pred'] = estimationFile['pred'].values
        declineFile['mileage_change'] = declineFile['累计里程_Pred'] - declineFile['累计里程_Pred'].shift(1)
        self.f = declineFile.drop([0, 1], axis=0)

    def viewing(self):
        self.f = self.f.loc[self.f['mileage_change'] <= 800]
        plt.hist(self.f['mileage_change'], bins=20)
        # counts, bins, patches=plt.hist(self.f['mileage_change'],bins=20)
        plt.xlabel('里程变化值', fontsize=20)
        plt.ylabel('数量', fontsize=20)
        plt.show()
        # hist_info = [bins[i] for i in range(len(bins) - 1)]
        # hist_counts = [int(c) for c in counts]
        # pd.DataFrame([item for item in zip(hist_info, hist_counts)], columns=['移动量', '数量']).to_csv(
        #     os.path.join(self.savePath, 'move_frequency.csv'), index=False)
        # self.f.to_csv(os.path.join(self.savePath,'mileageDistribution.csv'),index=False)
        max_mileage = self.f['累计里程_Pred'].max()
        n = math.ceil((max_mileage / 100))
        for i in range(n):
            subf = self.f.loc[(self.f['mileage_change'] >= i * 100) & (self.f['mileage_change'] < (i + 1) * 100)]
            if subf.shape[0] >= 10:
                counts, bins, patches=plt.hist(subf['decline_pred'], bins=20)
                hist_info = [bins[i] for i in range(len(bins) - 1)]
                hist_counts = [int(c)/sum(counts) for c in counts]
                pd.DataFrame([item for item in zip(hist_info, hist_counts)], columns=['容量变化量', '概率']).to_csv(
                    os.path.join(self.savePath, f'capacityChange_frequency{i * 100}-{(i + 1) * 100}.csv'), index=False)
                # plt.hist(subf['decline_pred'], bins=20)
                # plt.title(f'里程范围为[{i * 100},{i * 100 + 100}]km', fontsize=20)
                # plt.xlabel('容量变化', fontsize=20)
                # plt.ylabel('数量', fontsize=20)
                # plt.show()


if __name__ == '__main__':
    declineFile = r'E:\pycharm\DigitalCarRace\DeclineCapacity\LFPHC7PE0K1A07972_decline.csv'
    savePath = r'E:\pycharm\DigitalCarRace\FutureSampling'
    estimationFile = r'E:\pycharm\DigitalCarRace\DeclineModel4\DeclineModel8300.csv'
    model = ConditionDistribution(declineFile, estimationFile, savePath)
    model.viewing()

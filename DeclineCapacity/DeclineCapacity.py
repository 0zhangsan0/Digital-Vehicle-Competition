import os

import pandas as pd


class DeclineCapacity:
    def __init__(self, work_path, save_path):
        self.workPath = work_path
        self.savePath = save_path
        file = [item for item in os.listdir(work_path) if item.endswith('predict.csv')]
        self.f = pd.read_csv(os.path.join(self.workPath, file[0]), header=[0])

    def calculate_decline(self):
        self.f['decline'] = self.f['Pred'] - self.f['Pred'].shift(1)
        self.f['decline_avg'] = self.f['平均容量_pred'] - self.f['平均容量_pred'].shift(1)

    def save(self):
        self.f.to_csv(os.path.join(self.savePath, self.workPath[-4:] + '_decline.csv'), index=False)


if __name__ == '__main__':
    workPath = r'E:\pycharm\DigitalCarRace\CapacityModel3\3139'
    savePath = r'E:\pycharm\DigitalCarRace\DeclineCapacity'
    model = DeclineCapacity(workPath, savePath)
    model.calculate_decline()
    model.save()

import os.path as osp
import numpy as np
import pandas as pd
import logging
import plotly.graph_objects as go

from datetime import datetime, timedelta


class LgData():
    DATA_PATH = 'LG-18650HG2/'

    train_names = [
        '0degC/589_LA92',
        '0degC/589_UDDS',
        '0degC/589_US06',
        '0degC/589_HWFET',
        '0degC/589_Mixed1',
        '0degC/589_Mixed2',
        '0degC/590_Mixed7',
        '0degC/590_Mixed8',

        '10degC/582_LA92',
        '10degC/576_UDDS',
        '10degC/567_US06',
        '10degC/576_HWFET',
        '10degC/567_Mixed1',
        '10degC/567_Mixed2',
        '10degC/571_Mixed7',
        '10degC/571_Mixed8',

        '25degC/551_LA92',
        '25degC/551_UDDS',
        '25degC/551_US06',
        # '25degC/551_HWFET',
        '25degC/551_Mixed1',
        '25degC/551_Mixed2',
        '25degC/552_Mixed3',
        '25degC/552_Mixed7',
        '25degC/552_Mixed8',

        '40degC/556_LA92',
        '40degC/556_UDDS',
        '40degC/556_US06',
        '40degC/556_HWFET',
        '40degC/557_Mixed3',
        '40degC/562_Mixed4',
        '40degC/562_Mixed5',
        '40degC/562_Mixed6',
        '40degC/562_Mixed7',

        'n10degC/596_LA92',
        'n10degC/596_UDDS',
        'n10degC/596_HWFET',
        'n10degC/601_US06',
        'n10degC/601_Mixed1',
        'n10degC/601_Mixed2',
        # 'n10degC/604_Mixed3',
        'n10degC/602_Mixed4',
        'n10degC/604_Mixed7',
        'n10degC/604_Mixed8',

        'n20degC/610_LA92',
        'n20degC/610_UDDS',
        'n20degC/610_US06',
        'n20degC/610_HWFET',
        'n20degC/610_Mixed1',
        'n20degC/610_Mixed2',
        'n20degC/611_Mixed4',
        'n20degC/611_Mixed6',
        'n20degC/611_Mixed8',
    ]
    test_names = [
        '0degC/590_Mixed4',
        '0degC/590_Mixed5',
        '0degC/590_Mixed6',

        '10degC/571_Mixed4',
        '10degC/571_Mixed5',
        '10degC/571_Mixed6',

        '25degC/552_Mixed4',
        '25degC/552_Mixed5',
        '25degC/552_Mixed6',

        '40degC/556_Mixed1',
        '40degC/556_Mixed2',
        '40degC/562_Mixed8',

        # 'n10degC/602_Mixed4',
        'n10degC/602_Mixed5',
        'n10degC/604_Mixed3',
        'n10degC/604_Mixed6',

        'n20degC/611_Mixed3',
        'n20degC/611_Mixed5',
        'n20degC/611_Mixed7',
    ]

    SOCOCV_names = [
        'OCV--40degC--555_C20DisCh.csv',
        'OCV--25degC--549_C20DisCh.csv',
        'OCV--10degC--575_C20DisCh.csv',
        'OCV--0degC--585_C20DisCh.csv',
        'OCV--n10degC--593_C20DisCh.csv',
        'OCV--n20degC--607_C20DisCh.csv']

    coefficients = {40: [9.04188628, -26.68143663, 28.94018461, -14.16225798, 3.88040261, 3.14699274],
                    25: [7.72839897, -23.46251858, 26.22871258, -13.24717859, 3.78759186, 3.12893346],
                    10: [10.507937, -30.84566487, 33.24255085, -16.08258133, 4.20925841, 3.1376317],
                    0: [10.24909908, -30.04026548, 32.29372416, -15.55038046, 4.05511804, 3.16186655],
                    -10: [9.37839605, -27.95058355, 30.52695759, -14.93672446, 3.9785498, 3.16478754],
                    -20: [7.57317727, -23.68100254, 26.71499225, -13.36845676, 3.70040954, 3.18732864]}

    def __init__(self, base_path="./", steps = 300, interval = 300, scale_test = True):
        self.path = osp.join(base_path, self.DATA_PATH)
        self.steps = steps
        self.interval = interval

        self.unfinished = ['40degC/562_Mixed4', '40degC/562_Mixed5', '40degC/562_Mixed6', '40degC/562_Mixed7', '40degC/562_Mixed8']

        cycles = self.get_discharge_whole_cycle(self.train_names, self.test_names, output_capacity=False, scale_test=scale_test)

        self.train_x, self.train_y, self.test_x, self.test_y, self.train_k, self.test_k = self.get_discharge_multiple_step(cycles)

        self.train_SOCOCVs, self.test_SOCOCVs = self.get_SOCOCV()

    def OCV_to_SOC5(self, OCV, *popt):
        a, b, c, d, e, f = popt
        return np.roots([a, b, c, d, e, f - OCV])

    def get_SOCOCV(self):
        train_SOCOCVs = []
        for i in range(self.train_k.shape[0]):
            temp = int(self.train_k[i][0].split('_')[0])
            popt = self.coefficients[temp]
            pred = self.OCV_to_SOC5(self.train_x[i][0, 0], *popt)
            pred = np.real(pred[np.isreal(pred)])
            train_SOCOCVs.append(pred)

        test_SOCOCVs = []
        for i in range(self.test_x.shape[0]):
            temp = int(self.test_k[i][0].split('_')[0])
            popt = self.coefficients[temp]
            pred = self.OCV_to_SOC5(self.test_x[i][0, 0], *popt)
            pred = np.real(pred[np.isreal(pred)])
            test_SOCOCVs.append(pred)

        return train_SOCOCVs, test_SOCOCVs


    def get_discharge_whole_cycle(self, train_names, test_names, output_capacity=False, scale_test=False,
                                  output_time=False):
        train, train_k = self._get_data(train_names, output_capacity, output_time)
        test, test_k = self._get_data(test_names, output_capacity, output_time)
        train, test = self._scale_x(train, test, scale_test=scale_test)
        return (train, test, train_k, test_k)

    def _get_data(self, names, output_capacity, output_time=False):
        cycles = []
        ks = []
        for name in names:
            cycle = pd.read_csv(osp.join(self.path, name + '.csv'), skiprows=30)
            cycle.columns = ['Time Stamp', 'Step', 'Status', 'Prog Time', 'Step Time', 'Cycle',
                             'Cycle Level', 'Procedure', 'Voltage', 'Current', 'Temperature', 'Capacity', 'WhAccu',
                             'Cnt', 'Empty']
            cycle = cycle[(cycle["Status"] == "TABLE") | (cycle["Status"] == "DCH")]

            if name in self.unfinished:
                max_discharge = 2.52553
            else:
                max_discharge = abs(min(cycle["Capacity"]))
            cycle["SoC Capacity"] = max_discharge + cycle["Capacity"]
            cycle["SoC Percentage"] = cycle["SoC Capacity"] / max(cycle["SoC Capacity"])
            x = cycle[["Voltage", "Current", "Temperature"]].to_numpy()

            if output_time:
                cycle['Prog Time'] = cycle['Prog Time'].apply(self._time_string_to_seconds)
                cycle['Time in Seconds'] = cycle['Prog Time'] - cycle['Prog Time'][0]

            if output_capacity:
                if output_time:
                    y = cycle[["SoC Capacity", "Time in Seconds"]].to_numpy()
                else:
                    y = cycle[["SoC Capacity"]].to_numpy()
            else:
                if output_time:
                    y = cycle[["SoC Percentage", "Time in Seconds"]].to_numpy()
                else:
                    y = cycle[["SoC Percentage"]].to_numpy()

            if np.isnan(np.min(x)) or np.isnan(np.min(y)):
                print("There is a NaN in cycle " + name + ", removing row")
                x = x[~np.isnan(x).any(axis=1)]
                y = y[~np.isnan(y).any(axis=1)].reshape(-1, y.shape[1])

            cycles.append((x, y))

            k = name.split('/')[0][:-4]
            if 'n' in k:
                k = -int(k[1:])
            else:
                k = int(k)
            k = [str(k) + '_' + name.split('/')[-1]] * len(x)
            ks.append(k)

        return cycles, ks

    def _time_string_to_seconds(self, input_string):
        time_parts = input_string.split(':')
        second_parts = time_parts[2].split('.')
        return timedelta(hours=int(time_parts[0]),
                         minutes=int(time_parts[1]),
                         seconds=int(second_parts[0]),
                         microseconds=int(second_parts[1])).total_seconds()

    def _scale_x(self, train, test, scale_test=False):
        for index_feature in range(len(train[0][0][0])):
            feature_min = min([min(cycle[0][:, index_feature]) for cycle in train])
            feature_max = max([max(cycle[0][:, index_feature]) for cycle in train])
            for i in range(len(train)):
                train[i][0][:, index_feature] = (train[i][0][:, index_feature] - feature_min) / (
                            feature_max - feature_min)
            if scale_test:
                for i in range(len(test)):
                    test[i][0][:, index_feature] = (test[i][0][:, index_feature] - feature_min) / (
                                feature_max - feature_min)

        return train, test

    #################################
    #
    # get_stateful_cycle
    #
    #################################
    def get_stateful_cycle(self, cycles, pad_num=0, steps=100):
        max_lenght = max(max(len(cycle[0]) for cycle in cycles[0]), max(len(cycle[0]) for cycle in cycles[1]))
        train_x, train_y = self._to_padded_cycle(cycles[0], pad_num, max_lenght)
        test_x, test_y = self._to_padded_cycle(cycles[1], pad_num, max_lenght)
        train_x = self._split_cycle(train_x, steps)
        train_y = self._split_cycle(train_y, steps)
        test_x = self._split_cycle(test_x, steps)
        test_y = self._split_cycle(test_y, steps)
        print("Train x: %s, train y: %s | Test x: %s, test y: %s" %
                         (train_x.shape, train_y.shape, test_x.shape, test_y.shape))
        return (train_x, train_y, test_x, test_y)

    def _to_padded_cycle(self, cycles, pad_num, max_lenght):
        x_length = len(cycles[0][0][0])
        y_length = len(cycles[0][1][0])
        x = np.full((len(cycles), max_lenght, x_length), pad_num, dtype=float)
        y = np.full((len(cycles), max_lenght, y_length), pad_num, dtype=float)
        for i, cycle in enumerate(cycles):
            x[i, :cycle[0].shape[0]] = cycle[0]
            y[i, :cycle[1].shape[0]] = cycle[1]
        return x, y

    def _split_cycle(self, cycles, steps):
        features = cycles.shape[2]
        time_steps = cycles.shape[1]
        new_cycles = np.empty((0, time_steps // steps, steps, features), float)
        for cycle in cycles:
            new_cycle = np.empty((0, steps, features), float)
            for i in range(0, len(cycle) - steps, steps):
                next_split = np.array(cycle[i:i + steps]).reshape(1, steps, features)
                new_cycle = np.concatenate((new_cycle, next_split))
            new_cycles = np.concatenate((new_cycles, new_cycle.reshape(1, time_steps // steps, steps, features)))
        return new_cycles

    #################################
    #
    # get_discharge_multiple_step
    #
    #################################

    def get_discharge_multiple_step(self, cycles):
        train_x, train_y, train_k = self._split_to_multiple_step(cycles[0], cycles[2])
        test_x, test_y, test_k = self._split_to_multiple_step(cycles[1], cycles[3])
        print("Train x: %s, train y: %s | Test x: %s, test y: %s" %
                         (train_x.shape, train_y.shape, test_x.shape, test_y.shape))
        return (train_x, train_y, test_x, test_y, train_k, test_k)

    def _split_to_multiple_step(self, cycles, ks):
        x,y,k = [],[],[]
        for j, cycle in enumerate(cycles):
            for i in range(0, len(cycle[0]) - self.steps, self.interval):
                x.append(cycle[0][i:i + self.steps])
                y.append(cycle[1][i:i + self.steps])
                k.append(ks[j][i:i + self.steps])
        return np.array(x), np.array(y), np.array(k)
    # def _split_to_multiple_step(self, cycles, steps):
    #     x_length = len(cycles[0][0][0])
    #     y_length = len(cycles[0][1][0])
    #     x = np.empty((0, steps, x_length), float)
    #     y = np.empty((0, steps, y_length), float)
    #     for cycle in cycles:
    #         for i in range(0, len(cycle[0]) - steps, steps):
    #             next_x = np.array(cycle[0][i:i + steps]).reshape(1, steps, x_length)
    #             next_y = np.array(cycle[1][i:i + steps]).reshape(1, steps, y_length)
    #             x = np.concatenate((x, next_x))
    #             y = np.concatenate((y, next_y))
    #     return x, y

    def keep_only_y_end(self, y, step, is_stateful=False):
        if is_stateful:
            return y[:, :, [-1]]
        else:
            return y[:, [-1]]


__dataset__ = {
    'LgData': LgData,
}

if __name__ == '__main__':
    LgData('dataset/')
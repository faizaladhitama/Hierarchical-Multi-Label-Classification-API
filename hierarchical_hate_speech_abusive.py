import copy
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

import numpy as np

class HierarchicalHateSpeechAbusive:
    def __init__(self):
        self.svcs = [LogisticRegression() for i in range(9)]
        self.features = pickle.load(open('word1.pkl', 'rb'))
        self.word1 = pickle.load(open('word1_.pkl', 'rb'))
        self.y = pickle.load(open('label.pkl', 'rb'))
        self.y_ = pickle.load(open('label_merged_target_level.pkl', 'rb'))

    def fit(self, x_train, y_train, y_test):
        x_test = x_train
        y_temp = pd.DataFrame(0, columns=y_test.columns.values, index=y_test.index.values)

        x_res, y_res = x_train, y_train[0]
        self.svcs[0].fit(x_res, y_res)
        y_pred_hs = self.svcs[0].predict(x_test)

        y_temp[0] = y_pred_hs

        hs_idx = np.where(y_pred_hs == 0)
        not_hs_idx = np.where(y_pred_hs == 1)

        y_train_temp = copy.deepcopy(y_train)
        y_train_temp.index = range(len(y_train_temp))

        train_hs_idx = y_train_temp[y_train_temp[0] == 0].index.values

        x_res, y_res = x_train, y_train[1]
        self.svcs[1].fit(x_res, y_res)
        y_pred_abusive = self.svcs[1].predict(x_test)

        y_temp[1] = y_pred_abusive

        x_res, y_res = x_train[train_hs_idx, :], y_train[2].iloc[train_hs_idx]
        self.svcs[2].fit(x_res, y_res)
        y_pred_target = self.svcs[2].predict(x_test[hs_idx])

        for i in range(2):
            target_idx = np.where(y_pred_target == i + 1)
            change_idx = y_temp[i + 2].iloc[hs_idx].iloc[target_idx].index.values
            y_temp[i + 2].loc[change_idx] = 1

        makian_idx = None
        for i in range(4):
            x_res, y_res = x_train[train_hs_idx, :], y_train[i + 3].iloc[train_hs_idx]
            self.svcs[3 + i].fit(x_res, y_res)
            y_pred_golongan = self.svcs[3 + i].predict(x_test[hs_idx])

            golongan_idx = np.where(y_pred_golongan == 1)
            not_golongan_idx = np.where(y_pred_golongan == 0)
            change_idx = y_temp[i + 4].iloc[hs_idx].iloc[golongan_idx].index.values
            y_temp[i + 4].loc[change_idx] = 1

            if not isinstance(makian_idx, tuple) and not isinstance(makian_idx, np.ndarray):
                makian_idx = not_golongan_idx
            else:
                makian_idx = np.intersect1d(makian_idx, not_golongan_idx)

        change_idx = y_temp[8].iloc[hs_idx].iloc[makian_idx].index.values
        y_temp[8].loc[change_idx] = 1

        x_res, y_res = x_train[train_hs_idx, :], y_train[8].iloc[train_hs_idx]
        self.svcs[8].fit(x_res, y_res)
        y_pred = self.svcs[8].predict(x_test[hs_idx])

        for i in range(3):
            tingkat_idx = np.where(y_pred == i + 1)
            change_idx = y_temp[i + 9].iloc[hs_idx].iloc[tingkat_idx].index.values
            y_temp[i + 9].loc[change_idx] = 1
        return self

    def _predict(self, text):

        if type(text) is not list:
            text = [str(text)]

        word1_ = self.word1.transform(text)

        y_temp = pd.DataFrame(0, columns=[i for i in range(12)], index=range(len(text)))

        y_pred_hs = self.svcs[0].predict(word1_)

        hs_idx = np.where(y_pred_hs == 0)
        not_hs_idx = np.where(y_pred_hs == 1)

        y_temp[0] = y_pred_hs

        y_train_temp = copy.deepcopy(y_temp)
        y_train_temp.index = range(len(y_temp))

        train_hs_idx = y_train_temp[y_train_temp[0] == 0].index.values

        y_pred_abusive = self.svcs[1].predict(word1_)

        y_temp[1] = y_pred_abusive

        y_pred_target = self.svcs[2].predict(word1_[hs_idx])

        for i in range(2):
            target_idx = np.where(y_pred_target == i + 1)
            change_idx = y_temp[i + 2].iloc[hs_idx].iloc[target_idx].index.values
            y_temp[i + 2].loc[change_idx] = 1

        makian_idx = None
        for i in range(4):
            y_pred_golongan = self.svcs[3 + i].predict(word1_[hs_idx])

            golongan_idx = np.where(y_pred_golongan == 1)
            not_golongan_idx = np.where(y_pred_golongan == 0)
            change_idx = y_temp[i + 4].iloc[hs_idx].iloc[golongan_idx].index.values
            y_temp[i + 4].loc[change_idx] = 1

            if not isinstance(makian_idx, tuple) and not isinstance(makian_idx, np.ndarray):
                makian_idx = not_golongan_idx
            else:
                makian_idx = np.intersect1d(makian_idx, not_golongan_idx)

        change_idx = y_temp[8].iloc[hs_idx].iloc[makian_idx].index.values
        y_temp[8].loc[change_idx] = 1

        y_pred = self.svcs[8].predict(word1_[hs_idx])

        for i in range(3):
            tingkat_idx = np.where(y_pred == i + 1)
            change_idx = y_temp[i + 9].iloc[hs_idx].iloc[tingkat_idx].index.values
            y_temp[i + 9].loc[change_idx] = 1
        return y_temp

    def predict(self, text):
        result = self._predict(text)

        dict_result = []

        for i in range(result.shape[0]):
            dict_ = {
                'hate_speech': False,
                'abusive_language': False,
                'target': '',
                'category': [],
                'level': ''
            }
            row = result.iloc[i]
            if row[0] == 0:
                dict_['hate_speech'] = True
                if row[2] == 1:
                    dict_['target'] = 'individual'
                elif row[3] == 1:
                    dict_['target'] = 'group'

                if row[4] == 1:
                    dict_['category'].append('religion')
                if row[5] == 1:
                    dict_['category'].append('race')
                if row[6] == 1:
                    dict_['category'].append('physical')
                if row[7] == 1:
                    dict_['category'].append('gender')
                if row[8] == 1:
                    dict_['category'].append('offensive')

                if row[9] == 1:
                    dict_['level'] = 'weak'
                if row[10] == 1:
                    dict_['level'] = 'moderate'
                if row[11] == 1:
                    dict_['level'] = 'strong'
            if row[1] == 1:
                dict_['abusive_language'] = True

            dict_result.append(dict_)
        return dict_result
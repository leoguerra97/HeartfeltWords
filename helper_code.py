import argparse
import ast

import numpy as np
import pandas as pd
from scipy import signal
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold


def bandpass_filter(frame):
    b, a = signal.butter(2, [0.002,0.05], btype='bandpass') # 0.002 = 0.5 Hz / 250 Hz (which is Nyquist Frequency since Fs = 500 Hz)
    frame = signal.filtfilt(b,a,frame).copy()
    return frame

def load_ptbxl_data_paths(path):
    df = pd.read_csv(path + 'ptbxl_database_o.csv', index_col='ecg_id')
    df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))

    agg_df = pd.read_csv(path + 'scp_statements.csv', index_col=0)  # csv with aggregate diagnostic
    agg_df = agg_df[agg_df.diagnostic == 1]

    df['diagnostic_superclass'] = df.scp_codes.apply(aggregate_diagnostic, df=agg_df)
    selected_columns = ['patient_id', 'age', 'sex', 'height', 'weight', 'report', 'nurse', 'recording_date',
                        'scp_codes', 'strat_fold', 'filename_lr', 'filename_hr', 'diagnostic_superclass']
    df = df[selected_columns]
    return df

def aggregate_diagnostic(y_dic, df):
    tmp = []
    for key in y_dic.keys():
        if key in df.index:
            tmp.append(df.loc[key].diagnostic_class)
    return list(set(tmp))

def open_reports(path, lang):
    reports_path = path + lang + '_df_round4.csv'
    reports_df = pd.read_csv(reports_path, index_col=0)
    return reports_df

def correct_reports(reports_df):
    nan_rows = reports_df[reports_df['report'].isnull()]
    print(nan_rows)
    for el in nan_rows.index:
        reports_df['report'].loc[el] = 'no_report'
    return reports_df

def split_data(labels, y_all_combo, random_state=8008):
    folds = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state).split(labels,y_all_combo))
    print("Training split: {}".format(len(folds[0][0])))
    print("Validation split: {}".format(len(folds[0][1])))
    return folds

def encode_target(target_class):
    target_dict = np.unique(target_class)
    label_en = preprocessing.LabelEncoder()
    label_en.fit(target_class) #CHECK TYPE
    target = label_en.transform(target_class)
    return target

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

import pandas as pd 
from tqdm import tqdm
from pose_pred_algorithms import pose_mean_algo


pose_data = pd.read_excel('/content/positions_markup_train.xlsx')
pose_data = pose_data[pose_data['P1'] != '-'].reset_index(drop=True)

def acc_pose(model, df, data_path):
    corr, total = 0, df.shape[0] * (df.shape[1] - 1)
    for i in tqdm(range(df.shape[0])):
        row = pose_data.iloc[i]
        y_true = row[1:].tolist()
        y_pred = pose_mean_algo(model, data_path + row[0])
        corr += sum([abs(t - p) <= 60 for t, p in zip(y_true, y_pred)])
    return corr / total
  

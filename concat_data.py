import os
import pandas as pd
import numpy as np

def concat_data(standard_path,simulink_out_path,save_path):
    for name in os.listdir(standard_path):
        left = os.path.join(standard_path,name)
        right = os.path.join(simulink_out_path, name)
        save_path1 = os.path.join(save_path, name)
        left = pd.read_excel(left,sheet_name=0)
        if os.path.exists(right):
            right = pd.read_excel(right, sheet_name=0)
            # 拼接数据
            concat_out = pd.DataFrame(pd.concat((left,right.iloc[:len(left),1:]),axis=1))
            concat_out.to_excel(save_path1)
        else:
            pass
            # ValueError('Nothing can be match!')

if __name__ == '__main__':
    standard_path = 'E:\Datasets\Li_ion\data\WUT_experiments\Standard'
    simulink_out_path = 'E:\Datasets\Li_ion\data\WUT_experiments\C1C2R0R1R2SOCSOEpred'
    save_path = 'E:\Datasets\Li_ion\data\WUT_experiments\All_valid'
    concat_data(standard_path,simulink_out_path,save_path)

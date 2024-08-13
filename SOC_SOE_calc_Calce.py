import pandas as pd
import numpy as np
import os

def GetRaw(path):
    assert path is not None,'Path is invalid!'
    Context = pd.read_excel(path,sheet_name=1)
    data_step_index = Context.index[Context['Step_Index'] == 7]
    data_step = Context.iloc[data_step_index, :]

    Q = 2028
    E = 7.346678
    SOC_init = 0.8
    SOE_init = 0.8
    return data_step,Q,E,SOC_init,SOE_init

def calc_SOC_SOE(context,Q,E,SOC_init,SOE_init):
    valid = np.array(context.iloc[:, [1,6,7]])
    remain = {
        'Q_remain': SOC_init,
        'E_remain': SOE_init,
    }
    SOCs = []
    SOEs = []
    for index in range(len(context)):
        delta_time = valid[index, 0] - (valid[index - 1, 0]) if index > 0 else 0
        capacity = valid[index, 1] * delta_time
        energy = valid[index, 1] * valid[index, 2] * delta_time
        remain['Q_remain'] = remain['Q_remain'] + capacity * 1000 / 3600 / Q
        remain['E_remain'] = remain['E_remain'] + energy / 3600 / E
        if remain['Q_remain']>=0.1:
            SOCs.append(remain['Q_remain'])
            SOEs.append(remain['E_remain'])

    return SOCs, SOEs

def resave(context,soc,soe,name):
    context = context.reset_index()
    soc = pd.DataFrame(soc, columns=['soc'])
    soe = pd.DataFrame(soe, columns=['soe'])
    index = len(soc)
    dataframe = pd.concat([context.iloc[:index],soc,soe],axis=1)
    dataframe.to_excel(name)

def main():
    path = r'data\Li_ion\SP2\US06'
    save_main = r'data\Calce\US06'
    for i in os.listdir(path):
        if i.endswith('.xls'):
            excel_path = os.path.join(path,i)
            Context,Q,E,SOC_init,SOE_init = GetRaw(excel_path)
            SOCs,SOEs = calc_SOC_SOE(Context,Q,E,SOC_init,SOE_init)
            save_path = os.path.join(save_main,i)
            resave(Context, SOCs, SOEs, save_path)
            print(f'Excel {i} is processed and saved!')

if __name__ == '__main__':
    main()












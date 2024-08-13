from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse
import pandas
import matplotlib.pyplot as plt
import pandas as pd

# Import Data
df = pd.read_excel(r'E:\LSTF_Projects\PatchTST_supervised\data\Li_ion\hppc\1号lpf电池hppc实验（复测）.xlsx',sheet_name=0)
df_raw = pd.concat((df.loc[::300,'绝对时间'],df.loc[::300,'soc']),axis=1)
# df_raw['绝对时间'] = pd.to_datetime(df_raw['绝对时间'])
df_raw.set_index('绝对时间', inplace=True)
df_raw['soc'] += 0.01
# Multiplicative Decomposition
result_mul = seasonal_decompose(df_raw, model='multiplicative', extrapolate_trend='freq',period=8)

# Additive Decomposition
result_add = seasonal_decompose(df_raw, model='additive', extrapolate_trend='freq',period=8)

# Plot
plt.rcParams.update({'figure.figsize': (10, 10)})
result_mul.plot().suptitle('Multiplicative Decompose', fontsize=22)
result_add.plot().suptitle('Additive Decompose', fontsize=22)
plt.show()

import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# 获取分解结果的数据
observed = result_add.observed.values
trend = result_add.trend.values
seasonal = result_add.seasonal.values
resid = result_add.resid.values
dates = df_raw.index

# 创建一个新的3D绘图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 生成x轴的时间序列
x = np.arange(len(dates))

# 绘制原始数据
ax.plot(x, 4*np.ones_like(x), observed, label='Observed', color='#1399B2',linewidth=2)

# 绘制趋势成分
ax.plot(x, 3*np.ones_like(x), trend*0.9, label='Trend', color='#8E7FB8', linewidth=2, alpha=0.8)

# 绘制季节性成分
ax.plot(x, 2*np.ones_like(x), seasonal*60, label='Seasonal', color='#D77071', linewidth=2, alpha=0.5)

# 绘制残差成分
ax.plot(x, 1*np.ones_like(x), resid*1, label='Residual', color='orange', linewidth=2, alpha=0.5)

# 设置轴标签
ax.view_init(elev=20, azim=240)
ax.set_xlabel('Time slices',fontsize=16,labelpad=20)
ax.set_ylabel('Components',fontsize=16,labelpad=20)
ax.set_zlabel('Value',fontsize=16,labelpad=20)

# 设置y轴的刻度和标签
ax.set_yticks([4,3,2,1])
ax.set_yticklabels(['SOC', 'Trend', 'Seasonal', 'Residual'])
plt.tick_params(axis='x', labelsize=16)
plt.tick_params(axis='y', labelsize=16)
plt.tick_params(axis='z', labelsize=16)

# 添加图例
ax.legend(loc='center right')
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
import random

df = pd.read_excel(r'data\Li_ion\1号lpf电池0.5，1.5充放电实验.xlsx',sheet_name=0)
df_raw = pd.concat((df.loc[:,'电流(mA)':'电压(V)'],df.loc[:,'C1':'R2'],df.loc[:,'soc']),axis=1)
import pandas as pd
from dabl import plot
import klib
klib.corr_plot(df_raw, annot=False)
# df = pd.read_csv("titanic.csv")
plot(df_raw, 'soc')
plt.show()
X = pd.concat((df.loc[:,'电流(mA)':'电压(V)'],df.loc[:,'C1':'R2']),axis=1)
X.rename(columns={'电流(mA)':'Current'}, inplace = True)
X.rename(columns={'电压(V)':'Voltage'}, inplace = True)
y = df.loc[:,'soc']
sum = len(X)
train_num = int(sum*0.8)
val_num = sum - train_num
train_idx = np.sort(random.sample(list(range(len(X))),train_num))
vaL_idx = np.sort(list(set(range(len(X))) - set(train_idx)))
X_train, X_val, y_train, y_val  = X.loc[train_idx],X.loc[vaL_idx],y[train_idx],y[vaL_idx]
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# # 然后将训练集进一步划分为训练集和验证集
# X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, random_state=42)  # 0.125 x 0.8 = 0.1

# 数据集标准化
x_mean = X_train.mean()
x_std =  X_train.std()
y_mean = y.mean()
y_std = y.std()
X_train = (X_train - x_mean)/x_std
y_train = (y_train-y_mean)/y_std
X_val = (X_val - x_mean)/x_std
y_val = (y_val - y_mean)/y_std

import lightgbm as lgb

# LightGBM模型参数
params_lgb = {
    'learning_rate': 0.02,          # 学习率，控制每一步的步长，用于防止过拟合。典型值范围：0.01 - 0.1
    'boosting_type': 'gbdt',        # 提升方法，这里使用梯度提升树（Gradient Boosting Decision Tree，简称GBDT）
    'objective': 'mse',             # 损失函数
    'metric': 'rmse',               # 评估指标
    'num_leaves': 127,              # 每棵树的叶子节点数量，控制模型复杂度。较大值可以提高模型复杂度但可能导致过拟合
    'verbose': -1,                  # 控制 LightGBM 输出信息的详细程度，-1表示无输出，0表示最少输出，正数表示输出更多信息
    'seed': 42,                     # 随机种子，用于重现模型的结果
    'n_jobs': -1,                   # 并行运算的线程数量，-1表示使用所有可用的CPU核心
    'feature_fraction': 0.8,        # 每棵树随机选择的特征比例，用于增加模型的泛化能力
    'bagging_fraction': 0.9,        # 每次迭代时随机选择的样本比例，用于增加模型的泛化能力
    'bagging_freq': 4               # 每隔多少次迭代进行一次bagging操作，用于增加模型的泛化能力
}

model_lgb = lgb.LGBMRegressor(**params_lgb)
model_lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              eval_metric='rmse')
pred_train = model_lgb.predict(X_train)
pred_val = model_lgb.predict(X_val)

y_train_h = y_train*y_std+y_mean
pred_train_h = pred_train*y_std+y_mean

y_val_h = y_val*y_std+y_mean
pred_val_h = pred_val*y_std+y_mean


mae = mean_absolute_error(y_val_h,pred_val_h)
rmse = np.sqrt(mean_squared_error(y_val_h,pred_val_h))
r2 = r2_score(y_val_h,pred_val_h)
print(f'mae:{mae}, rmse:{rmse}, r2:{r2}')

import seaborn as sns
colors = sns.color_palette("husl", 3)

plt.figure(dpi=300)
plt.plot(y_val_h.values[:], label='True', alpha=1, color=colors[0])
plt.plot(pred_val_h[:], label='Pred', alpha=1, color=colors[1])
plt.xlabel('t')
plt.ylabel('SOC')
plt.legend()
plt.tight_layout()
plt.show()


import shap
# 构建 shap解释器
explainer = shap.TreeExplainer(model_lgb)
# 计算测试集的shap值
shap_values = explainer.shap_values(X_train)
# 特征标签
labels = X_train.columns
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times new Roman'
plt.rcParams['font.size'] = 20
#cmap="?"配色viridis Spectral coolwar mRdYlGn RdYlBu RdBu RdGy PuOr BrBG PRGn PiYG
fig,ax  = plt.subplots(figsize=(10,10))
shap.summary_plot(shap_values, X_train, feature_names=labels, plot_type="violin",show=False)
# 设置x轴和y轴标签的字体大小
ax.set_xlabel(ax.get_xlabel(), fontsize=20)  # 设置x轴标签字体大小
ax.set_ylabel(ax.get_ylabel(), fontsize=20)  # 设置y轴标签字体大小

# 设置y轴刻度标签的字体大小
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)

# 计算训练集和测试集的SHAP值
shap_values_train = explainer.shap_values(X_train)
shap_values_val = explainer.shap_values(X_val)
plt.show()
# 绘制SHAP值总结图（Summary Plot）
plt.figure(dpi=300)
plt.subplot(1, 2, 1)
shap.summary_plot(shap_values_train, X_train, plot_type="violin", show=False,color='#23BAC5')
plt.title("feature_train",fontsize=20)
plt.xlabel('')  # 移除 x 轴标签避免x轴重叠

plt.subplot(1, 2, 2)
shap.summary_plot(shap_values_val, X_val, plot_type="violin", show=False,color='#FD763F')
plt.title("feature_val",fontsize=20)

plt.tight_layout()
plt.show()

#依赖图（Dependence Plot）
shap.dependence_plot('C1', shap_values, X_train, interaction_index='R0')
#绘制单个样本的SHAP解释（Force Plot）
# sample_index = 40  # 选择一个样本索引进行解释
# shap.force_plot(explainer.expected_value, shap_values_val[sample_index], X_val.iloc[sample_index], matplotlib=True)

# 创建 shap.Explanation 对象
shap_explanation = shap.Explanation(values=shap_values_val[:3000,:],
                                    base_values=explainer.expected_value,
                                    data=X_val.iloc[:3000,:], feature_names=X_val.columns)
# 绘制热图
fig1,ax  = plt.subplots()
shap.plots.heatmap(shap_explanation,show=False)
# 设置x轴和y轴标签的字体大小
ax.set_xlabel(ax.get_xlabel(), fontsize=20)  # 设置x轴标签字体大小
ax.set_ylabel(ax.get_ylabel(), fontsize=20)  # 设置y轴标签字体大小

# 设置y轴刻度标签的字体大小
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
plt.show()
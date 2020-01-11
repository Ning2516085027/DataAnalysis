#%%
#多元线性回归
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score                #直接调用库函数进行输出R2

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
data=pd.read_csv("../DataSet/wine-quality/winequality-white.csv",sep=',')

plt.style.use("ggplot")#格式美化
data.hist(figsize=(12,12), color="#476DD5", edgecolor="k")#绘制直方图
#使用sklearn.preprocessing.scale()函数进行数据标准化（Z-Score）
data_scaled = scale(data)
columns = ["非挥发性酸","挥发性酸","柠檬酸","剩余糖分","氯化物","游离二氧化硫","总二氧化硫",
            "密度","酸碱性","硫酸盐","酒精","质量"]
data_df_scaled = pd.DataFrame(data_scaled)
#绘制图形
data_df_scaled.hist(figsize=(15,8))
#交叉验证
xx_train, xx_test, yy_train, yy_test = train_test_split(data_df_scaled.iloc[:,:11], data_df_scaled.iloc[:,-1], test_size=0.3,random_state=80)
LR  = LinearRegression()
LR.fit(xx_train, yy_train)
yy_predict = LR.predict(xx_test)
#输出多元回归算法的各个特征的系数矩阵
print("线性回归方程的回归系数矩阵:",LR.coef_)
print("线性回归方程的截距：", LR.intercept_)
#MSE均方误差
print("MSE均方误差:",mean_squared_error(yy_test,yy_predict))
#MAE平均绝对误差
print("MAE平均绝对误差:",mean_absolute_error(yy_test,yy_predict))
#R2 决定系数（拟合优度）
print("R2 决定系数（拟合优度）:",r2_score(yy_test,yy_predict))

#计算出损失函数的值
print("损失函数的值: %.2f" % np.mean((LR.predict(xx_test) - yy_test) ** 2))
#计算预测性能得分
print("预测性能得分: %.2f" % LR.score(xx_test, yy_test))
Coef1_df = pd.DataFrame(LR.coef_,index=columns[0:11],columns=["线性回归系数"])
Coef1_df.sort_values(by="线性回归系数",ascending=False)

#%%
#LassoCV回归

from sklearn.linear_model import LassoCV
L1_CV = LassoCV(cv=10).fit(xx_train, yy_train)
yy1_pred = L1_CV.predict(xx_test)
print("均方误差：",mean_squared_error(yy_test, yy1_pred))
print("LassoCV回归模型的截距：",L1_CV.intercept_)
print("LassoCV回归模型的回归系数：\n",L1_CV.coef_)
#计算出损失函数的值
print("损失函数的值: %.2f" % np.mean((L1_CV.predict(xx_test) - yy_test) ** 2))
#计算预测性能得分
print("预测性能得分: %.2f" % L1_CV.score(xx_test, yy_test))

Coef2_df = pd.DataFrame(L1_CV.coef_,index=columns[0:11],columns=["LassoCV回归系数"])
Coef2_df.sort_values(by="LassoCV回归系数",ascending=False)

#%%
#Ridge regression 岭回归
from sklearn.linear_model import RidgeCV
R1_CV = RidgeCV(cv=10).fit(xx_train, yy_train)
yyR1_pred = R1_CV.predict(xx_test)
print("均方误差：",mean_squared_error(yy_test, yyR1_pred))
print("岭回归模型的截距：",R1_CV.intercept_)
print("岭回归模型的回归系数：\n",R1_CV.coef_)
#计算出损失函数的值
print("损失函数的值: %.2f" % np.mean((R1_CV.predict(xx_test) - yy_test) ** 2))
#计算预测性能得分
print("预测性能得分: %.2f" % R1_CV.score(xx_test, yy_test))

Coef3_df = pd.DataFrame(R1_CV.coef_,index=columns[0:11],columns=["岭回归系数"])
Coef3_df.sort_values(by="岭回归系数",ascending=False)


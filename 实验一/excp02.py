# 导入第三方模块
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
# 导入数据集
income = pd.read_csv(r'dataset+codes/line-ext.csv')

# 利用收入数据集，构建回归模型
fit = sm.formula.ols("Salary ~ YearsExperience", data=income).fit()

# 创建线性回归对象
model = LinearRegression()
# 利用收入数据集，拟合回归模型
model.fit(income.YearsExperience.values.reshape(-1, 1), income.Salary)
# 返回模型的参数值
print('回归参数 w 的值：', model.coef_[0])
print('回归参数 b 的值：', model.intercept_)

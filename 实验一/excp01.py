import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# 导入数据集
income = pd.read_csv(r'dataset+codes/line-ext.csv')
# 绘制散点图
sns.lmplot(x = 'YearsExperience', y = 'Salary', data = income, ci = None)
# 显示图形
plt.show()
# 样本量
n = income.shape[0]
# 计算自变量、因变量、自变量平方、自变量与因变量乘积的和
sum_x = income.YearsExperience.sum()
sum_y = income.Salary.sum()
sum_x2 = income.YearsExperience.pow(2).sum()
xy = income.YearsExperience * income.Salary
sum_xy = xy.sum()
# 根据公式计算回归模型的参数
w = (sum_xy-sum_x*sum_y/n)/(sum_x2-sum_x**2/n)
b = income.Salary.mean()-w*income.YearsExperience.mean()
# 打印出计算结果
print('回归参数w 的值：',w)
print('回归参数b 的值：',b)
print('模型表达式：f(x)=',w,'x+',b)
# 打印出均方误差
# 计算预测值
y_pred = w * income.YearsExperience + b
# 计算均方误差
mse = ((y_pred - income.Salary) ** 2).sum() / n
# 打印出均方误差
print('均方误差：', mse)
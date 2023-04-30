from sklearn import model_selection
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
#横向最多显示多少个字符， 一般80不适合横向的屏幕，平时多用200
pd.set_option('display.width', 200)
#显示所有列
pd.set_option('display.max_columns',None)
#显示所有行
pd.set_option('display.max_rows', None)
#导入数据
Profit = pd.read_excel(r'./dataset+codes/Predict to Profit.xlsx')
#生成由State变量衍生的哑变量
dummies = pd.get_dummies(Profit.State)
print(dummies)
#将哑变量与原始数据集水平合并
Profit_New = pd.concat([Profit, dummies], axis=1)
print('Profit_New:\n',Profit_New)
#删除State变量和New York变量（因为State变量已被分解为哑变量，New York变量需要作为参照组）
Profit_New.drop(labels=['State', 'New York'], axis=1, inplace=True)
#将数据集拆分为训练集和测试集
train, test = model_selection.train_test_split(Profit_New, test_size=0.2, random_state=1234)
#根据train数据集建模
model = sm.formula.ols('Profit ~ RD_Spend + Administration + Marketing_Spend + Florida + California', data=train).fit()
print('模型的偏回归系数分别为：\n', model.params)
#删除test数据集中的Profit变量，用剩下的自变量进行预测
test_X = test.drop(labels='Profit', axis=1)
pred = model.predict(exog=test_X)
print('对比预测值和实际值的差异：\n', pd.DataFrame({'Prediction': pred, 'Real':test.Profit}))


# Scatter plot of predicted values versus real values
plt.scatter(x=pred, y=test.Profit)
plt.xlabel('Predicted Values')
plt.ylabel('Real Values')
plt.title('Predicted vs Real Values')

# Add the regression line
x = pred
y = test.Profit
m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b, color='red')

plt.show()
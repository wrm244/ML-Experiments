import matplotlib.pyplot as plt
import pandas as pd
from sklearn import ensemble, metrics, model_selection, tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score

# 读入数据
fr = open("dataset+codes/glass-lenses.txt")
lenses = [inst.strip().split("\t") for inst in fr.readlines()]
lensesLabels = ["age", "prescript", "astigmatic", "tearRate", "type"]
lens = pd.DataFrame.from_records(lenses, columns=lensesLabels)

# 哑变量处理
dummy = pd.get_dummies(lens[["age", "prescript", "astigmatic", "tearRate"]])
# 水平合并数据集和哑变量的数据集
lens = pd.concat([lens, dummy], axis=1)
# 删除原始的 age, prescript, astigmatic 和 tearRate 变量
lens.drop(["age", "prescript", "astigmatic", "tearRate"], inplace=True, axis=1)
lens.head()

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    lens.loc[:, "age_pre":"tearRate_reduced"],
    lens.type,
    test_size=0.25,
    random_state=1234,
)

# 创建一个决策树分类器
clf = DecisionTreeClassifier()

# 将分类器拟合到训练数据上
clf.fit(X_train, y_train)

# 对测试数据进行分类预测
y_pred = clf.predict(X_test)

'''
在这段代码中，我们首先从scikit-learn的tree模块中导入plot_tree函数。然后，我们创建一个20x10大小的matplotlib画布，并在上面调用plot_tree函数。plot_tree函数的参数包括：

clf：训练好的分类器。
filled：是否使用颜色填充节点。
rounded：是否使用圆角矩形节点。
feature_names：训练数据集的特征名字列表。
class_names：训练数据集的目标变量名字列表。
fontsize：节点字体的大小。
最后，我们使用plt.show()函数将画布显示出来。

注意，如果您的决策树非常大，可能会因为画布太小而无法完全可视化所有节点。在这种情况下，您可以调整画布的大小或使用其他可视化工具来查看决策树。
'''
plt.figure(figsize=(20,10))
plot_tree(clf, filled=True, rounded=True, 
          feature_names=X_train.columns, 
          class_names=y_train.unique(), 
          fontsize=10)
#plt.show()

'''
在这段代码中，我们首先从scikit-learn的metrics模块中导入accuracy_score函数。然后，我们使用训练好的分类器clf在测试集X_test上进行预测，并将预测结果存储在变量y_pred中。接下来，我们调用accuracy_score函数来计算模型在测试集上的预测准确率，并将结果存储在变量accuracy中。最后，我们使用format函数将预测准确率以百分比形式输出，并保留两位小数。
'''
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("预测准确率: {:.2%}".format(accuracy))
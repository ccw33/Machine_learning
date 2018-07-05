#encoding:utf-8
from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

# 加载数据集
california_housing_dataframe = pd.read_csv("file/california_housing_train.csv", sep=",")
#我们将对数据进行随机化处理，以确保不会出现任何病态排序结果（可能会损害随机梯度下降法的效果）。
# 此外，我们会将 median_house_value 调整为以千为单位，这样，模型就能够以常用范围内的学习速率较为轻松地学习这些数据。
california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe["median_house_value"] /= 1000.0
# print(california_housing_dataframe)

# 先了解一下数据
print(california_housing_dataframe.describe())

# 构建模型：
#   标签：median_house_value ， 特征：total_rooms
#   使用的模型是线性回归，会调用TensorFlow Estimator API 提供的 LinearRegressor 接口。
#       此 API 负责处理大量低级别模型搭建工作，并会提供执行模型训练、评估和推理的便利方法。

# 为了将我们的训练数据导入 TensorFlow，我们需要指定每个特征包含的数据类型。在本练习及今后的练习中，我们主要会使用以下两类数据：
#   分类数据：一种文字数据。在本练习中，我们的住房数据集不包含任何分类特征，但您可能会看到的示例包括家居风格以及房地产广告词。
#   数值数据：一种数字（整数或浮点数）数据以及您希望视为数字的数据。有时您可能会希望将数值数据（例如邮政编码）视为分类数据（我们将在稍后的部分对此进行详细说明）。

# ----------- 第 1 步：定义特征并配置特征列
# Define the input feature: total_rooms.
# 拿出特征咧数据
my_feature = california_housing_dataframe[["total_rooms"]]
# Configure a numeric feature column for total_rooms.
# 拿出特征列名称
feature_columns = [tf.feature_column.numeric_column("total_rooms")]

# ----------- 第 2 步：定义目标
# Define the label.
# 拿出便签列数据
targets = california_housing_dataframe["median_house_value"]

# ----------- 第 3 步：配置 LinearRegressor
#   使用 LinearRegressor 配置线性回归模型，
#   并使用 GradientDescentOptimizer（它会实现小批量随机梯度下降法 (SGD)）训练该模型。learning_rate 参数可控制梯度步长的大小。
#   为了安全起见，我们还会通过 clip_gradients_by_norm 将梯度裁剪应用到我们的GradientDescentOptimizer。梯度裁剪可确保梯度大小在训练期间不会变得过大，梯度过大会导致梯度下降法失败。
# Use gradient descent as the optimizer for training the model.
# 定义梯度下贱调整器，会同时定义好学习速率(其实就是步长，是w和b每次变化的跨度)，clip_gradients_by_norm是为了防止步长过长
my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)#(TODO 5.0不知道什么意思)

# Configure the linear regression model with our feature columns and optimizer.
# Set a learning rate of 0.0000001 for Gradient Descent.
# 定义线性回归方程，同时会声明特征列和梯度下降调整器（此时的线性回归模型是学习前的模型，没有成型）
linear_regressor = tf.estimator.LinearRegressor(
    feature_columns=feature_columns,
    optimizer=my_optimizer
)


# ----------- 第 4 步：定义输入函数
# 要将加利福尼亚州住房数据导入 LinearRegressor，我们需要定义一个输入函数，让它告诉 TensorFlow 如何对数据进行预处理，以及在模型训练期间如何批处理、随机处理和重复数据。
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    '''
    首先，我们将 Pandas 特征数据转换成 NumPy 数组字典。然后，我们可以使用 TensorFlow Dataset API 根据我们的数据构建 Dataset 对象，并将数据拆分成大小为 batch_size 的多批数据，以按照指定周期数 (num_epochs) 进行重复。

    注意：如果将默认值 num_epochs=None 传递到 repeat()，输入数据会无限期重复。

    然后，如果 shuffle 设置为 True，则我们会对数据进行随机处理，以便数据在训练期间以随机方式传递到模型。buffer_size 参数会指定 shuffle 将从中随机抽样的数据集的大小。

    最后，输入函数会为该数据集构建一个迭代器，并向 LinearRegressor 返回下一批数据。
    :param features: pandas DataFrame of features 特征数据
    :param targets: pandas DataFrame of targets 对应的目标
    :param batch_size:  Size of batches to be passed to the model  每次计算损失时所使用的数据量
    :param shuffle:True or False. Whether to shuffle the data. 书否大蓝数据模型，如果打断，则会把传递给模型的数据打乱
    :param num_epochs:Number of epochs for which data should be repeated. None = repeat indefinitely (TODO 不懂)
    :return:
    '''

    # Convert pandas data into a dict of np arrays.
    features = {key: np.array(value) for key, value in dict(features).items()}

    # Construct a dataset, and configure batching/repeating
    # 构造数据集（有标签样本数据集，用于训练模型）
    ds = Dataset.from_tensor_slices((features, targets))  # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)# 是否重复输入数据（TODO 暂时不明白）

    # Shuffle the data, if specified
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)

    # Return the next batch of data
    features, labels = ds.make_one_shot_iterator().get_next() # 返回下一批数据
    return features, labels

# ----------- 第 5 步：训练模型
_ = linear_regressor.train(
    input_fn = lambda:my_input_fn(my_feature, targets),
    steps=100
)

# ----------- 第 6 步：评估模型
# 我们基于该训练数据做一次预测，看看我们的模型在训练期间与这些数据的拟合情况。
#
# 注意：训练误差可以衡量您的模型与训练数据的拟合情况，但并不能衡量模型泛化到新数据的效果。在后面的练习中，您将探索如何拆分数据以评估模型的泛化能力。

# Create an input function for predictions.
# Note: Since we're making just one prediction for each example, we don't
# need to repeat or shuffle the data here.
# 为了查看得出的模型是否正确，我们需要查看使用模型计算出来的结果跟实际结果的差距
# 用训练完的模型来预测结果（使用原有的指标），同样需要用定义好的my_input_fn方法来转换数据
prediction_input_fn =lambda: my_input_fn(my_feature, targets, num_epochs=1, shuffle=False)

# Call predict() on the linear_regressor to make predictions.
predictions = linear_regressor.predict(input_fn=prediction_input_fn)# 得出模型计算出来的结果集

# Format predictions as a NumPy array, so we can calculate error metrics.
predictions = np.array([item['predictions'][0] for item in predictions])

# Print Mean Squared Error and Root Mean Squared Error.
mean_squared_error = metrics.mean_squared_error(predictions, targets) # 计算出均方损失
root_mean_squared_error = math.sqrt(mean_squared_error) # 计算出根均方损失
print("Mean Squared Error (on training data): %0.3f" % mean_squared_error)
print("Root Mean Squared Error (on training data): %0.3f" % root_mean_squared_error)


# 由于均方误差 (MSE) 很难解读，因此我们经常查看的是均方根误差 (RMSE)。RMSE 的一个很好的特性是，它可以在与原目标相同的规模下解读。
# 我们来比较一下 RMSE（根均方损失） 与目标最大值和最小值的差值：
min_house_value = california_housing_dataframe["median_house_value"].min()
max_house_value = california_housing_dataframe["median_house_value"].max()
min_max_difference = max_house_value - min_house_value

print("Min. Median House Value: %0.3f" % min_house_value)
print("Max. Median House Value: %0.3f" % max_house_value)
print("Difference between Min. and Max.: %0.3f" % min_max_difference)
print("Root Mean Squared Error: %0.3f" % root_mean_squared_error)


# 我们的误差跨越目标值的近一半范围，可以进一步缩小误差吗？
#
# 这是每个模型开发者都会烦恼的问题。我们来制定一些基本策略，以降低模型误差。
#
# 首先，我们可以了解一下根据总体摘要统计信息，预测和目标的符合情况

# 把使用模型计算出来的数据和实际数据进行比较（使用panda）
# 先把两个数据放在一个数据及中
calibration_data = pd.DataFrame()
calibration_data["predictions"] = pd.Series(predictions)
calibration_data["targets"] = pd.Series(targets)
print(calibration_data.describe())

# 我们还可以将数据和学到的线可视化。我们已经知道，单个特征的线性回归可绘制成一条将输入 x 映射到输出 y 的线。
#
# 首先，我们将获得均匀分布的随机数据样本，以便绘制可辨的散点图。

# 然后实际样本中的300个样本
sample = california_housing_dataframe.sample(n=300)

# 然后，我们根据模型的偏差项和特征权重绘制学到的线，并绘制散点图。该线会以红色显示。
# Get the min and max total_rooms values.
# 获取样本中最大和最小x
x_0 = sample["total_rooms"].min()
x_1 = sample["total_rooms"].max()

# Retrieve the final weight and bias generated during training.
# 获取模型的截距和斜率（因为是线性回归模型）
weight = linear_regressor.get_variable_value('linear/linear_model/total_rooms/weights')[0]
bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

# Get the predicted median_house_values for the min and max total_rooms values.
# 因为是线性回归，可以算出x_0 和 x_1 对应的y
y_0 = weight * x_0 + bias
y_1 = weight * x_1 + bias

# Plot our regression line from (x_0, y_0) to (x_1, y_1).
# 画出模型
plt.plot([x_0, x_1], [y_0, y_1], c='r')

# Label the graph axes.
# 给图形的x、y轴贴上标签
plt.ylabel("median_house_value")
plt.xlabel("total_rooms")

# Plot a scatter plot from our data sample.
# 画图（实际样本的散点图）
plt.scatter(sample["total_rooms"], sample["median_house_value"])

# Display graph.
plt.show()

# 得出的结果是相差巨大




# 调整模型超参数

b = 1
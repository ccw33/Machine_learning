#encoding:utf-8
import pandas as pd
# print(pd.__version__)
#创建列，并和数据一起放入DataFrame
city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])
cities=pd.DataFrame({ 'City name': city_names, 'Population': population })


# # 不自己创建，直接冲csv文件读取
# # california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")
# california_housing_dataframe = pd.read_csv("file/california_housing_train.csv", sep=",")
# print(california_housing_dataframe.describe())#展示数据
# print(california_housing_dataframe.head())#展示头几个数据
# # california_housing_dataframe.hist('housing_median_age')#绘制图表（有问题）


#访问数据
## 直接向访问list，dict那样访问DataFrame
# print(type(cities['City name']))
# print(cities['City name'])
## DataFarm展示
# print(cities.describe())
# print(cities.head())
# cities.hist('Population')#绘制图表（有问题）
## Series操作
# print(population/1000)
# print(population.apply(lambda val: val > 1000000))
import numpy as np # 直接通过numpy操作
# print(np.log(population))
# #DataFrames 的修改
# cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
# cities['Population density'] = cities['Population'] / cities['Area square miles']
# print(cities)

## 索引
# print(city_names.index)
# print(cities.index)
# 重新排序(只是返回一个重排后的结果，原cities并没有变)
print(cities.reindex([2, 0, 1]))
print(cities.reindex(np.random.permutation(cities.index)))
print(cities.reindex([0, 4, 5, 2])) # 索引超出数据数量也没关系，4和5不在原索引中，会显示NaN
print(cities.reindex([0, 4, 5, 2]))

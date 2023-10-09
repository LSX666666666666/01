import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
data = pd.read_csv('天气-gbk.csv', encoding='GBK')

# 添加筛选条件，只保留北京的数据
data = data[data['城市'] == '北京']

# 数据预处理
data['最高温度'] = data['最高温度'].str.replace('°', '').astype(float)
data['日期'] = pd.to_datetime(data['日期'].str.split(' ', expand=True)[0])
data = data.sort_values('日期')

# 划分训练集和测试集
train = data[data['日期'] < '2021-01-01']
test = data[data['日期'] >= '2021-01-01']

# 数据归一化
#数据归一化的步骤使用了MinMaxScaler将数据特征缩放到0和1之间。训练集和测试集的最高温度值被归一化。
#如果特征没有经过归一化，具有较大尺度的特征可能会主导计算过程，从而影响模型的性能。
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train['最高温度'].values.reshape(-1, 1))
test_scaled = scaler.transform(test['最高温度'].values.reshape(-1, 1))

# 构建数据集
#作用是根据给定的时间窗口大小，将时间序列数据转换为LSTM模型的输入格式。它将时间窗口内连续的观测作为输入特征，
# 下一个观测作为输出标签，生成多个这样的(X, Y)样本。这样，就可以用于训练LSTM模型进行时间序列预测。
#构建数据集可以将原始的时间序列数据整理成适合机器学习模型训练的输入和输出形式，引入时间窗口概念，
# 并控制数据集的大小和复杂度，从而提高训练效率和模型的预测能力。

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 10
X_train, y_train = create_dataset(train_scaled, look_back)
X_test, y_test = create_dataset(test_scaled, look_back)

# 重塑输入数据的形状以适应模型 [samples, time steps, features]
#LSTM 模型的输入需要满足 (样本数量, 时间步数, 特征数) 的格式要求。
#其中 X_train.shape[0] 表示训练集的样本数量，X_train.shape[1] 表示每个样本的时间步数。
# 重塑后的结果将赋值给 X_train，即训练集的输入特征
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# 创建并训练LSTM网络
#通过 Sequential() 函数创建了一个序列模型 model。
#model.add() 方法向模型中逐层添加网络层。
#通过 model.add(Dense(1)) 添加了一个全连接层，该层的输出维度为 1。
#使用 model.compile() 方法配置了模型的损失函数和优化器。loss='mean_squared_error' 表示使用均方误差作为损失函数，
# optimizer='adam' 表示使用 Adam 优化器进行参数优化。
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(1, look_back)))  # 增加LSTM层的单元数，并返回序列
model.add(LSTM(50))  # 增加一个新的LSTM层
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=6, batch_size=1, verbose=2)

# 预测
#通过调用 model.predict() 方法对训练集和测试集的输入数据进行预测。
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 反归一化
#得到了经过模型预测并反归一化后的训练集和测试集的预测结果。
# 这样可以将预测结果与原始数据进行对比，评估模型的预测性能和准确度。
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])

# 绘制结果
plt.figure(figsize=(20,10))
plt.plot(train['日期'][look_back+1:], y_train[0], label='Train Actual')
plt.plot(train['日期'][look_back+1:], train_predict[:,0], label='Train Predict')
plt.plot(test['日期'][look_back+1:], y_test[0], label='Test Actual')
plt.plot(test['日期'][look_back+1:], test_predict[:,0], label='Test Predict')
plt.legend()
plt.show()

# 创建2022年和2023年的日期
dates_2022 = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
dates_2023 = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')

# 预测2022年和2023年的数据
future_predict = []
for i in range(len(dates_2022) + len(dates_2023)):
    if i < len(test_scaled) - look_back:
        X_future = np.array([test_scaled[i:i+look_back]])  # 使用过去的实际数据来预测未来的数据
    X_future_reshaped = np.reshape(X_future, (X_future.shape[0], 1, X_future.shape[1]))
    predict = model.predict(X_future_reshaped)
    future_predict.append(predict[0, 0])
    if i >= len(test_scaled) - look_back:
        X_future = np.append(X_future[0, 1:], predict).reshape(1, -1)  # 只有在没有实际数据可用时，才使用预测数据作为输入数据的一部分

# 反归一化
future_predict = scaler.inverse_transform(np.array(future_predict).reshape(-1, 1))

# 获取2022年的真实数据
real_2022 = data[(data['日期'] >= '2022-01-01') & (data['日期'] <= '2022-12-31')]['最高温度'].values[:len(dates_2022)]



# 计算2013年至2022年夏季的平均最高气温
#summer_data = data[(data['日期'].dt.month.isin([1,2,3,4,5,6,7, 8,9,10,11,12])) & (data['日期'].dt.year.between(2013, 2022))]
s1_data = data[(data['日期'].dt.month.isin([1])) & (data['日期'].dt.year.between(2013, 2022))]
s7_data = data[(data['日期'].dt.month.isin([7])) & (data['日期'].dt.year.between(2013, 2022))]
average_s1_temp = s1_data['最高温度'].mean()
average_s7_temp = s7_data['最高温度'].mean()



# 计算2013-2022年每个月的平均最高气温
average_monthly_temp = data[(data['日期'].dt.year.between(2013, 2022))].groupby(data['日期'].dt.month)['最高温度'].mean()

# 绘制2022年的图表
plt.figure(figsize=(20, 10))
plt.plot(dates_2022, future_predict[:len(dates_2022)].flatten(), label='2022 Predicted')
plt.plot(dates_2022, real_2022, label='2022 Actual')
for month, temp in average_monthly_temp.items():
    start_date = pd.Timestamp(year=2022, month=month, day=1)
    end_date = (start_date + pd.offsets.MonthEnd(1))
    plt.hlines(y=temp, xmin=start_date, xmax=end_date, colors='r', linestyles='-', label=f'Average for month {month}')
plt.legend()
plt.title('Beijing-Predicted vs Actual Average Temperature for 2022')
plt.show()

# 绘制2023年的图表
plt.figure(figsize=(20, 10))
plt.plot(dates_2023, future_predict[len(dates_2022):].flatten(), label='2023 Predicted')
for month, temp in average_monthly_temp.items():
    start_date = pd.Timestamp(year=2023, month=month, day=1)
    end_date = (start_date + pd.offsets.MonthEnd(1))
    plt.hlines(y=temp, xmin=start_date, xmax=end_date, colors='r', linestyles='-', label=f'Average for month {month}')
plt.legend()
plt.title('Beijing-Predicted Average Temperature for 2023')
plt.show()

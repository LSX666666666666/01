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
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train['最高温度'].values.reshape(-1, 1))
test_scaled = scaler.transform(test['最高温度'].values.reshape(-1, 1))

# 构建数据集
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
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# 创建并训练LSTM网络
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(1, look_back)))  # 增加LSTM层的单元数，并返回序列
model.add(LSTM(50))  # 增加一个新的LSTM层
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=6, batch_size=1, verbose=2)

# 预测
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 反归一化
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
# 在图表中添加平均最高气温的直线
plt.figure(figsize=(20, 10))
plt.plot(dates_2022, future_predict[:len(dates_2022)].flatten(), label='2022 Predicted')
plt.plot(dates_2022, real_2022, label='2022 Actual')
#plt.axhline(y=average_s1_temp, color='r', linestyle='-', label='2013-2022  Average')

plt.legend()
plt.title('Beijing-Predicted vs Actual Average Temperature for 2022')
plt.show()

plt.figure(figsize=(20, 10))
plt.plot(dates_2023, future_predict[len(dates_2022):].flatten(), label='2023 Predicted')
#plt.axhline(y=average_s1_temp, color='r', linestyle='-', label='2013-2022  Average')

plt.legend()
plt.title('Beijing-Predicted Average Temperature for 2023')
plt.show()
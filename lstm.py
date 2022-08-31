import data_load as dl
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


# 数组nan插值
def fill_ndarray(array):
    for i in range(array.shape[1]):
        temp_col = array[:, i]
        # 判断当前列中是否含nan值
        nan_num = np.count_nonzero(temp_col != temp_col)
        if nan_num != 0:
            temp_not_nan_col = temp_col[temp_col == temp_col]
            # 用0填充nan所在位置
            temp_col[np.isnan(temp_col)] = 0
    return array


# 数据标准化
def data_standardization(data):
    dataset = data.values
    data_mean = dataset[:].mean(axis=0)
    data_std = dataset[:].std(axis=0)
    dataset = (dataset - data_mean) / data_std
    return dataset, data_std, data_mean


# 数据还原
def data_reduct(data, data_std, data_mean):
    data = (data * data_std) + data_mean
    return data


# 绘制loss曲线
def plot_loss(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()
    plt.show()
    return


# 特征和标签切片
def data_slice(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []
    start_index = start_index + history_size

    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        data.append(dataset[indices])
        labels.append(dataset[i:i + target_size])

    return np.array(data), np.array(labels)


def multivariate_model(dataset, x_train, y_train, x_val, y_val):
    # 设置种子以确保可重复性。
    tf.random.set_seed(10)

    # 使用tf.data来随机整理，批处理和缓存数据集
    # BATCH_SIZE 的设置参考https://blog.csdn.net/hesongzefairy/article/details/105159892
    BATCH_SIZE = 32

    # 隐藏层神经元数量，理论上数量越多，预测效果越好
    # RNN的隐藏层也可以叫循环核，简单来说循环核循环的次数叫时间步，这里设置为20，循环核的个数就是隐藏层层数。
    # 循环核可以有两个输入（来自样本的输入x、来自上一时间步的激活值a）和两个输出（输出至下一层的激活值h、输出至本循环核下一时间步的激活值a）
    # 参考：https://blog.csdn.net/super67269/article/details/126282842
    # 参考：https://blog.csdn.net/ygfrancois/article/details/90270492?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-1-90270492-blog-121186066.pc_relevant_vip_default&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-1-90270492-blog-121186066.pc_relevant_vip_default&utm_relevant_index=2
    NUM_NEURONS = 256
    lstm_model = tf.keras.models.Sequential()

    # 第一层神经元
    lstm_model.add(tf.keras.layers.LSTM(NUM_NEURONS,
                                        return_sequences=True,
                                        input_shape=x_train.shape[-2:]))
    # 第二层神经元
    lstm_model.add(tf.keras.layers.LSTM(128, activation='relu'))

    # 输出为n*features个维度，需要n个时间步长，就需要n*features个维度
    lstm_model.add(tf.keras.layers.Dense(y_train.shape[1] * dataset.shape[1]))
    lstm_model.add(tf.keras.layers.Reshape((y_train.shape[1], dataset.shape[1])))

    # 优化器和损失函数
    lstm_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')

    # 迭代次数
    EPOCHS = 17

    # 拟合模型
    model_fit = lstm_model.fit(x_train, y_train,
                               epochs=EPOCHS,
                               batch_size=BATCH_SIZE,
                               validation_data=(x_val, y_val),
                               verbose=1)

    # 打印loss曲线
    plot_loss(model_fit,
              'Training and validation loss')

    return lstm_model


if __name__ == '__main__':
    # 设置训练集大小
    TRAIN_SPLIT = 212

    # 读取文件
    city = 'beijing'
    data = dl.read_csv_data('Date', f'dataset/{city}.csv')
    data = dl.data_wash(data)

    # 用来正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 用来正常显示负号
    plt.rcParams['axes.unicode_minus'] = False

    # 特征选择
    features = ['Ave.T', 'Max.T', 'Min.T', 'Prec(mm)', 'Press(Hpa)', 'Wind dir', 'Wind sp (km/h)']
    data = data[features]
    data.head()

    # 数据标准化
    dataset, data_std, data_mean = data_standardization(data)
    print("平均值和标准差是", data_std, data_mean)

    # 特征和标签切片
    past_history = 20
    future_target = 7
    x_train, y_train = data_slice(dataset, 0, TRAIN_SPLIT,
                                  past_history,
                                  future_target)
    x_val, y_val = data_slice(dataset, TRAIN_SPLIT - past_history, None,
                              past_history,
                              future_target)

    # 模型拟合
    model = multivariate_model(dataset, x_train, y_train, x_val, y_val)

    # 模型保存
    model.save(f'model/{city}7.h5')

    # 模型调用
    model = tf.keras.models.load_model(f'model/{city}7.h5')

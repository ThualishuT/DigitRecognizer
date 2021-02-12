"""这是一个利用人工神经网络实现手写数字识别的Python程序。"""

import numpy
import scipy.special
import scipy.ndimage


class NeuralNetwork:
    """人工神经网络"""

    def __init__(self, __input_nodes, __hidden_nodes, __output_nodes, __learning_rate):
        """初始化神经网络。

        i_nodes     输入层节点数
        h_nodes     隐藏层节点数
        o_nodes     输出层节点数
        lr          学习率
        w_ih        输入层与隐藏层之间的链接权重矩阵
        w_ho        隐藏层与输出层之间的链接权重矩阵
        """

        self.i_nodes = __input_nodes
        self.h_nodes = __hidden_nodes
        self.o_nodes = __output_nodes
        self.lr = __learning_rate

        # 使用正态概率分布采样初始权重，平均值为0.0，标准差为(传入链接数目)^(-0.5)。
        self.w_ih = numpy.random.normal(0.0, pow(self.h_nodes, -0.5), (self.h_nodes, self.i_nodes))
        self.w_ho = numpy.random.normal(0.0, pow(self.o_nodes, -0.5), (self.o_nodes, self.h_nodes))

        self.activation_function = lambda x: scipy.special.expit(x)  # 定义激活函数

    def query(self, inputs_list):
        """查询神经网络中的信号。"""

        __inputs = numpy.array(inputs_list, ndmin=2).T  # 将输入列表转换为二维数组

        hidden_inputs = numpy.dot(self.w_ih, __inputs)  # 将输入数据进行计算得到隐藏层输入信号
        hidden_outputs = self.activation_function(hidden_inputs)  # 计算隐藏层输出信号

        final_inputs = numpy.dot(self.w_ho, hidden_outputs)  # 将隐藏层输出信号进行计算得到输出层输入信号
        final_outputs = self.activation_function(final_inputs)  # 计算神经网络最终输出信号

        return [__inputs, hidden_outputs, final_outputs]

    def train(self, inputs_list, targets_list):
        """训练神经网络。"""

        # 调用query方法，获取训练过程中需要的信号值。
        signals = self.query(inputs_list)
        __inputs, hidden_outputs, final_outputs = signals

        __targets = numpy.array(targets_list, ndmin=2).T  # 将目标值列表转换为二维数组

        output_errors = __targets - final_outputs  # 计算预期目标值与计算值的误差
        hidden_errors = numpy.dot(self.w_ho.T, output_errors)  # 计算隐藏层节点反向传播的误差

        # 更新权重
        self.w_ho += self.lr * numpy.dot(output_errors * final_outputs * (1 - output_errors),
                                         numpy.transpose(hidden_outputs))
        self.w_ih += self.lr * numpy.dot(hidden_errors * hidden_outputs * (1 - hidden_outputs),
                                         numpy.transpose(__inputs))


if __name__ == '__main__':
    input_nodes = 784      # 输入层节点数
    hidden_nodes = 200     # 隐藏层节点数
    output_nodes = 10      # 输出层节点数
    learning_rate = 0.1    # 学习率

    n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)  # 创建用于识别手写数字的人工神经网络实例

    # 训练神经网络

    with open("数据集/train_kaggle.csv", 'r') as training_data_file:
        epochs_1 = 5  # 第一次训练世代数

        for e in range(epochs_1):
            for record in training_data_file:
                values_list = record.split(',')

                inputs = numpy.asfarray(values_list[1:]) / 255.0 * 0.99 + 0.01  # 格式化输入值

                # 格式化输出值
                targets = numpy.zeros(output_nodes) + 0.01
                targets[int(values_list[0])] = 0.99

                n.train(inputs, targets)  # 训练

        # 将输入图像按顺时针、逆时针方向各旋转10°，并适当调整神经网络参数后，再次训练。
        epochs_2 = 10  # 第二次训练世代数

        n.lr = 0.01  # 调整学习率

        def rotate_and_train(__inputs, angle, __targets):
            """将输入图像旋转指定角度后训练。"""
            # 将输入图像旋转指定角度。
            rotated_inputs = \
                scipy.ndimage.interpolation.rotate(__inputs.reshape(28, 28), angle, cval=0.01, reshape=False)

            n.train(rotated_inputs, __targets)  # 训练

        for e in range(epochs_2):
            for record in training_data_file:
                values_list = record.split(',')

                inputs = numpy.asfarray(values_list[1:]) / 255.0 * 0.99 + 0.01  # 格式化输入值

                # 格式化输出值
                targets = numpy.zeros(output_nodes) + 0.01
                targets[int(values_list[0])] = 0.99

                # 将输入图像旋转±10°后分别训练。
                rotate_and_train(inputs, -10, targets)
                rotate_and_train(inputs, 10, targets)

    # 测试神经网络，得出识别结果，写入一个csv文件中。

    with open("数据集/test_kaggle.csv", 'r') as test_data_file, open("数据集/predictions by cxy.csv", 'a') as predictions:
        i = 1  # 识别结果的序号

        for record in test_data_file:
            values_list = record.split(',')

            inputs = numpy.asfarray(values_list) / 255.0 * 0.99 + 0.01  # 格式化输入值
            outputs = n.query(inputs)[2]  # 获取神经网络的识别结果
            output_label = numpy.argmax(outputs)  # 输出标签为输出矩阵中最大值的索引
            predictions.write(f"{i},{output_label}\n")  # 记录识别结果
            i += 1  # 序号加1

# DigitRecognizer
①这是一个由人工神经网络实现的手写数字识别程序，主要参考了《Python神经网络编程》一书，根据实际需求作适当修改、优化，形成了本项目中的三份程序，一份用于调参、提高识别率，另两份用于识别Kaggle上提供的测试集，输出识别结果。  
②学习编写本项目的过程总体比较顺利，没有太多难点，只要能够耐心读完参考书籍，就不会有太大问题。但是编写过程中依然是有值得分享的地方的。首先是关于编码规范，经过前两轮的学习，我已经对Python基础语法及少量中级语法有了比较熟练的掌握，但是规范性方面未做太多研究，我也注意到，之前的项目中IDE总是会报出数十处弱警告，因此，我抽空仔细研读了相关规范，并且运用到了此次的项目中，使编写的代码规范、可读性好。其次是关于代码冗余的处理，我在研读参考书的过程中注意到，作者提供的代码有不少冗余部分，可以通过代码复用进一步优化，比如：
```bash
    def query(self, inputs_list):
        """查询神经网络中的信号。"""

        __inputs = numpy.array(inputs_list, ndmin=2).T  # 将输入列表转换为二维数组

        hidden_inputs = numpy.dot(self.w_ih, __inputs)  # 将输入数据进行计算得到隐藏层输入信号
        hidden_outputs = self.activation_function(hidden_inputs)  # 计算隐藏层输出信号
        ......
        return final_outputs

    def train(self, inputs_list, targets_list):
        """训练神经网络。"""

        __inputs = numpy.array(inputs_list, ndmin=2).T  # 将输入列表转换为二维数组

        hidden_inputs = numpy.dot(self.w_ih, __inputs)  # 将输入数据进行计算得到隐藏层输入信号
        hidden_outputs = self.activation_function(hidden_inputs)  # 计算隐藏层输出信号
        ......
```
经过优化可以压缩为：
```bash
    def query(self, inputs_list):
        """查询神经网络中的信号。"""

        __inputs = numpy.array(inputs_list, ndmin=2).T  # 将输入列表转换为二维数组

        hidden_inputs = numpy.dot(self.w_ih, __inputs)  # 将输入数据进行计算得到隐藏层输入信号
        hidden_outputs = self.activation_function(hidden_inputs)  # 计算隐藏层输出信号
        ......
        return [__inputs, hidden_outputs, final_outputs]

    def train(self, inputs_list, targets_list):
        """训练神经网络。"""

        # 调用query方法，获取训练过程中需要的信号值。
        signals = self.query(inputs_list)
        __inputs, hidden_outputs, final_outputs = signals
        ......
```
很好地解决了冗余问题。最后也是本项目中最难的一点就是调参，我运用控制变量法设计了多次试验，运用程序将试验数据自动写入本地的记事本中，再对所得数据进行可视化等处理，最终才确定了最优的参数，这种调参方式值得记录。  
③本项目从开始学习相关资料到完全完成最终版本的代码编写历时2周左右，可以说还是一个比较有研究价值的项目，我从中受益匪浅，代码能力得到了不小的提升，希望能够再接再厉，更上一层楼。

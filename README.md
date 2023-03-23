这里是无人机识别与检测技术报告的代码部分

训练步骤如下：
将数据集放入Origin_dataset文件夹中，运行data_prepossess.ipynb的所有步骤，这部分将把数据集转换为适合输入YOLOv7架构的形式。
之后在终端中输入 python train.py 即可开始训练，训练结果在runs/train文件夹中。

最新的训练版本在runs/train/exp7中，之后的测试也将使用这里的参数。

测试步骤如下：
对单个视频要输出结果和可视化图像，先要运行moveTraindata.ipynb，将数据转入VOC文件夹，
之后在终端运行 python test.py 即开始测试，测试结果在runs/test文件夹中。
对所有视频输出精确度 运行eval.py 文件

输出步骤如下：
运行testall.py即可输出要求的包含json串的txt文件
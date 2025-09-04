import struct
import numpy as np
import gzip

def add(x, y):
    """一个简单的add函数，以便熟悉自动测试（pytest）

    Args:
        x (Python数字 或者 numpy array)
        y (Python数字 或者 numpy array)

    Return:
        x+y的和
    """
    ### 你的代码开始
    return x + y
    ### 你的代码结束


def parse_mnist(image_filename, label_filename):
    """ 读取 MNIST 格式的图像和标签文件。有关文件格式的说明，请参阅此页面：
    http://yann.lecun.com/exdb/mnist/。

    参数：
    image_filename（字符串）：MNIST 格式的 gzip 压缩图像文件的名称
    label_filename（字符串）：MNIST 格式的 gzip 压缩标签文件的名称

    返回：
    tuple (X,y)：
    x (numpy.ndarray[np.float32])：包含已加载数据的二维 numpy 数组。数据的维度应为
    (num_examples x input_dim)，其中“input_dim”是数据的完整维度，例如，由于 MNIST 图像为 28x28，因此
    input_dim 为 784。值应为 np.float32 类型，并且数据应被归一化为最小值为 0.0，
    最大值为 1.0 （即将原始值 0 缩放为 0.0，将 255 缩放为 1.0）。

    y (numpy.ndarray[dtype=np.uint8])：包含示例标签的一维 NumPy 数组。值应为 np.uint8 类型，对于 MNIST，将包含 0-9 的值。
    """

    ### 你的代码开始
    # 读取图像文件

    # 训练文件格式：
    # 偏移量 0：魔数（4字节，MSB first），固定为 0x00000803 (2051)
            # 0x0000：保留部分
            # 0x08: 表示数据类型（08 代表无符号字节）
            # 0x03: 表示维度数量（03 代表3维：

    # 偏移量 4：图像数量（4字节）
    # 偏移量 8：行数（4字节）
    # 偏移量 12：列数（4字节）
    # 偏移量 16：开始是图像数据，每个像素是1字节的无符号整数（0-255）

    with gzip.open(image_filename, 'rb') as f:
        # 读取文件头信息
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        # 读取所有图像数据
        image_data = f.read()
        # 转换为numpy数组并归一化到[0,1]
        X = np.frombuffer(image_data, dtype=np.uint8).astype(np.float32)
        X = X.reshape(num_images, rows * cols) / 255.0

    # 标签文件格式：
    # 偏移量 0：魔数（4字节，MSB first），固定为 0x00000801 (2049)
        # 0x0000：保留部分
        # 0x08: 表示数据类型（08 代表无符号字节）
        # 0x01: 表示维度数量（01 代表1维：
    # 偏移量 4：标签数量（4字节）
    # 偏移量 8：开始是标签数据，每个标签是1字节的无符号整数（0-9）
    with gzip.open(label_filename, 'rb') as f:
        # 读取文件头信息
        magic, num_labels = struct.unpack('>II', f.read(8))
        # 读取所有标签数据
        label_data = f.read()
        # 转换为numpy数组
        y = np.frombuffer(label_data, dtype=np.uint8)

    return X, y
    ### 你的代码结束


def softmax_loss(Z, y):
    """ 返回 softmax 损失。

    参数：
    z (np.ndarray[np.float32])：形状为 (batch_size, num_classes) 的二维 NumPy 数组，
    包含每个类别的 对数概率 预测值 （softmax函数激活之前的值）。

    y (np.ndarray[np.uint8])：形状为 (batch_size, ) 的一维 NumPy 数组，包含每个样本的真实标签。

    返回：
    样本的平均 softmax 损失。
    """

    ### 你的代码开始
    m, n_classes = Z.shape

    zy = Z[np.arange(m), y]
    log_sum = np.log(np.sum(np.exp(Z), axis=1))

    return np.sum(log_sum - zy) / m
    ### 你的代码结束


def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ 使用步长 lr 和指定的批次大小，对数据运行单轮 小批量梯度下降 进行 softmax 回归。此函数会修改
    θ 矩阵，并迭代 X 中的批次，但不对顺序进行随机化。

    参数：
    X (np.ndarray[np.float32])：大小为
    (num_examples x input_dim) 的二维输入数组。
    y (np.ndarray[np.uint8])：大小为 (num_examples,) 的一维类别标签数组。
    theta (np.ndarrray[np.float32])：softmax 回归的二维数组参数，形状为 (input_dim, num_classes)。
    lr (float)：SGD 的步长（学习率）。
    batch (int)：SGD 小批次的大小。

    返回：
    无
    """

    ### 你的代码开始
    k = theta.shape[1]
    m = X.shape[0]
    for i in range(0, m, batch):
        X_batch = X[i : i + batch]
        y_batch = y[i : i + batch]
        m = X_batch.shape[0]

        Z_batch = np.exp(np.dot(X_batch, theta))
        Z_batch /= Z_batch.sum(axis=1, keepdims=True)
        batch_gradient = np.dot(X_batch.T / m, Z_batch - (y_batch.reshape((-1,1)) == np.arange(k)))
        theta -= lr * batch_gradient

    ### 你的代码结束

def relu(x):
    return np.maximum(x, 0)

def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ 对由权重 W1 和 W2 定义的双层神经网络（无偏差项）运行一个 小批量梯度下降 迭代轮次：
    logits = ReLU(X * W1) * W2
    该函数应使用步长 lr 和指定的批次大小（并且
    同样，不随机化 X 的顺序）。它应修改 W1 和 W2 矩阵。

    参数：
    X (np.ndarray[np.float32])：大小为 (num_examples x input_dim) 的二维输入数组。
    y (np.ndarray[np.uint8])：大小为 (num_examples,) 的一维类别标签数组。
    W1 (np.ndarray[np.float32])：第一层权重的二维数组，形状为(input_dim, hidden_dim)
    W2 (np.ndarray[np.float32])：第二层权重的二维数组。形状
    (hidden_dim, num_classes)
    lr (float)：SGD 的步长（学习率）
    batch (int)：SGD 小批次的大小

    返回：
    无
    """

    ### 你的代码开始
    k = W2.shape[1]
    eye_k = np.eye(k)
    total_sample = X.shape[0]
    for i in range(0, total_sample, batch):
        X_batch = X[i : i + batch]
        y_batch = y[i : i + batch]
        m = X_batch.shape[0]

        # forward pass
        Z1 = relu(np.dot(X_batch, W1))
        Z2 = np.exp(np.dot(Z1, W2))
        Z2 /= Z2.sum(axis=1, keepdims=True)

        # backward pass
        G2 = Z2 - eye_k[y_batch]
        G1 = (Z1 > 0) * (np.dot(G2, W2.T))

        dW2 = np.dot(Z1.T, G2) / m
        dW1 = np.dot(X_batch.T, G1) / m

        W2 -= lr * dW2
        W1 -= lr * dW1
    ### 你的代码结束



### 下面的代码不用编辑，只是用来展示功能的

def loss_err(h,y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100):
    """ 示例函数，用softmax回归训练 """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """ 示例函数，训练神经网络 """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))



if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.1)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr = 0.2)
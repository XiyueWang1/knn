import tensorflow as tf

class AlexNet:
    def __init__(self):
        self.in_x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1], name="in_x")
        self.in_y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name="in_y")
        # 卷积层 (batch, 28, 28, 1) -> (batch, 8, 8, 96)  # (原文:filters=96, kernel_size=11, strides=(4, 4))
        self.conv1 = tf.layers.Conv2D(filters=96, kernel_size=7, strides=(3, 3),
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=tf.sqrt(1 / 48)))
        # 池化层 (batch, 8, 8, 96) -> (batch, 4, 4, 96)  # (原文:pool_size=(3, 3), strides=(2, 2))
        self.pool1 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
        # 卷积层 (batch, 4, 4, 96) -> (batch, 4, 4, 256)  # (原文:filters=256, kernel_size=5)
        self.conv2 = tf.layers.Conv2D(filters=256, kernel_size=3, padding="SAME",
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=tf.sqrt(1 / 128)))
        # 池化层 (batch, 4, 4, 256) -> (batch, 2, 2, 256)  # (原文:pool_size=(3, 3), strides=(2, 2))
        self.pool2 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
        # 卷积层 (batch, 2, 2, 256) -> (batch, 2, 2, 384)  # (原文:filters=384, kernel_size=3)
        self.conv3 = tf.layers.Conv2D(filters=384, kernel_size=1, padding="SAME",
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=tf.sqrt(1 / 192)))
        # 卷积层 (batch, 2, 2, 384) -> (batch, 2, 2, 384)  # (原文:filters=384, kernel_size=3)
        self.conv4 = tf.layers.Conv2D(filters=384, kernel_size=1, padding="SAME",
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=tf.sqrt(1 / 192)))
        # 卷积层 (batch, 2, 2, 384) -> (batch, 2, 2, 256)  # (原文:filters=256, kernel_size=3)
        self.conv5 = tf.layers.Conv2D(filters=256, kernel_size=1, padding="SAME",
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=tf.sqrt(1 / 128)))
        # 池化层 (batch, 2, 2, 256) -> (batch, 1, 1, 256)  # (原文:pool_size=(3, 3), strides=(2, 2))
        self.pool3 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        # reshape(-1, 256) -> (batch, 256)  # (原文:units=9216)
        self.fc1 = tf.layers.Dense(units=256,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=tf.sqrt(1 / 128)))
        # (batch, 256) -> (batch, 128)  # (原文:units=4096)
        self.fc2 = tf.layers.Dense(units=128,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=tf.sqrt(1 / 64)))
        # (batch, 128) -> (batch, 128)  # (原文:units=4096)
        self.fc3 = tf.layers.Dense(units=128,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=tf.sqrt(1 / 64)))
        # (batch, 128) -> (batch, 10)  # (原文:根据自己分类需要)
        self.fc4 = tf.layers.Dense(units=10, kernel_initializer=tf.truncated_normal_initializer(stddev=tf.sqrt(1 / 5)))

    def forward(self):  # 因为方便训练集 所以改了部分AlexNet网络参数 和训练的训练集 改的地方已注明
        self.conv1_out = tf.nn.relu(self.conv1(self.in_x))
        self.poo1_out = self.pool1(self.conv1_out)
        self.conv2_out = tf.nn.relu(self.conv2(self.poo1_out))
        self.poo2_out = self.pool2(self.conv2_out)
        self.conv3_out = tf.nn.relu(self.conv3(self.poo2_out))
        self.conv4_out = tf.nn.relu(self.conv4(self.conv3_out))
        self.conv5_out = tf.nn.relu(self.conv5(self.conv4_out))
        self.pool3 = self.pool3(self.conv5_out)
        self.flat = tf.reshape(self.pool3, shape=[-1, 256])
        self.fc1_out = tf.nn.relu(self.fc1(self.flat))
        self.fc2_out = tf.nn.relu(self.fc2(self.fc1_out))
        self.fc3_out = tf.nn.relu(self.fc3(self.fc2_out))
        self.fc4_out = self.fc4(self.fc3_out)

    def backward(self):  # 后向计算
        self.loss = tf.reduce_mean((self.fc4_out - self.in_y) ** 2)
        self.opt = tf.train.AdamOptimizer().minimize(self.loss)

    def acc(self):  # 精度计算(可不写, 不影响网络使用)
        self.acc1 = tf.equal(tf.argmax(self.fc4_out, 1), tf.argmax(self.in_y, 1))
        self.accaracy = tf.reduce_mean(tf.cast(self.acc1, dtype=tf.float32))
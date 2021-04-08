#coding: utf-8
from tensorflow.keras import datasets
# import scipy.misc
import os
from PIL import Image

# 读取MNIST数据集。如果不存在会事先下载。
# mnist = tf.keras.datasets.mnist.load_data(
#     path_data = "D:/文档/lico-demo/datasets/mnist/")
data_path = 'D:/文档/lico-demo/datasets/mnist/mnist.npz'
(train_images, train_labels), (test_images, test_labels) = \
    datasets.mnist.load_data(path=data_path)


# 保存前20张图片
for i in range(60000):
    # print("-------------", train_images[i, :], train_labels[i])

    # 我们把原始图片保存在MNIST_data/raw/文件夹下
    # 如果没有这个文件夹会自动创建
    save_dir = 'D:/文档/TensorFlow2.0_InceptionV3/original_dataset/' + \
        str(train_labels[i]) + '/'
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

    # 请注意，mnist.train.images[i, :]就表示第i张图片（序号从0开始）
    image_array = train_images[i, :]
    # TensorFlow中的MNIST图片是一个784维的向量，我们重新把它还原为28x28维的图像。
    image_array = image_array.reshape(28, 28)
    # 保存文件的格式为 mnist_train_0.jpg, mnist_train_1.jpg, ... ,mnist_train_19.jpg
    filename = save_dir + 'mnist_train_%d.jpg' % i
    # 将image_array保存为图片
    # 先用scipy.misc.toimage转换为图像，再调用save直接保存。
    # scipy.misc.toimage(image_array, cmin=0.0, cmax=1.0).save(filename)
    Image.fromarray(image_array).convert(
        'L').save(filename)

print('Please check: %s ' % save_dir)

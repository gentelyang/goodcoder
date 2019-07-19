#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author:gentelyang  time:2019-07-19
import os
import random
import paddle
import shutil
import numpy as np
import image_reader
from PIL import Image
import paddle.fluid as fluid
import matplotlib.pyplot as plt
from multiprocessing import cpu_count

def train_mapper(sample):
    """
    Image preprocessing operation.
    :param sample:
    :return:
    """
    img, crop_size = sample
    img = Image.open(img)
    r1 = random.random()
    if r1 > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    width = img.size[0]
    height = img.size[1]
    if width < height:
        ratio = width / crop_size
        width = width / ratio
        height = height / ratio
        img = img.resize((int(width), int(height)), Image.ANTIALIAS)
        height = height / 2
        crop_size2 = crop_size / 2
        box = (0, int(height - crop_size2), int(width), int(height + crop_size2))
    else:
        ratio = height / crop_size
        height = height / ratio
        width = width / ratio
        img = img.resize((int(width), int(height)), Image.ANTIALIAS)
        width = width / 2
        crop_size2 = crop_size / 2
        box = (int(width - crop_size2), 0, int(width + crop_size2), int(height))
    img = img.crop(box)
    img = img.resize((crop_size, crop_size), Image.ANTIALIAS)
    if len(img.getbands()) == 1:
        img1 = img2 = img3 = img
        img = Image.merge('RGB', (img1, img2, img3))
    img = np.array(img).astype(np.float32)
    img = img.transpose((2, 0, 1))
    img = img[(2, 1, 0), :, :] / 255.0
    return img
def reader(train_image_path, crop_size):
    """
    Read the image of the specified path.
    :param train_image_path:
    :param crop_size:
    :return:
    """
    pathss = []
    for root, dirs, files in os.walk(train_image_path):
        path = [os.path.join(root, name) for name in files]
        pathss.extend(path)

    def reader():
        for line in pathss:
            yield line, crop_size
    return paddle.reader.xmap_readers(train_mapper, reader, cpu_count(), 1024)

image_size = 112  # Size of training data set

def Generator(y, name="G"):
    """
    Definition Generator.
    :param y:
    :param name:
    :return:
    """
    def deconv(x,
               num_filters,
               filter_size=5,
               stride=2,
               dilation=1,
               padding=2,
               output_size=None,
               act=None):
        return fluid.layers.conv2d_transpose(input=x,
                                             num_filters=num_filters,
                                             output_size=output_size,
                                             filter_size=filter_size,
                                             stride=stride,
                                             dilation=dilation,
                                             padding=padding,
                                             act=act)
    with fluid.unique_name.guard(name + "/"):
        y = fluid.layers.fc(y, size=2048)
        y = fluid.layers.batch_norm(y)
        y = fluid.layers.fc(y, size=int(128 * (image_size / 4) * (image_size / 4)))
        y = fluid.layers.batch_norm(y)
        y = fluid.layers.reshape(y, shape=[-1, 128, int((image_size / 4)), int((image_size / 4))])
        y = deconv(x=y, num_filters=128, act='relu', output_size=[int((image_size / 2)), int((image_size / 2))])
        y = deconv(x=y, num_filters=3, act='sigmoid', output_size=[image_size, image_size])
    return y

def Discriminator(images, name="D"):
    """
    Definition discriminator.
    :param images:
    :param name:
    :return:
    """
    def conv_pool(input, num_filters, act=None):
        """
        Define a convolution pooling group.
        :param input:
        :param num_filters:
        :param act:
        :return:
        """
        return fluid.nets.simple_img_conv_pool(input=input,
                                               filter_size=3,
                                               num_filters=num_filters,
                                               pool_size=2,
                                               pool_stride=2,
                                               act=act)

    with fluid.unique_name.guard(name + "/"):
        y = fluid.layers.reshape(x=images, shape=[-1, 3, image_size, image_size])
        y = conv_pool(input=y, num_filters=64, act='leaky_relu')
        y = conv_pool(input=y, num_filters=128)
        y = fluid.layers.batch_norm(input=y, act='leaky_relu')
        y = fluid.layers.fc(input=y, size=1024)
        y = fluid.layers.batch_norm(input=y, act='leaky_relu')
        y = fluid.layers.fc(input=y, size=1, act='sigmoid')
    return y

train_d_fake = fluid.Program()
train_d_real = fluid.Program()
train_g = fluid.Program()
startup = fluid.Program()  # Create a common initialization program
z_dim = 100  # Noise Dimension

def getparams(program, prefix):
    """
    Get the parameter name starting with prefix from Program.
    :param program:
    :param prefix:
    :return:
    """
    all_params = program.global_block().all_parameters()
    return [t.name for t in all_params if t.name.startswith(prefix)]

with fluid.program_guard(train_d_real, startup):
    real_image = fluid.layers.data('image', shape=[3, image_size, image_size])
    ones = fluid.layers.fill_constant_batch_size_like(real_image, shape=[-1, 1], dtype='float32', value=1)
    p_real = Discriminator(real_image)
    real_cost = fluid.layers.sigmoid_cross_entropy_with_logits(p_real, ones)
    real_avg_cost = fluid.layers.mean(real_cost)
    d_params = getparams(train_d_real, "D")
    optimizer = fluid.optimizer.Adam(learning_rate=2e-4)
    optimizer.minimize(real_avg_cost, parameter_list=d_params)

with fluid.program_guard(train_d_fake, startup):
    z = fluid.layers.data(name='z', shape=[z_dim])
    zeros = fluid.layers.fill_constant_batch_size_like(z, shape=[-1, 1], dtype='float32', value=0)
    p_fake = Discriminator(Generator(z))
    fake_cost = fluid.layers.sigmoid_cross_entropy_with_logits(p_fake, zeros)
    fake_avg_cost = fluid.layers.mean(fake_cost)
    d_params = getparams(train_d_fake, "D")
    optimizer = fluid.optimizer.Adam(learning_rate=2e-4)
    optimizer.minimize(fake_avg_cost, parameter_list=d_params)

fake = None
with fluid.program_guard(train_g, startup):
    z = fluid.layers.data(name='z', shape=[z_dim])
    ones = fluid.layers.fill_constant_batch_size_like(z, shape=[-1, 1], dtype='float32', value=1)
    fake = Generator(z)
    infer_program = train_g.clone(for_test=True)
    p = Discriminator(fake)
    g_cost = fluid.layers.sigmoid_cross_entropy_with_logits(p, ones)
    g_avg_cost = fluid.layers.mean(g_cost)
    g_params = getparams(train_g, "G")
    optimizer = fluid.optimizer.Adam(learning_rate=2e-4)
    optimizer.minimize(g_avg_cost, parameter_list=g_params)

def z_reader():
    """
    Noise Generation.
    :return:
    """
    while True:
        yield np.random.uniform(-1.0, 1.0, (z_dim)).astype('float32')

def cifarreader(reader):
    """
    Read the cifar dataset without label.
    :param reader:
    :return:
    """
    def r():
        for img, label in reader():
            yield img.reshape(3, 32, 32)
    return r

def show_image_grid(images):
    """
    Save pictures.
    :param images:
    :return:
    """
    for i, image in enumerate(images):
        image = image.transpose((2, 1, 0))
        save_image_path = 'train_image'
        if not os.path.exists(save_image_path):
            os.makedirs(save_image_path)
        plt.imsave(os.path.join(save_image_path, "test_%d.png" % i), image)
mydata_generator = paddle.batch(reader=image_reader.train_reader('datasets', image_size), batch_size=32)
z_generator = paddle.batch(z_reader, batch_size=32)()
test_z = np.array(next(z_generator))
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(startup)
for pass_id in range(100):
    for i, real_image in enumerate(mydata_generator()):
        r_fake = exe.run(program=train_d_fake,
                         fetch_list=[fake_avg_cost],
                         feed={'z': test_z})
        r_real = exe.run(program=train_d_real,
                         fetch_list=[real_avg_cost],
                         feed={'image': np.array(real_image)})
        r_g = exe.run(program=train_g,
                      fetch_list=[g_avg_cost],
                      feed={'z': test_z})
        if i % 100 == 0:
            print("Pass：%d, Batch：%d, 训练判别器D识别真实图片Cost：%0.5f, "
                  "训练判别器D识别生成器G生成的假图片Cost：%0.5f, "
                  "训练生成器G生成符合判别器D标准的假图片Cost：%0.5f" % (pass_id, i, r_fake[0], r_real[0], r_g[0]))

    r_i = exe.run(program=infer_program,
                  fetch_list=[fake],
                  feed={'z': test_z})

    r_i = np.array(r_i).astype(np.float32)
    show_image_grid(r_i[0])
    save_path = 'infer_model/'
    shutil.rmtree(save_path, ignore_errors=True)
    os.makedirs(save_path)
    fluid.io.save_inference_model(save_path,
                                  feeded_var_names=[z.name],
                                  target_vars=[fake],
                                  executor=exe,
                                  main_program=train_g)

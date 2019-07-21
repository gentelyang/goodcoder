#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import paddle
import paddle.fluid as fluid
import matplotlib.pyplot as plt


def Generator(y, name="G"):
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
        y = fluid.layers.fc(y, size=128 * 7 * 7)
        y = fluid.layers.batch_norm(y)
        y = fluid.layers.reshape(y, shape=(-1, 128, 7, 7))
        y = deconv(x=y, num_filters=128, act='relu', output_size=[14, 14])
        y = deconv(x=y, num_filters=1, act='tanh', output_size=[28, 28])
    return y


def Discriminator(images, name="D"):
    def conv_pool(input, num_filters, act=None):
        return fluid.nets.simple_img_conv_pool(input=input,
                                               filter_size=5,
                                               num_filters=num_filters,
                                               pool_size=2,
                                               pool_stride=2,
                                               act=act)

    with fluid.unique_name.guard(name + "/"):
        y = fluid.layers.reshape(x=images, shape=[-1, 1, 28, 28])
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

startup = fluid.Program()

z_dim = 100

def get_params(program, prefix):
    all_params = program.global_block().all_parameters()
    return [t.name for t in all_params if t.name.startswith(prefix)]

with fluid.program_guard(train_d_real, startup):
    real_image = fluid.layers.data('image', shape=[1, 28, 28])
    ones = fluid.layers.fill_constant_batch_size_like(real_image,
                                                      shape=[-1, 1],
                                                      dtype='float32',
                                                      value=1)

    p_real = Discriminator(real_image)
    real_cost = fluid.layers.sigmoid_cross_entropy_with_logits(p_real, ones)
    real_avg_cost = fluid.layers.mean(real_cost)

    d_params = get_params(train_d_real, "D")
    optimizer = fluid.optimizer.Adam(learning_rate=2e-4)
    optimizer.minimize(real_avg_cost, parameter_list=d_params)

with fluid.program_guard(train_d_fake, startup):
    z = fluid.layers.data(name='z', shape=[z_dim])
    zeros = fluid.layers.fill_constant_batch_size_like(z,
                                                       shape=[-1, 1],
                                                       dtype='float32',
                                                       value=0)

    p_fake = Discriminator(Generator(z))

    fake_cost = fluid.layers.sigmoid_cross_entropy_with_logits(p_fake, zeros)
    fake_avg_cost = fluid.layers.mean(fake_cost)

    d_params = get_params(train_d_fake, "D")
    optimizer = fluid.optimizer.Adam(learning_rate=2e-4)
    optimizer.minimize(fake_avg_cost, parameter_list=d_params)

with fluid.program_guard(train_g, startup):
    z = fluid.layers.data(name='z', shape=[z_dim])
    ones = fluid.layers.fill_constant_batch_size_like(z,
                                                      shape=[-1, 1],
                                                      dtype='float32',
                                                      value=1)

    fake = Generator(z)
    infer_program = train_g.clone(for_test=True)

    p = Discriminator(fake)

    g_cost = fluid.layers.sigmoid_cross_entropy_with_logits(p, ones)
    g_avg_cost = fluid.layers.mean(g_cost)
    g_params = get_params(train_g, "G")
    optimizer = fluid.optimizer.Adam(learning_rate=2e-4)
    optimizer.minimize(g_avg_cost, parameter_list=g_params)


def z_reader():
    while True:
        yield np.random.uniform(-1.0, 1.0, (z_dim)).astype('float32')

def mnist_reader(reader):
    def r():
        for img, label in reader():
            yield img.reshape(1, 28, 28)

    return r

def show_image_grid(images):
    for i, image in enumerate(images[:64]):
        image = image[0]
        plt.imsave("image/test_%d.png" % i, image, cmap='Greys_r')

mnist_generator = paddle.batch(
    paddle.reader.shuffle(mnist_reader(paddle.dataset.mnist.train()), 30000), batch_size=128)
z_generator = paddle.batch(z_reader, batch_size=128)()

place = fluid.CPUPlace()
# place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
exe.run(startup)
test_z = np.array(next(z_generator))
for pass_id in range(20):
    for i, real_image in enumerate(mnist_generator()):
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


    show_image_grid(r_i[0])
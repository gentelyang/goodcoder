#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author:gentelyang  time:2019-07-19
import unittest
from train import *
class trainTestCase(unittest.TestCase):

    def test_train_mapper(self,sample):
        res = train_mapper(sample)
        self.assertEqual(res.shape,sample.shape)

    def test_reader(self,train_image_path,crop_size):
        path = "/Users/liyang109/PycharmProjects/distributed-paddle-resnet/liyang109"
        res = reader(path,5)
        y = paddle.reader.xmap_readers(train_mapper, reader, cpu_count(), 1024)
        self.assertEqual(res,y)

    def test_Generator(self,y,name="G"):
        res = Generator(y,name="G")
        y = [32,32]
        self.assertEqual(res,y)

    def test_Discriminator(self,images,name="D"):
        res = Discriminator(images,name="D")
        self.assertEqual(res,0.5)

    def test_getparams(self,program, prefix):
        res = getparams(program, prefix)
        self.assertEqual(res,prefix)

    def test_cifarreader(self,reader):
        res = cifarreader(reader)
        y=[3,32,32]
        self.assertEqual(res,y)

if __name__ == "__main__":
    unittest.main()


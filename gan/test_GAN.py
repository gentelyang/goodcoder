#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
from GAN import *
class trainTestCase(unittest.TestCase):
    def test_Generator(self, y, name="G"):

        res = Generator(y, name="G")
        x = [28,28]
        self.assertEqual(res.shape, x)

    def test_Discriminator(self, images, name="D"):

        res = Discriminator(images, name="D")
        predict = [0,1]
        self.assertEqual(res, predict)

    def test_get_params(self, program, prefix):

        res = get_params(program, prefix)
        self.assertEqual(res, prefix)

    def test_z_reader(self):

        res = z_reader()
        self.assertEqual(res.astype, "float32")

    def test_mnist_reader(self, reader):

        res = mnist_reader(reader)
        x = [28,28]
        self.assertEqual(res.shape, x)


if __name__ == '__main__':
    unittest.main()
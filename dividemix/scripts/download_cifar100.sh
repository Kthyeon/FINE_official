#!/bin/bash

wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar -zxvf cifar-100-python.tar.gz
mv cifar-100-python/ cifar-100/
rm cifar-100-python.tar.gz
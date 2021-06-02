#!/bin/bash

wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -zxvf cifar-10-python.tar.gz
mv cifar-10-batches-py/ cifar-10/
rm cifar-10-python.tar.gz
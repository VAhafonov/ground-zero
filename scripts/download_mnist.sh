#!/bin/bash
curl -L -o ./data/mnist-dataset.zip\
  https://www.kaggle.com/api/v1/datasets/download/hojjatk/mnist-dataset
unzip ./data/mnist-dataset.zip -d ./data/mnist/
rm -rf ./data/mnist-dataset.zip
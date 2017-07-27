#! /usr/bin/bash

wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/multicore-liblinear/liblinear-multicore-2.11-2.zip
unzip liblinear-multicore-2.11-2.zip
rm -rf liblinear-multicore-2.11-2.zip
git clone https://github.com/JohnLangford/vowpal_wabbit
cd vowpal_wabbit
make
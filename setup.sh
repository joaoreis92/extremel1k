
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/multicore-liblinear/liblinear-multicore-2.11-2.zip
unzip liblinear-multicore-2.11-2.zip
rm -rf liblinear-multicore-2.11-2.zip
cd liblinear-multicore-2.11-2/python/
make
cd ~/project/extremel1k/
git clone https://github.com/JohnLangford/vowpal_wabbit
cd vowpal_wabbit
make
cd ~/project/extremel1k/
mkdir data_experiments
mkdir model_experiments
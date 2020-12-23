#!/bin/bash

rm -rf DIV2K/

wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_unknown.zip
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_unknown.zip

unzip DIV2K_train_HR.zip
unzip DIV2K_train_LR_unknown.zip
unzip DIV2K_valid_HR.zip
unzip DIV2K_valid_LR_unknown.zip

mkdir DIV2K
mkdir DIV2K/train
mkdir DIV2K/valid

mv DIV2K_train_HR DIV2K/train/HR/
mv DIV2K_train_LR_unknown DIV2K/train/LR/
mv DIV2K_valid_HR DIV2K/valid/HR/
mv DIV2K_valid_LR_unknown DIV2K/valid/LR/

rm DIV2K_train_HR.zip
rm DIV2K_train_LR_unknown.zip
rm DIV2K_valid_HR.zip
rm DIV2K_valid_LR_unknown.zip

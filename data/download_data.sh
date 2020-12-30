#!/bin/bash

rm -rf DIV2K/

wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip

unzip DIV2K_train_HR.zip
unzip DIV2K_valid_HR.zip

mkdir DIV2K
mkdir DIV2K/train
mkdir DIV2K/valid

mv DIV2K_train_HR DIV2K/train/HR/
mv DIV2K_valid_HR DIV2K/valid/HR/

rm DIV2K_train_HR.zip
rm DIV2K_valid_HR.zip

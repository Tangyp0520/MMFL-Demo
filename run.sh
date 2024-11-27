#!/bin/bash


cd "$(dirname "$0")" || exit
cd src || exit

echo "Run experiment scripts"

echo "Local test"
python local_test.py
if [ $? -ne 0 ]; then
    echo "local_test.py failed. Exiting."
    exit 1
fi

echo "VFL test"
python vfl_test.py
if [ $? -ne 0 ]; then
    echo "hfl_test.py failed. Exiting."
    exit 1
fi


echo "HFL test"
python hfl_test.py
if [ $? -ne 0 ]; then
    echo "hfl_test.py failed. Exiting."
    exit 1
fi

echo "HFM test"
python main.py
if [ $? -ne 0 ]; then
    echo "main.py failed. Exiting."
    exit 1
fi

echo "All scripts ran successfully."

# 返回上级目录（即最初的脚本所在目录）
cd..
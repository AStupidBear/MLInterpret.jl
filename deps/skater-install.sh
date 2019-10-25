#!/bin/bash
sudo apt-get remove libopenblas*
pip install keras tzlocal
conda install -y gxx_linux-64
cd /tmp/ && git clone https://github.com/oracle/Skater.git && cd Skater
sed -i "s/sudo python/python/g" setup.sh
sed -i "s/.*install python.*//g" setup.sh
python setup.py install --ostype=linux-ubuntu --rl=True
conda uninstall -y gxx_linux-64
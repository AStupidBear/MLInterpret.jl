#!/bin/bash
if ! docker &> /dev/null; then
    sudo apt install docker.io
fi
ver=1.7.0
cuver=10.0
prefix=https://s3.amazonaws.com/artifacts.h2o.ai/releases/ai/h2o/dai/rel-$ver-214/x86_64-centos7
cd /tmp/ && aria2c -x 4 -c $prefix/dai-docker-centos7-x86_64-$ver-$cuver.tar.gz
docker load < dai-docker-*.tar.gz
docker tag h2oai/dai-centos7-x86_64:$ver-cuda$cuver dai
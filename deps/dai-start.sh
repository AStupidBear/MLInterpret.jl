#!/bin/bash
cd $(dirname $0)
mkdir -p dai && cd dai
mkdir -p data log license tmp
if [ ! "$(docker ps -a | grep dai)" ]; then
    docker run -d \
        --name dai \
        --pid=host \
        --init \
        -u `id -u`:`id -g` \
        -p 12345:12345 \
        -v $PWD/data:/data \
        -v $PWD/log:/log \
        -v $PWD/license:/license \
        -v $PWD/tmp:/tmp \
        -v $HOME:$HOME \
        -v /dev/shm:/dev/shm \
        dai
else
    docker start dai
fi
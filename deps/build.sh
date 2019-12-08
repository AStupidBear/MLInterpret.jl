#!/bin/bash
FAKETIME=$JULIA_DEPOT_PATH/faketime
if [ ! -f $FAKETIME/lib/libfaketime.so.1 ]; then
    git clone https://github.com/wolfcw/libfaketime.git
    cd libfaketime/src && PREFIX=$FAKETIME LIBDIRNAME=/lib make install
    cd /tmp && rm -rf libfaketime
fi
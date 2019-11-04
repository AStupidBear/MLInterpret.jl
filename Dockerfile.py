#!/usr/bin/env python
try:
    from pydocker import DockerFile
except ImportError:
    from urllib.request import urlopen
    exec(urlopen('https://raw.githubusercontent.com/jen-soft/pydocker/master/pydocker.py').read())

import argparse
import logging
import os
import sys
import tempfile

logging.getLogger('').setLevel(logging.INFO)
logging.root.addHandler(logging.StreamHandler(sys.stdout))

d = DockerFile(base_img='ubuntu:18.04', name='astupidbear/mli:latest')

d.ENV = 'LC_ALL=C.UTF-8'
d.ENV  = 'DEBIAN_FRONTEND=noninteractive'
d.ENV  = 'DEBCONF_NONINTERACTIVE_SEEN=true'
d.RUN = '''\
apt-get update && apt-get install -y sudo gnupg software-properties-common wget build-essential tzdata \
&& wget https://julialang-s3.julialang.org/bin/linux/x64/1.2/julia-1.2.0-linux-x86_64.tar.gz \
&& tar xvzf julia-*.tar.gz && rm julia-*.tar.gz && mv julia-* /opt/julia
'''
d.ENV = 'PATH=/opt/julia/bin:$PATH'
d.RUN = "julia -e 'using Pkg; pkg\"add https://github.com/AStupidBear/MLInterpret.jl\"'"
d.RUN = 'julia -e "using MLInterpret; MLInterpret.install_brl()"'

d.CMD = ['julia']

os.chdir(tempfile.mkdtemp())
d.build_img()

os.system('docker run -it --rm astupidbear/mli:latest')
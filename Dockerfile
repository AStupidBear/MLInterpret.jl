FROM ubuntu:18.04
ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN true
RUN apt-get update && apt-get install -y sudo gnupg software-properties-common wget build-essential tzdata
RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.2/julia-1.2.0-linux-x86_64.tar.gz && \
    tar xvzf julia-*.tar.gz && rm julia-*.tar.gz && mv julia-* /opt/julia
ENV PATH "/opt/julia/bin:$PATH"
RUN julia -e 'using Pkg; pkg"add MLInterpret"'
RUN julia -e 'using MLInterpret; MLInterpret.install_brl()'
CMD julia
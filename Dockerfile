FROM ubuntu:18.04
RUN apt-get update && apt-get install -y sudo gnupg software-properties-common wget build-essential
RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.2/julia-1.2.0-linux-x86_64.tar.gz && \
    tar xvzf julia-*.tar.gz && rm julia-*.tar.gz && mv julia-* /opt/julia
ENV PATH "/opt/julia/bin:$PATH"
ENV DEBIAN_FRONTEND "noninteractive"
RUN julia -e '''\
    using Pkg; \
    pkg"add https://github.com/AStupidBear/Pandas.jl.git"; \
    pkg"add https://github.com/AStupidBear/PyCallUtils.jl.git"; \
    pkg"add https://github.com/AStupidBear/MLI.jl.git"; \
    '''
CMD julia
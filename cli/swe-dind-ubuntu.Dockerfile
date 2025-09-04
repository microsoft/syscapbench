FROM cruizba/ubuntu-dind

RUN apt-get update && apt-get install -y \
    build-essential \
    python3 \
    python3-pip \
    python3-venv \
    wget \
    curl \
    git
# SWE-ReX will always attempt to install its server into your docker container
# however, this takes a couple of seconds. If we already provide it in the image,
# this is much faster.
# RUN pip3 install --upgrade pip --break-system-packages
RUN pip3 install pipx --break-system-packages
RUN pipx install swe-rex 
RUN pipx ensurepath

# ADD . /app
# WORKDIR /app

RUN pip3 install flake8 --break-system-packages

ENV GOLANG_VERSION=1.22.3

RUN apt-get update && apt-get install -y wget tar git build-essential \
    && wget https://go.dev/dl/go${GOLANG_VERSION}.linux-amd64.tar.gz \
    && tar -C /usr/local -xzf go${GOLANG_VERSION}.linux-amd64.tar.gz \
    && rm go${GOLANG_VERSION}.linux-amd64.tar.gz \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH="/usr/local/go/bin:${PATH}"

RUN python3 --version && go version

# This is where pipx installs things
ENV PATH="$PATH:/root/.local/bin/" 

RUN python3 --version && go version



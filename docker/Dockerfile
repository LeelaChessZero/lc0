FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

ENV TRAINING=false
ENV TRAINING_USER=foo
ENV TRAINING_PASSWORD=bar


# Build Time Args

# Branch/tag 
ARG LC0_BRANCH=master

# Training Client version
ARG CLIENT_VERSION=v34
ENV CLIENT_VERSION=${CLIENT_VERSION}



# Create working directory
RUN mkdir -p /chess
WORKDIR /chess

# Install dependencies
RUN apt-get update && apt-get install -y \
    wget \
    meson \
    ninja-build \
    git \
    nano \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Clone and build lc0
RUN git clone https://github.com/LeelaChessZero/lc0.git
WORKDIR /chess/lc0

RUN git checkout ${LC0_BRANCH}

# Build lc0
RUN ./build.sh

# Create training script
RUN echo '#!/bin/bash\n\
NUM_GPUS=$(nvidia-smi -L | wc -l)\n\
echo "Found $NUM_GPUS GPU(s)"\n\
for ((i=0; i<$NUM_GPUS; i++)); do\n\
    echo "Starting training client on GPU $i"\n\
    ./lc0-training-client-linux -report-gpu -gpu=$i -user="$1" -password="$2" &\n\
done\n\
wait' > /chess/start_training.sh && chmod +x /chess/start_training.sh


# Training mode setup (will run when container starts with TRAINING=true)
RUN echo '#!/bin/bash\n\
if [ "$TRAINING" = "true" ]; then\n\
    echo "Training mode enabled"\n\
    cd /chess/lc0/build/release\n\
    if [ ! -f "lc0-training-client-linux" ]; then\n\
        wget -q https://github.com/LeelaChessZero/lczero-client/releases/download/${CLIENT_VERSION}/lc0-training-client-linux\n\
        chmod 755 ./lc0-training-client-linux\n\
    fi\n\
    /chess/start_training.sh "$TRAINING_USER" "$TRAINING_PASSWORD"\n\
else\n\
    if [ -t 0 ] && [ -t 1 ]; then\n\
        exec /chess/lc0/build/release/lc0 "$@"\n\
    else\n\
        echo "Warning: Not running in interactive mode (-it flag missing)"\n\
        echo "There is no way to give uci input. Consider running with: docker run -it ..."\n\
        exec /chess/lc0/build/release/lc0 "$@"\n\
    fi\n\
fi' > /chess/docker-entrypoint.sh && chmod +x /chess/docker-entrypoint.sh


ENTRYPOINT ["/chess/docker-entrypoint.sh"]

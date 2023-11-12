FROM ubuntu:22.04
LABEL authors="xspanger3770"

# Update the package list and install essential dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    openjdk-17-jdk \
    maven \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install NVIDIA CUDA Toolkit
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y --no-install-recommends \
    nvidia-cuda-toolkit \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for CUDA
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH

RUN addgroup --gid 1000 group && \
    adduser --gid 1000 --uid 1000 --disable-password --gecos user user

USER 1000

# Set the working directory inside the container
WORKDIR /usr/src/app

# Copy the application JAR file into the container at the working directory
COPY out/artifacts/GlobalQuakeServer/GlobalQuakeServer_v0.10.0_pre5.jar .

# Copy station database and configuration files
COPY Container ./.GlobalQuakeServerData

# Copy CUDA library
COPY GQHypocenterSearch/build/lib ./lib

# Run GQ Server
CMD ["java", "-jar", "-Djava.library.path=./lib", "GlobalQuakeServer_v0.10.0_pre5.jar", "--headless"]

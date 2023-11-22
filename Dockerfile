FROM ubuntu:22.04
LABEL authors="xspanger3770"

# Update the package list and install essential dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    openjdk-17-jdk \
    maven \
    build-essential \
    curl \
    btop \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install NVIDIA CUDA Toolkit
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y --no-install-recommends \
    nvtop \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for CUDA
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH

RUN addgroup --gid 1000 group && \
    adduser --gid 1000 --uid 1000 --disabled-password --gecos user user

# Set the working directory inside the container
WORKDIR /home/user/

# Copy the application JAR file into the container at the working directory
COPY out/artifacts/GlobalQuakeServer/GlobalQuakeServer.jar .

# Copy station database and configuration files
COPY Container ./.GlobalQuakeServerData

# Copy CUDA library
COPY GQHypocenterSearch/build/lib ./lib

RUN chown -R user:group ./.GlobalQuakeServerData
RUN chown -R user:group GlobalQuakeServer.jar
RUN chown -R user:group ./lib
USER 1000

# Run GQ Server
ENTRYPOINT ["java", "-jar", "-Djava.library.path=./lib", "GlobalQuakeServer.jar", "--headless"]

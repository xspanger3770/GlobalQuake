FROM ubuntu:22.04
LABEL authors="xspanger3770"

# Update the package list and install essential dependencies
RUN apt-get update && \
    apt-get install -y \
    openjdk-17-jdk \
    maven \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install NVIDIA CUDA Toolkit and cuDNN (adjust versions as needed)
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y \
    nvidia-cuda-toolkit \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for CUDA
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH


# Set the working directory inside the container
WORKDIR /usr/src/app

# Copy the application JAR file into the container at the working directory
COPY out/artifacts/GlobalQuakeServer/GlobalQuakeServer_v0.10.0_pre5.jar .

COPY Container ./.GlobalQuakeServerData

COPY GQHypocenterSearch/build/lib ./lib

# Specify the command to run your application
CMD ["java", "-jar", "-Dtinylog.writer.level=debug", "-Djava.library.path=./lib", "GlobalQuakeServer_v0.10.0_pre5.jar", "--headless"]

FROM ubuntu:22.04
LABEL authors="xspanger3770"

RUN apt-get update && \
    apt-get install -y \
    openjdk-17-jdk \
    maven \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /usr/src/app

# Copy the application JAR file into the container at the working directory
COPY out/artifacts/GlobalQuakeServer/GlobalQuakeServer_v0.10.0_pre4.jar .

# Specify the command to run your application
CMD ["java", "-jar", "GlobalQuakeServer_v0.10.0_pre4.jar"]

FROM ubuntu:22.04
LABEL authors="xspanger3770"

# Update the package list and install essential dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Update the package list and install additional helpful packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    btop \
    htop \
    nano \
    && rm -rf /var/lib/apt/lists/*

RUN addgroup --gid 1000 group && \
    adduser --gid 1000 --uid 1000 --disabled-password --gecos user user

# Set the working directory inside the container
WORKDIR /home/user/

# Copy the application JAR file into the container at the working directory
COPY GlobalQuakeServer/target/GlobalQuakeServer-*-jar-with-dependencies.jar ./GlobalQuakeServer.jar

# Copy station database and configuration files
COPY Container ./.GlobalQuakeServerData

# Copy CUDA library
COPY GQHypocenterSearch/build/lib ./lib

RUN chown -R user:group ./.GlobalQuakeServerData &&  \
    chown -R user:group GlobalQuakeServer.jar && \
    chown -R user:group ./lib
USER 1000

# Run GQ Server
ENTRYPOINT ["java", "-jar", "-Djava.library.path=./lib", "GlobalQuakeServer.jar", "--headless"]

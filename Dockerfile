FROM ubuntu:latest
LABEL authors="jakub"

ENTRYPOINT ["top", "-b"]
#!/bin/bash -e
# usage: ./build_docker.sh <build_args>
# example: ./build_docker.sh --no-cache

export DOCKER_NAME=unsloth
#export DOCKER_NAME=rgr-base
export DOCKER_REGISTRY=localhost:5000
# docker run -d -p 5000:5000 --name registry registry:2.7
docker start registry

export DOCKER_BUILDKIT=1
export VERSION=0.1
#export VERSION=$(cat version)
#export UID=$(id -u)
export GID=$(id -g)

# clean up previous builds
find ./ -iname build -exec rm -rf \{\} \; && \
find ./ -iname *egg-info -exec rm -rf \{\} \; && \

echo "Building docker $DOCKER_NAME:$VERSION for uid=$UID gid=$GID"
id
docker build $1 $2 $3 $4 \
    --network=host --build-arg UID=$UID --build-arg GID=$GID \
    -t $DOCKER_NAME:$VERSION . \
    |& tee docker.log

# tag as latest
docker tag $DOCKER_NAME:$VERSION $DOCKER_NAME:latest

# register the image
docker tag $DOCKER_NAME:$VERSION $DOCKER_REGISTRY/$DOCKER_NAME:$VERSION
docker push $DOCKER_REGISTRY/$DOCKER_NAME:$VERSION
curl http://$DOCKER_REGISTRY/v2/_catalog

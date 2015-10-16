#!/bin/sh

mkdir -p build
cd build
mkdir -p sw
mkdir -p hw

echo "[info] build software version"
cd sw
cmake ../.. $@
make
cd ..

echo

echo "[info] build OpenCL version"
cd hw
cmake ../.. -DOPENCL=TRUE $@
make
cd ..

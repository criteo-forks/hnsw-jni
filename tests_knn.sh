#!/usr/bin/env bash

cd build/cmake_unix; cmake ../..; make tests_knn; ./tests_knn;
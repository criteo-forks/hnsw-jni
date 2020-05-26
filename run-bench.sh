#!/usr/bin/env bash

./../../gradlew --stop
./../../gradlew :thirdparty:hnsw-jni:clean :thirdparty:hnsw-jni:jmh --info
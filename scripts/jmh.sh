#!/bin/sh
set -x
set -e

SCRIPT_ROOT=$(dirname $(readlink -f $0))
WORKSPACE_PATH=${WORKSPACE}
mkdir -p $WORKSPACE_PATH/artifacts/hnsw-jni
${WORKSPACE_PATH}/gradlew jmh
cp ${SCRIPT_ROOT}/../build/artifacts/* ${WORKSPACE_PATH}/artifacts/hnsw-jni/.
cat << EOF >> ${WORKSPACE_PATH}/REVIEW_MESSAGE
Benchmark results can be found here: ${BUILD_URL}artifact/artifacts/hnsw-jni
EOF
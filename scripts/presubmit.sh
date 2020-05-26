#!/bin/sh
# this script is executed by JMOAB-pre-submit
set -x
set -e

SCRIPT_ROOT=$(dirname $(readlink -f $0))

if [[ $1 = 'end' ]] && [[ ${JOB_NAME} == JMOAB-* ]]
then
    /bin/sh ${SCRIPT_ROOT}/jmh.sh
fi
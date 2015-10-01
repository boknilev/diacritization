#!/bin/sh
export CURRENNT_CUDA_DEVICE=0
echo "CUDA DEVICE:" $CURRENNT_CUDA_DEVICE
/path/to/currennt "$@" --options_file config.cfg 



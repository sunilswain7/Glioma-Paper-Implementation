#!/bin/bash
echo "Start running on Astra (bash test)"
hostname
nvidia-smi
$HOME/Project1/env/bin/python $HOME/Project1/code/glioma_seg/astra/gpu_test.py
echo "Done running on Astra"

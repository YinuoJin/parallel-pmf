#!/bin/bash
k_val=$1
cd ..

./main.tsk \
	--task train \
	-i ../data/expr.csv \
	-m ../data/genes.csv \
  -n 100 \
  -k $k_val \
  --lambda_theta 1 \
  --eta_theta 1 \
  --lambda_beta 0.5 \
  --eta_beta 5 \
  --gamma 0.01 \
  --loss_interval 1 \
	--thread 10


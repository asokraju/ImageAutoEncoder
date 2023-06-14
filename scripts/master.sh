#!/bin/bash -l
for learning_rates in 0.001
do
  for latent_dims in 6 8
  do
    for batch_sizes in 128
    do
      ./scripts/worker.sh $learning_rates $latent_dims $batch_sizes
    done
  done
done
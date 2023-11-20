#!/bin/bash

# Define the parameters
params=(
  "cfg11"
)

# Loop over the parameters
for param in "${params[@]}"; do
  IFS=' ' read -r -a param_array <<< "$param"
  param1=${param_array[0]}

  # Create a batch script for this set of parameters
  batch_script=$(sed "s/{param1}/$param1/g" jobs/train_photons.sub)

  # Submit the batch script to Slurm
  echo "$batch_script" | sbatch
done

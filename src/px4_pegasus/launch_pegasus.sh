#!/bin/bash

# Set the conda environment
export ISAAC_CONDA_ENV="isaac_env"

# Isaac Sim root directory  
export ISAACSIM_PATH="$HOME/isaacsim"

# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $ISAAC_CONDA_ENV

# Source the isaacsim conda environment setup
source $ISAACSIM_PATH/setup_conda_env.sh

# Startup MicroXRCEAgent 
MicroXRCEAgent udp4 -p 8888 &

# Execute the Python script
echo "Running Python script..."
python pegasus_simulation/pegasus_sim.py

# Deactivate the conda environment
conda deactivate

# End of file




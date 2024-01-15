Data for this repository is not included. Contact krishnabhattaram@berkeley.edu for details

# Directory Organization
codeDQN -- Contains the model-free approaches to RL for structure design

codeMBPO -- Contains the model-based approach (including the distribution shift script) 

Neural-Network-Materials -- Code outlining the GNN architecture

# Usage
Use requirements.txt located in codeMBPO to configure required packages using PyPI

To run the model-free DQN implementation:
1. Navigate to the codeDQN directory
2. Run command of the form: python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/nanoslab_cached.yaml --seed 1


To run the Model-Based DQN implementation:
1. Navigate to the codeMBPO directory
2. Run command of the form: python3 cs285/scripts/run_discrete.py -cfg experiments/mpc/nanoslab_mbpo.yaml --dqn_config_file experiments/dqn/nanoslab_clipq.yaml

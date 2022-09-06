# Short-Term Plasticity Neurons Learning to Learn and Forget

This repository reproduces the experiments of [this ICML 2022 paper](https://proceedings.mlr.press/v162/rodriguez22b.html), which has been accepted as Spotlight.

## Abstract
Short-term plasticity (STP) is a mechanism that stores decaying memories in synapses of the cerebral cortex. In computing practice, STP has been used, but mostly in the niche of spiking neurons, even though theory predicts that it is the optimal solution to certain dynamic tasks. Here we present a new type of recurrent neural unit, the STP Neuron (STPN), which indeed turns out strikingly powerful. Its key mechanism is that synapses have a state, propagated through time by a self-recurrent connection-within-the-synapse. This formulation enables training the plasticity with backpropagation through time, resulting in a form of learning to learn and forget in the short term. The STPN outperforms all tested alternatives, i.e. RNNs, LSTMs, other models with fast weights, and differentiable plasticity. We confirm this in both supervised and reinforcement learning (RL), and in tasks such as Associative Retrieval, Maze Exploration, Atari video games, and MuJoCo robotics. Moreover, we calculate that, in neuromorphic or biological circuits, the STPN minimizes energy consumption across models, as it depresses individual synapses dynamically. Based on these, biological STP may have been a strong evolutionary attractor that maximizes both efficiency and computational power. The STPN now brings these neuromorphic advantages also to a broad spectrum of machine learning practice.  Code is available in https://github.com/NeuromorphicComputing/stpn.

##Instructions
### Quick start
    # Install and activate base anaconda environment for STPN
    conda env create -n baseSTPN --file base_conda_environment.yml
    conda activate baseSTPN

    # Train and evaluate STPN in associative retrieval task (Ba et al, 2016)
    python simple_script.py --gpu <-1,0,1,...>

### Experiments in the paper
The plots and tables displayed in the paper can be generated using ICML_all_plots.ipynb

For each task described in the paper, the repository includes a separate folder with each own README.md with instructions to setup, run experiments, and analyze results.

## Cite
To cite this work please use the following citation

    @InProceedings{garcia2022stpn, title = 	 {Short-Term Plasticity Neurons Learning to Learn and Forget}, author =       {Rodriguez, Hector Garcia and Guo, Qinghai and Moraitis, Timoleon}, booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning}, pages = 	 {18704--18722}, year = 	 {2022}, editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan}, volume = 	 {162}, series = 	 {Proceedings of Machine Learning Research}, month = 	 {17--23 Jul}, publisher =    {PMLR}, pdf = 	 {https://proceedings.mlr.press/v162/rodriguez22b/rodriguez22b.pdf}, url = 	 {https://proceedings.mlr.press/v162/rodriguez22b.html}, abstract = 	 {Short-term plasticity (STP) is a mechanism that stores decaying memories in synapses of the cerebral cortex. In computing practice, STP has been used, but mostly in the niche of spiking neurons, even though theory predicts that it is the optimal solution to certain dynamic tasks. Here we present a new type of recurrent neural unit, the STP Neuron (STPN), which indeed turns out strikingly powerful. Its key mechanism is that synapses have a state, propagated through time by a self-recurrent connection-within-the-synapse. This formulation enables training the plasticity with backpropagation through time, resulting in a form of learning to learn and forget in the short term. The STPN outperforms all tested alternatives, i.e. RNNs, LSTMs, other models with fast weights, and differentiable plasticity. We confirm this in both supervised and reinforcement learning (RL), and in tasks such as Associative Retrieval, Maze Exploration, Atari video games, and MuJoCo robotics. Moreover, we calculate that, in neuromorphic or biological circuits, the STPN minimizes energy consumption across models, as it depresses individual synapses dynamically. Based on these, biological STP may have been a strong evolutionary attractor that maximizes both efficiency and computational power. The STPN now brings these neuromorphic advantages also to a broad spectrum of machine learning practice. Code is available in https://github.com/NeuromorphicComputing/stpn.} }
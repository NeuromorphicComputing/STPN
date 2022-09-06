### Training, evaluating commands

#### Atari Pong
This experiment can be run with the pongSTPN conda environment described in ```AtariMujoco/pong_conda_environment.yml```
Change the path to store result (```~/ray_results``` by default) in ```run-pong.py```
Experiment configuration is specified in the scripts.
To train agents, run:
```
conda activate pongSTPN
python run-pong.py
```

To evaluate eval-trainer.py can be used for both environments, however for the results in the paper eval-pong.py was used for PongNoFrameskip-v4. 
To evaluate, specify path to checkpoint in ```eval-pong.py```.
Note, like for training, configuration is detailed in the file.
Finally, run
```
python eval-pong.py
```

#### MuJoCo Inverted Pendulum
This experiment can be run with the baseSTPN conda environment described in ```base_conda_environment.yml```
Change the path to store result (```~/ray_results``` by default) in ```run-pong.py```
Experiment configuration is specified in the scripts.
Set up dependencies and ```LD_LIBRARY_PATH``` as described below.
To train agents for Inverted Pendulum, run:
```
python run-invertedpendulum-ppo.py
```

To evaluate, specify path to checkpoint in ```eval-trainer.py```.
Like for training, evaluation configuration is detailed in the file.
Finally, run
```
python eval-trainer.py
```
### Result files and where to expect them
An example path to training results would be ```~/ray_results/A2C/PongNoFrameskip-v4/stpn_experiments/A2C_PongNoFrameskip-v4_3e5b7_00000_0_2022-01-22_01-40-37```

We use Ray's RLLib to run these experiments. By default, RLLib uses ```~/ray_results/``` to store results.
Under this directory, experiments are first gruped according to agent architecture and environment, here:
```A2C/PongNoFrameskip-v4``` and ```PPO/InvertedPendulum-v2```
Under this, experiments are organized based on the category of the network in the core of the agent:
stpn_experiments , stpnf_experiments  and nonplastic_experiments
Then, for each experiment rllib creates a directory, that includes agent architecture, environment, some of the parameters tuned, creation time and some experiment codes.


Inside each experiment's directory, experiment parameters (params.json and params.pkl), result files (progress.csv and result.json) and optionally tensorboard files and checkpoing directories can be found.
We predominantly use params.json for experiment configurations and progress.csv to analyse proficiency results during training.
For evaluation, we create ```eval/``` directory under the chosen environmnent. The network type and specific environment are the same as for training. Results are stored under a directory with the same name of the checkpoint that was evaluated. Energy consumption, reward and episo lenthgs are stored. FOr example:
```~/ray_results/A2C/PongNoFrameskip-v4/eval/stpn_experiments/A2C_PongNoFrameskip-v4_3e5b7_00000_0_2022-01-22_01-40-37/checkpoint_010794/```

### Dependencies
1. Install ray rllib from wheels, in this case for linux and python 3.9, gives ray==2.0.0dev0 https://docs.ray.io/en/latest/ray-overview/installation.html#daily-releases-nightlies eg:
```
pip uninstall -y ray # clean removal of previous install, otherwise version number may cause pip not to upgrade
pip install -U https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-2.0.0.dev0-cp39-cp39-manylinux2014_x86_64.whl
pip install gym[atari]==0.19.0
pip install ray[rllib] # to install necessary dependencies
```

#### Pong
2. Install cmake and zlib
```
sudo apt get install cmake
sudo apt-get install zlib1g-dev
```
3. Install gym and accept autorom license
```pip install gym[atari]==0.19.0 autorom[accept-rom-license]```
4. The rest of the conda environment can be installed and activated
```conda env create -n pong --file pong_conda_environment.yml```


##### Troubleshooting
a. Might want to try installing rllib from pip instead of binaries.
pip install ray[rllib]
b. Some version conflicts for gym and autorom
might have to update pip (this only happened when installing pytorch through conda)
```
pip3 install --upgrade pip
```
And update version of some packages 
```
pip install pyqt5==5.9.2 pathlib2==2.3.6 ruamel_yaml==0.15.100
```
c. If having issues with accepting ROM license due to proxy settings
Edit ```~/anaconda3/envs/rllib/lib/python3.9/site-packages/AutoROM/AutoROM.py``` to
``` 
proxies={'http': 'http://yourproxy', 'https': 'http://yourproxy'}
    with requests.get(url, stream=True, proxies=proxies, verify=False)
```
then run 
```
python -m atari_py.import_roms ROMS
```
The installed atari-py version might not work. In that case run ( see https://ymd_h.gitlab.io/ymd_blog/posts/yanked_atari_py/ )
```
pip install -U atari-py
```
#### Mujoco Inverted Pendulum
2. Install MuJoCo 2.10
   1. Download the MuJoCo version 2.1 binaries for Linux or OSX. 
   2. Extract the downloaded mujoco210 directory into ~/.mujoco/mujoco210
   3. pip3 install -U 'mujoco-py<2.2,>=2.1'
    Troubleshooting. Might have to install:
   4. ```sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3```
   5. ```sudo apt install libglew-dev```

3. This experiment can be run with the base conda environment described in ```base_conda_environment.yml```.

4. Before executing training/evaluation, ensure the following environment variables are present:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/<username>/.mujoco/mujoco210/bin
```



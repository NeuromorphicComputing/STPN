### Training, evaluating commands
This experiment can be run with the baseSTPN conda environment described in ```base_conda_environment.yml```
Configure your path to store STPN results in ```Scripts/utils.py```
This experiment's config is a mixture of flags (from the original codebase) and .json config files (mostly for model variants).
Each .json config file has commands for training and evaluation. For example, with the base conda environment:
```
# for training
python3 run_maze.py --net_type stpn --nbiter 50000 --bs 512 --eplen 200 --hs 55 --type stpn --lr 1e-4 --l2 0 --addpw 3 --pe 1000 --blossv 0.1 --bent 0.03 --rew 10 --save_every 1000 --rsp 1  --da tanh  --msize 13 --wp 0.0  --gc 4.0  --eval-energy --config_file_path config/STPN_Maze_gNBfNGA_1Clamp_h55.json --rngseed 0  --gpu 0
# for evaluation
python3 run_maze.py --net_type stpn --nbiter 100 --bs 512 --eplen 200 --hs 55 --type stpn --lr 1e-4 --l2 0 --addpw 3 --pe 10 --blossv 0.1 --bent 0.03 --rew 10 --save_every 10 --rsp 1  --da tanh  --msize 13 --wp 0.0  --gc 4.0  --eval-energy --config_file_path config/STPN_Maze_gNAfNGA_1Clamp_h55.json --config_file_train config/STPN_Maze_gNAfNGA_1Clamp_h55.json  --gpu 0 --eval --rngseed 0
```

### Result files and where to expect them
All results are stored under ```<Scripts.utils.RESULTS>/Maze/```.
Result files are located under ```efficiency/``` and ```logs/``` directories, following original implementation.
Result files are named based on the type of result stored, some experiment flags and config file, for instance:
```
torchmodel_bs_512_eplen_200_eval_False_hs_55_lr_0.0001_nbiter_50000_net_type_stpn_rngseed_0_type_stpn_config_{config_file_name}.dat
```
In the ```logs/``` directory, for each experiment, we store the model (```torchmodel_```), the optimiser state (```torchopt_```), current iteration (```numiter_```), gradient norms (```norm_```), rewards (```loss_```) and experiment settings (```params_```)
In the ```efficiency/``` directory, for each experiment, we store the energy consumption (```energy_```).

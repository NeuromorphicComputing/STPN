### Training, evaluating commands
This experiment can be run with the baseSTPN conda environment described in ```base_conda_environment.yml```
Configure your path to store STPN results in ```Scripts/utils.py```
Majority of the configurations regarding data, experiment and model settings are configured via a .json file
The command to run the training and/or evaluation configures the path to this config file and very high level configurations, with the base conda environment:
```
python run_art.py --config_file_path <abs_path_to_config_file> --train --eval --eval-energy --seed 0 --gpu 0
```

### Result files and where to expect them
All result files will be under: ```<Scripts.utils.RESULTS>/AssociativeRetrievalTask```
Grouped into:
- ```proficiency/```
  - Validation and test accuracy
- ```efficiency/```
  - Validation and test energy consumption
- ```models/```
  - Checkpoint of trained model

Under each group of metrics, for each experiment a directory in the form  ```{test_/val_}ART_from_{config_file_name}_seed_{seed}```
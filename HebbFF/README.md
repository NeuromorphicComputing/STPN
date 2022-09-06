
### Training, evaluating commands
This experiment can be run with the baseSTPN conda environment described in ```base_conda_environment.yml```
Configure your path to store STPN results in ```Scripts/utils.py```
The original script from Tyulmankov et al (2022) is adapted to accept flags. 
These are the commands to run all the experiments performed, with the base conda environment:

    python script_simple.py --netType uSTPNrNet --N 62 --trainMode dat --R 3 --epochs 3000 --gpu 0
    python script_simple.py --netType STPNfNet --N 34 --trainMode dat --R 3 --epochs 3000 --gpu 0
    python script_simple.py --netType STPNrNet --N 27 --trainMode dat --R 3 --epochs 3000 --gpu 0
    python script_simple.py --netType uSTPNfNet --N 100 --trainMode dat --R 3 --epochs 3000 --gpu 0
    python script_simple.py --netType HebbNet --N 100 --trainMode dat --R 3 --epochs 3000 --gpu 0
    python script_simple.py --netType nnLSTM --N 21 --trainMode dat --R 3 --epochs 3000 --gpu 0
    
    python script_simple.py --netType uSTPNrNet --N 62 --trainMode inf --R 3 --epochs 1600 --gpu 0
    python script_simple.py --netType STPNfNet --N 34 --trainMode inf --R 3 --epochs 1600 --gpu 0
    python script_simple.py --netType STPNrNet --N 27 --trainMode inf --R 3 --epochs 1600 --gpu 0
    python script_simple.py --netType uSTPNfNet --N 100 --trainMode inf --R 3 --epochs 1600 --gpu 0
    python script_simple.py --netType HebbNet --N 100 --trainMode inf --R 3 --epochs 1600 --gpu 0
    python script_simple.py --netType nnLSTM --N 21 --trainMode inf --R 3 --epochs 1600 --gpu 0
    
    python script_simple.py --netType uSTPNrNet --N 62 --trainMode inf --R 6 --epochs 3500 --gpu 0
    python script_simple.py --netType STPNfNet --N 34 --trainMode inf --R 6 --epochs 3500 --gpu 0
    python script_simple.py --netType STPNrNet --N 27 --trainMode inf --R 6 --epochs 3500 --gpu 0
    python script_simple.py --netType uSTPNfNet --N 100 --trainMode inf --R 6 --epochs 3500 --gpu 0
    python script_simple.py --netType HebbNet --N 100 --trainMode inf --R 6 --epochs 3500 --gpu 0
    python script_simple.py --netType nnLSTM --N 21 --trainMode inf --R 6 --epochs 3500 --gpu 0

### Result files and where to expect them
All results for this task are stored in ```<Scripts.utils.RESULTS>/HebbFF/```.
Each experiment for each model will be sorted in a pickle file.
The attributes of the object store the results, following the approach in the original repository.
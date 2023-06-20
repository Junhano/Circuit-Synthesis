# Learning to Design Analog Circuits to Meet Threshold Specifications



## Overview
 Code for Learning to Design Analog Circuits to Meet Threshold Specifications (Accepted by ICML 2023 as Poster)

 - Preprint Coming Soon
 - Website Comning Soon

## Docker
To export running results to host machine: 

1. Create two folders, one for out_plot and another for result_out. 
2. Run the docker using command 

```
docker run -v {absolute path to out_plot folder}:/Circuit-Synthesis/out_plot -v {absolute path to result_out folder}:/Circuit-Synthesis/result_out {docker image name} --path={Train config path}
```

## Usage
### Problem 1 
Base dataset, DL, across data sizes with base test dataset, success rate as function of (two-sided) error margin:  
```
python main.py --path=./config/config_template/problem1-compare-datasize-relative-Error-margin.yaml
```
### Problem 2 Number 1 and Number 4
Compare datasets construction methods using deep learning, 10-fold Cross Validation as a function of error margin:  
```
python main.py --path=./config/config_template/problem2-compare-dataset-DL-10fold-absolute-Error-margin.yaml
```

### Problem 2 Number 2
Test success rate; Compare training methods (DL, lookup, RF, …) with “softargmax”, 10-fold cross validation as a function of error margin: 
```
python main.py --path=./config/config_template/problem2-compare-method-Softargmax-10fold-absolute-Error-margin.yaml
```

### Problem 2 Number 3
Test success rate; Compare data sizes with DL, “softargmax” as a function of error margin:  

```
python main.py --path=./config/config_template/problem2-compare-datasize-softArgmax-DL-Absolute-Error-margin.yaml
```


### Training Config
#### dataset: 
The dataset that is used to construct the training data. There are six defined datasets available:

- **Base** (D0): The simulation dataset.
- **SoftArgmax** (D-*ε): Our proposed method, with perturbation to match the test distribution.
- **SoftBase** (D ε): Dataset that has perturbed performance metrics that resemble the threshold query distribution, used for method evaluation.
- **Ablation** (D-mε): Dataset for ablation study.
- **Argmax** (D-*0): Our proposed method, without perturbation to match the test distribution.
- **Lourenco** (Dmε): Baseline similar to Lourenco et al.
<br>

####metric: 
Specifies the type of metrics used to evaluate performance. There are two options available:

- **relative**: Performance evaluation is based on the relative difference between predicted and ground truth circuit performance.
- **absolute**: Performance evaluation is based on a threshold specification, where predictions performance that are better than required performance are considered acceptable.
<br>

####model_config: 
Specifies the type of machine learning model or method to be used for the pipeline. <br>

####rerun_training: 
Specifies whether the pipeline should redo the simulation for the circuit before running the pipeline. By default, this parameter is set to `false` since the code can detect if any changes have been made to the circuit configuration file. 
If set to `true`, the pipeline will rerun the training process regardless of whether there have been changes. <br>

####device: 
Specifies the device to be used for PyTorch training. Choose between `cpu` or `gpu` depending on the availability of hardware resources. By default, the code will use the CPU for training if no specific device is specified. <br>

####check_circuit: 
Specifies whether to sample a couple of data points before running the algorithm in order to double-check the correctness of the netlist. By enabling this option, the pipeline will generate a small subset of data points and simulate the circuit using these samples. This can be useful for verifying the functionality and accuracy of the netlist before running the pipeline.
<br>

####num_sample_check: 
Specifies the number of samples that will be used to check the circuit netlist when the `check_circuit` option is enabled.

By setting the `num_sample_check` parameter, you can control the number of data points that will be used for the circuit netlist check. The pipeline will generate `num_sample_check` samples and perform circuit simulations on these selected data points.<br>

####random_sample_scale: 
Specifies the scaling factor applied to the circuit parameters during random sampling. 

By setting the `random_sample_scale` parameter, you can control the extent of scaling for the sampled circuit parameters. For example, if `random_sample_scale` is set to 2, the pipeline will sample values from -2 to 2 and then unnormalize them to their original range.
<br>

####log_experiments: 
Specifies whether to use Wandb (Weights & Biases) for experiment logging. 
Please note that using Wandb for experiment logging may require additional setup and dependencies. Ensure that you have installed and configured Wandb properly before enabling this option.
<br>

####simulator_config:<br>
- num_worker: Specifies the number of workers for parallel circuit simulation. Increasing the number of workers can speed up simulation, but is limited by hardware resources. Exceeding the limit may cause errors.<br>
- sim_size: Specifies the number of simulation points that each worker will simulate during circuit simulation.
Due to the design of Ngspice, the simulator may become slower over time. To address this issue, the simulation needs to be restarted periodically. By setting a smaller value for `sim_size`, the simulation will be divided into smaller segments, resulting in more frequent restarts.
It is recommended to set `sim_size` to a smaller value for larger circuits that take a long time to simulate. This helps to mitigate the performance degradation caused by the simulator over time. <br>
- multiprocessing: Specifies whether the circuit simulation should be run using multiprocessing. Enabling multiprocessing allows for parallel execution of ngspice simulation processes, which can significantly reduce the overall simulation time. <br>

####epsilon: 
Specifies the epsilon parameter for the perturbed datasets `Softbase`, `SoftArgmax`, `Lourenco`, `Argmax` and `Ablation`.  <br>

####subset: 
Specifies a list of percentages indicating the proportion of training data to be used for training. The pipeline will loop through the list and perform the appropriate train-test split for each percentage.<br>

####circuits: 

Specifies the circuits that will be used for training. The available circuits are:

- nmos
- lna
- cascode
- mixer
- vco
- pa
- two_stage

To add a new circuit, you need to provide a circuit YAML file and the corresponding NGSpice netlist. Additionally, you will need to modify the Python code to load the new circuit.

In the future, the circuit loading process will be made dynamic, allowing users to simply provide the required files without modifying the code. This enhancement will provide more flexibility for incorporating custom circuits into the training pipeline. <br>

####loss_per_epoch: 
Specifies whether the training process will keep track of the loss per epoch. Enabling this option allows you to monitor the loss value at each epoch during training. <br>

####test_accuracy_per_epoch: 
Specifies whether the training process will keep track of the test accuracy per epoch. Enabling this option allows you to monitor the accuracy of the model on a separate test dataset at certain epochs during training. <br>

####train_accuracy_per_epoch: 
Specifies whether the training process will keep track of the train accuracy per epoch. Enabling this option allows you to monitor the accuracy of the model on the training dataset at certain epochs during training. <br>

####compare_dataset: 
Specifies whether the training process will compare different datasets during the batch run. Enabling this option allows you to evaluate and compare the performance of the model on multiple datasets.

####compare_method: 
Specifies whether the training process will compare different methods during the batch run. Enabling this option allows you to evaluate and compare the performance of the model trained using different training methods.
Please note that enabling `compare_method` may have an impact on other training configuration variables such as `loss_per_epochs` and `train_accuracy_per_epochs`.<br>

####mode:
Specifies the mode to handle circuit performance requirement satisfaction when constructing the Argmax dataset. The available options are:

- `drop`: If a training circuit parameter dataset does not satisfy the circuit performance requirement, the algorithm will drop that datapoint from the Argmax dataset.

- `replace`: If a training circuit parameter dataset does not satisfy the circuit performance requirement, the algorithm will replace it with the circuit parameter that has the least performance error. <br>

####subset_parameter_check: 
Specifies whether enable the mode for evaluate the performance of the training dataset or method using only a small percentage of the available training data, rather than the entire training data. <br>

####independent_sample: 

Specifies whether to use independent train-test data combinations when performing multiple runs to calculate the average result. By enabling this option, 
the pipeline will generate different train-test splits for each run, ensuring that the training and testing data are independent of each other. If it's not enabled, 
the pipeline will use cross fold validation to calculate average result<br>

## Citation
(Coming soon)

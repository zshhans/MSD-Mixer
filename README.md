# **MSD-Mixer**: A Multi-Scale Decomposition MLP-Mixer for Time Series Analysis

This is the [PyTorch](https://pytorch.org/) implementation of our paper: ***A Multi-Scale Decomposition MLP-Mixer for Time Series Analysis***. (https://arxiv.org/abs/2310.11959)

If you find this repo useful, please consider cite our paper:
```
@misc{zhong2023multiscale,
      title={A Multi-Scale Decomposition MLP-Mixer for Time Series Analysis}, 
      author={Shuhan Zhong and Sizhe Song and Guanyao Li and Weipeng Zhuo and Yang Liu and S. -H. Gary Chan},
      year={2023},
      eprint={2310.11959},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Abstract
Time series data, often characterized by unique composition and complex multi-scale temporal variations, requires special consideration of decomposition and multi-scale modeling in its analysis. Existing deep learning methods on this best fit to only univariate time series, or have not sufficiently accounted for sub-series modeling and the decomposition completeness. To overcome these challenges, we propose **MSD-Mixer**, a **M**ulti-**S**cale **D**ecomposition MLP-Mixer which learns to explicitly decompose the input time series into different components, and represents the components in different layers. To handle multi-scale temporal patterns and inter-channel dependencies, we propose a novel temporal patching approach to model the time series as multi-scale sub-series, i.e., patches, and employ MLPs to mix intra- and inter-patch variations and channel-wise correlations. In addition, we propose a loss function to constrain both the magnitude and autocorrelation of the decomposition residual for decomposition completeness. Through extensive experiments on various real-world datasets for five common time series analysis tasks (long- and short-term forecasting, imputation, anomaly detection, and classification), we demonstrate that MSDMixer consistently achieves significantly better performance in comparison with other state-of-the-art task-general and taskspecific approaches.

![overview](./figs/overview.png)

## Dependency Setup

* Create a conda virtual environment
  ```bash
  conda create -n msd-mixer python=3.10
  conda activate msd-mixer
  ```
* Install Python Packages
  ```bash
  pip install -r requirements.txt
  ```

## Dataset Preparation

<img src="./figs/benchmarks.png" width="60%" height="auto">

Please download the datasets from [dataset.zip](https://hkustconnect-my.sharepoint.com/:u:/g/personal/szhongaj_connect_ust_hk/EYHKJ1krsyNLi2JAK8xtHKcBjAOwti7clrKzjWTU2U5HDw), unzip the content into the `dataset` folder and structure the directory as follows:
```
/path/to/MSD-Mixer/dataset/
  electricity/
  ETT-small/
  exchange_rate/
  m4/
  MSL/
  Multivariate_ts/
  PSM/
  SMAP/
  SMD/
  SWaT/
  traffic/
  weather/
```

## Run

Please use `python main.py` to run the experiments. Please use the `-h` or `--help` argument for details.

### Long-Term Forecasting

Example training commands:
* Run all benchmarks
    ```bash
    python main.py ltf

    # equivalent
    python main.py ltf --dataset all --pred_len all
    ```
* Run specific benchmarks
    ```bash
    python main.py ltf --dataset etth1 etth2 --pred_len 96 192 336
    ```
Logs, results, and model checkpoints will be saved in `/path/to/MSD-Mixer/logs/ltf`

### Short-Term Forecasting

Example training commands:
* Run all benchmarks
    ```bash
    python main.py stf

    # equivalent
    python main.py stf --dataset all
    ```
* Run specific benchmarks
    ```bash
    python main.py stf --dataset yearly quarterly monthly
    ```
Logs, results, and model checkpoints will be saved in `/path/to/MSD-Mixer/logs/stf`

### Imputation

Example training commands:
* Run all benchmarks
    ```bash
    python main.py imp

    # equivalent
    python main.py imp --dataset all --mask_rate all
    ```
* Run specific benchmarks
    ```bash
    python main.py imp --dataset ecl ettm1 --mask_rate 0.25 0.5
    ```
Logs, results, and model checkpoints will be saved in `/path/to/MSD-Mixer/logs/imp`

### Anomaly Detection

Example training commands:
* Run all benchmarks
    ```bash
    python main.py ad

    # equivalent
    python main.py ad --dataset all
    ```
* Run specific benchmarks
    ```bash
    python main.py ad --dataset smd msl swat
    ```
Logs, results, and model checkpoints will be saved in `/path/to/MSD-Mixer/logs/ad`

### Classification

Example training commands:
* Run all benchmarks
    ```bash
    python main.py cls

    # equivalent
    python main.py cls --dataset all
    ```
* Run specific benchmarks
    ```bash
    python main.py cls --dataset awr scp1 scp2
    ```
Logs, results, and model checkpoints will be saved in `/path/to/MSD-Mixer/logs/cls`

## Acknowledgements

* [Time Series Library (TSlib)](https://github.com/thuml/Time-Series-Library): datasets, experiment settings, and data processing
* [UEA Time Series Classification datasets](https://www.timeseriesclassification.com/): datasets

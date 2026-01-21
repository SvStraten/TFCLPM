# TFCLPM

This is the official GitHub repository for Task-Free diversity-aware Continual Learning for predictive Process Mining (TFCLPM). 

![The Framework](framework.png)

## Datasets

All the datasets provided in the paper can be downloaded from the following [Google Drive link](https://drive.google.com/drive/folders/1HT0_BM1AvMBQOpQglEH8xoT7NiiSaqVG?usp=share_link). After downloading, locate the .csv files in the 'Data' folder.

## Hyperparameter Setup for Competitors
Window Size (* means that the value is fixed because of the nature of the method).

| Dataset | w = Last Drift * | w = 1 * | DynaTrainCDD | Incremental | FullRetrain | DDM |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **IRO5000** | 1 month | 1 month | 500 | 500 | 500 | 500 |
| **ORI5000** | 1 month | 1 month | 500 | 500 | 500 | 500 |
| **RIO5000** | 1 month | 1 month | 500 | 500 | 500 | 500 |
| **ROI5000** | 1 month | 1 month | 500 | 500 | 500 | 500 |
| **OIR5000** | 1 month | 1 month | 500 | 500 | 500 | 500 |
| **InternationalDeclarations** | 1 month | 1 month | 500 | 500 | 500 | 500 |
| **DomesticDeclarations** | 1 month | 1 month | 500 | 500 | 500 | 500 |
| **RequestForPayment** | 1 month | 1 month | 500 | 500 | 500 | 500 |
| **BPIC 2017** | 1 month | 1 month | 500 | 500 | 500 | 500 |
| **BPIC 2015** | 1 month | 1 month | 500 | 500 | 500 | 500 |
| **HelpdeskDrift** | 1 month | 1 month | 500 | 500 | 500 | 500 |
| **RecurrentRequest** | 1 month | 1 month | 500 | 500 | 500 | 500 |

## Installation
To install the required libraries, run the following command in your terminal:

```bash
pip install -r requirements.txt
## Installation
To install the required libraries, run the following command in your terminal:

```bash
pip install -r requirements.txt

## Installation
To install the required libraries, run the following command in your terminal:

```bash
pip -r install requirements.txt
```

## Command-Line Arguments

This script accepts several command-line arguments:

```
--dataset| "Data/RecurrentRequest.csv" | Path to the dataset (CSV file). 
--method | str | "TFCLPM" | Prediction method to use. 
--recent_buffer_size | int | 500 | Recent buffer size.
--hard_buffer_size | int | 100 | Hard buffer size.
--history_buffer_size | int | 300 | History buffer size.
--MAS_weight | float | 0.5 | MAS weight.
--history_buffer | bool | True | Whether to use history buffer (True/False).
--repetitions | int | 5 | Number of repetitions.
```

## **Example Usage**
```
python Model/main.py --dataset "Data/HelpdeskDrift.csv"
```

Note that if you want to exclude the Hard Buffer, you need to uncomment 'Online Continual No Hardbuffer' instead of 'Online Continual'.


# Identify Video Scene Boundaries using Transformer Encoding Linker Network(TELNet)

## Dataset and Feature Download Link
Please download the BBC Dataset, MSC Dataset, OVSD Dataset and their shot features from [here](https://google.com). Afterwards, reference the directory tree and modify config files to make sure directories are valid. 

### Directory tree
```
TELNet
├── README.md
├── main.py
├── eval.py
├── tools.py
├── coverage_overflow.py
├── boundary_plot.py
├── model
│   ├── layer_norm.py
│   ├── TELNet_Model.py
├── final_result
│   ├── bbc_result
│   ├── msc_result
│   └── ovsd_result
├── BBC_Earth_Dataset (Put the dataset here)
├── MSC_Dataset (Put the dataset here)
└── OVSD_Dataset (Put the dataset here)
```

## How to train on dataset
Use **main.py** to train datset in different modes. For **BBC** and **OVSD** datasets, use **"leave one out"** method for training. For **MSC** dataset, use **"train test split"** for training.  

parsers:
--type : "cross", "leave_one_out", "train_test_split"
--dataset : "bbc", "ovsd", "msc"

```
python main.py --type {type_name} --dataset {dataset_name}
```

## How to evaluate model on different datasets
Use **eval.py** to evaluate datset using weights trained for different dataset.

parsers:
--dataset : "bbc", "ovsd", "msc"
```
python eval.py --dataset {dataset_name}
```

## Results

### BBC Dataset
| Video \ Model |  ACRNet  |  TELNET  |
|:-------------:|:--------:|:--------:|
|      B01      | **0.83** |   0.77   |
|      B02      | **0.82** |   0.68   |
|      B03      | **0.77** |   0.69   |
|      B04      |   0.72   | **0.75** |
|      B05      |   0.70   | **0.74** |
|      B06      |   0.70   | **0.75** |
|      B07      |   0.70   | **0.74** |
|      B08      |   0.73   | **0.76** |
|      B09      | **0.80** |   0.70   |
|      B10      |   0.75   | **0.77** |
|      B11      |   0.71   | **0.77** |
|    Average    | **0.76** |   0.74   |

### OVSD Dataset
| Video \ Model | (Trojahn and Goularte 2021) | (Rotman et al.2020)OSG-Triplet | (Rotman, Porat, and Ashour 2017)OSG |  ACRNet  | TELNET_V2 |
|:-------------:|:---------------------------:|:------------------------------:|:-----------------------------------:|:--------:|:---------:|
|      BBB      |            0.57             |              0.81              |              **0.83**               |   0.74   |   0.69    |
|     BWNS      |            0.53             |            **0.75**            |                0.63                 |          |   0.60    |
|      CL       |            0.64             |              0.49              |                0.62                 |   0.61   | **0.88**  |
|      FBW      |            0.57             |            **0.76**            |                0.57                 |          |   0.66    |
|     Honey     |            0.60             |              0.73              |                0.58                 |          | **0.77**  |
|     LCDUP     |            0.63             |              0.73              |                0.72                 |          | **0.76**  |
|   Meridian    |            0.45             |            **0.69**            |                0.63                 |          | **0.75**  |
|   Route 66    |            0.63             |            **0.72**            |                0.54                 |          |   0.64    |
|  Star Wreck   |            0.62             |              0.66              |                0.55                 |          | **0.71**  |
|    Average    |            0.58             |              0.70              |                0.61                 | **0.73** |   0.72    |

### Cross dataset

#### F-score of ACRNet on cross dataset
| Train \ Test |   MSC    |   BBC    |   OVSD   |
|:------------:|:--------:|:--------:|:--------:|
|     MSC      | **0.67** |   0.64   |   0.63   |
|     BBC      |   0.28   | **0.76** |   0.22   |
|     OVSD     |   0.29   |   0.23   | **0.73** |

#### F-score of TELNet on cross dataset
| Train \ Test |   MSC    |   BBC    |   OVSD   |
|:------------:|:--------:|:--------:|:--------:|
|     MSC      | **0.69** |   0.62   |   0.60   |
|     BBC      | **0.64** |   0.74   | **0.56** |
|     OVSD     | **0.64** | **0.64** |   0.72   |
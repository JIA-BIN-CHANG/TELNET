# TELNET
TELNET with merging algorithm version

## main.py
Use to train datset in two different mode
parsers:
--type : "cross" or "leave_one_out"
--dataset : "bbc", "ovsd", "msc"
```
python main.py --type {type_name} --dataset {dataset_name}
```

## eval.py
Use to evaluate datset using weights trained for different dataset
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
|      B05      |   0.7    | **0.74** |
|      B06      |   0.7    | **0.75** |
|      B07      |   0.7    | **0.74** |
|      B08      |   0.73   | **0.76** |
|      B09      |   0.8    |   0.70   |
|      B10      |   0.75   | **0.77** |
|      B11      |   0.71   | **0.77** |
|    Average    |   0.76   |   0.74   |

### OVSD Dataset
| Video \ Model | (Trojahn and Goularte 2021) | (Rotman et al.2020)OSG-Triplet | (Rotman, Porat, and Ashour 2017)OSG |  ACRNet  | TELNET_V2 |
|:-------------:|:---------------------------:|:------------------------------:|:-----------------------------------:|:--------:|:---------:|
|      BBB      |            0.57             |              0.81              |              **0.83**               |   0.74   |   0.69    |
|     BWNS      |            0.53             |            **0.75**            |                0.63                 |          |   0.60    |
|      CL       |            0.64             |              0.49              |                0.62                 |   0.61   | **0.88**  |
|      FBW      |            0.57             |            **0.76**            |                0.57                 |          |   0.66    |
|     Honey     |            0.60             |              0.73              |                0.58                 |          | **0.77**  |
|     LCDUP     |                             |                                |                                     |          |   0.76    |
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
| Train \ Test | MSC | BBC  | OVSD |
|:------------:|:---:|:----:|:----:|
|     MSC      |     |      |      |
|     BBC      |     | 0.74 | 0.57 |
|     OVSD     |     | 0.62 | 0.72 |
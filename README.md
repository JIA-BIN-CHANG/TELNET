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

## tools.py

## eval.py
Use to evaluate datset using weights trained for different dataset
parsers:
--dataset : "bbc", "ovsd", "msc"
```
python eval.py --dataset {dataset_name}
```

## Results
| Train \ Test | MSC | BBC  | OVSD |
|:------------:|:---:|:----:|:----:|
|     MSC      |     |      |      |
|     BBC      |     | 0.74 | 0.57 |
|     OVSD     |     | 0.62 | 0.72 |
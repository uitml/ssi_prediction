## Create data

sh Prepare_Data.sh

## Parameters

```
python compute_{sgd|gboost}.py {prz|org|aug}
features: prz (Kocbek), org (weekly w/o augmentation), aug (weekly with augmentation)
```
Options
 -tcv 10   10 fold CV train/test
 -vcv 5    5 fold CV train/valid
 -i 100    100 iterations
 -ovs      over_sampling
 -std      standardize
 -l        log_sapce
 -t 6      truncate weekly features at 5 "windows"
 -as 10    number of augmented samples
 -g {a|f|m} restrict gender
 -r 0      random seed
 -v        verbose or not
 -s {1|0}  save or not
 
 -ab {bin|ssi}   ONLY FOR ABLATION: select a class. Binary is -ab 1,2


## Compute Tables
```
python compute_sgd.py prz -std -ovs -l -i 100 ;
python compute_sgd.py org -std -ovs -l -i 100 ;

python compute_gboost.py prz -std -ovs -l -i 100 ;
python compute_gboost.py org -std -ovs -l -i 100 ;

python compute_sgd.py org -std -ovs -i 100 ;
python compute_sgd.py org -std -l -i 100 ;
python compute_sgd.py org -std -i 100 ;

python compute_gboost.py org -std -ovs -i 100 ;
python compute_gboost.py org -std -l -i 100 ;
python compute_gboost.py org -std -i 100 ;
```
Print tables only for -i 100 -tcv 10 -vcv 5
```
python print_tables.py {--latex}
```
## Compute Ablations
# Only for GBoost
```
python compute_ablation_gboost.py org -std -ovs -l -i 10 -ab bin ;
python compute_ablation_gboost.py org -std -ovs -l -i 10 -ab ssi ;
```
```
python plot_ablation_gboost.py
```

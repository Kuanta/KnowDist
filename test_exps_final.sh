#!/bin/bash

EXP_NO=$1
: '
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=3 --fuzzy_type=1 --dataset=1 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=3 --fuzzy_type=2 --dataset=1 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=3 --fuzzy_type=2 --dataset=1 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=3 --fuzzy_type=2 --dataset=1 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=0
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=1 --dataset=1 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=2 --dataset=1 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=2 --dataset=1 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=2 --dataset=1 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=0

python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=0
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=0

python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=15 --fuzzy_type=1 --dataset=1 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=15 --fuzzy_type=2 --dataset=1 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=15 --fuzzy_type=2 --dataset=1 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=15 --fuzzy_type=2 --dataset=1 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=0
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=1 --dataset=1 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=2 --dataset=1 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=2 --dataset=1 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=2 --dataset=1 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=0

python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=3 --fuzzy_type=1 --dataset=2 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=3 --fuzzy_type=2 --dataset=2 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=3 --fuzzy_type=2 --dataset=2 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=3 --fuzzy_type=2 --dataset=2 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=0
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=1 --dataset=2 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=2 --dataset=2 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=2 --dataset=2 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=2 --dataset=2 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=0

python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=7 --fuzzy_type=1 --dataset=2 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=7 --fuzzy_type=2 --dataset=2 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=7 --fuzzy_type=2 --dataset=2 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=7 --fuzzy_type=2 --dataset=2 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=0
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=2 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=2 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=2 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=2 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=0

python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=15 --fuzzy_type=1 --dataset=2 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=15 --fuzzy_type=2 --dataset=2 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=15 --fuzzy_type=2 --dataset=2 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=15 --fuzzy_type=2 --dataset=2 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=0
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=1 --dataset=2 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=2 --dataset=2 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=2 --dataset=2 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=2 --dataset=2 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=0

python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=3 --fuzzy_type=1 --dataset=1 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=3 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=3 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=3 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=1 --dataset=1 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0

python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0

python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=15 --fuzzy_type=1 --dataset=1 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=15 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=15 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=15 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=1 --dataset=1 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0

python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=3 --fuzzy_type=1 --dataset=2 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=3 --fuzzy_type=2 --dataset=2 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=3 --fuzzy_type=2 --dataset=2 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=3 --fuzzy_type=2 --dataset=2 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=1 --dataset=2 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=2 --dataset=2 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=2 --dataset=2 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=2 --dataset=2 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0

python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=7 --fuzzy_type=1 --dataset=2 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=7 --fuzzy_type=2 --dataset=2 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=7 --fuzzy_type=2 --dataset=2 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=7 --fuzzy_type=2 --dataset=2 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=2 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=2 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=2 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=2 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0

python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=15 --fuzzy_type=1 --dataset=2 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=15 --fuzzy_type=2 --dataset=2 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=15 --fuzzy_type=2 --dataset=2 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=15 --fuzzy_type=2 --dataset=2 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=1 --dataset=2 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=2 --dataset=2 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=2 --dataset=2 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=2 --dataset=2 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0

python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=3 --fuzzy_type=1 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=3 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=3 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=3 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=1 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0

python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0

python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=15 --fuzzy_type=1 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=15 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=15 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=15 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=1 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0

python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=3 --fuzzy_type=1 --dataset=2 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=3 --fuzzy_type=2 --dataset=2 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=3 --fuzzy_type=2 --dataset=2 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=3 --fuzzy_type=2 --dataset=2 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=1 --dataset=2 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=2 --dataset=2 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=2 --dataset=2 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=2 --dataset=2 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0

python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=7 --fuzzy_type=1 --dataset=2 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=7 --fuzzy_type=2 --dataset=2 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=7 --fuzzy_type=2 --dataset=2 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=7 --fuzzy_type=2 --dataset=2 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=2 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=2 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=2 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=2 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0

python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=15 --fuzzy_type=1 --dataset=2 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=15 --fuzzy_type=2 --dataset=2 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=15 --fuzzy_type=2 --dataset=2 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=15 --fuzzy_type=2 --dataset=2 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=1 --dataset=2 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=2 --dataset=2 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=2 --dataset=2 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=2 --dataset=2 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0
'
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=3 --fuzzy_type=1 --dataset=4 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=3 --fuzzy_type=2 --dataset=4 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=3 --fuzzy_type=2 --dataset=4 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=3 --fuzzy_type=2 --dataset=4 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=0
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=1 --dataset=4 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=2 --dataset=4 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=2 --dataset=4 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=2 --dataset=4 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=0

python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=7 --fuzzy_type=1 --dataset=4 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=7 --fuzzy_type=2 --dataset=4 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=7 --fuzzy_type=2 --dataset=4 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=7 --fuzzy_type=2 --dataset=4 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=0
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=4 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=4 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=4 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=4 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=0

python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=15 --fuzzy_type=1 --dataset=4 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=15 --fuzzy_type=2 --dataset=4 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=15 --fuzzy_type=2 --dataset=4 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=15 --fuzzy_type=2 --dataset=4 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=0
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=1 --dataset=4 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=2 --dataset=4 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=2 --dataset=4 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=2 --dataset=4 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=0

python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=3 --fuzzy_type=1 --dataset=4 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=3 --fuzzy_type=2 --dataset=4 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=3 --fuzzy_type=2 --dataset=4 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=3 --fuzzy_type=2 --dataset=4 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=1 --dataset=4 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=2 --dataset=4 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=2 --dataset=4 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=2 --dataset=4 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0

python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=7 --fuzzy_type=1 --dataset=4 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=7 --fuzzy_type=2 --dataset=4 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=7 --fuzzy_type=2 --dataset=4 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=7 --fuzzy_type=2 --dataset=4 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=4 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=4 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=4 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=4 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0

python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=15 --fuzzy_type=1 --dataset=4 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=15 --fuzzy_type=2 --dataset=4 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=15 --fuzzy_type=2 --dataset=4 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=15 --fuzzy_type=2 --dataset=4 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=1 --dataset=4 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=2 --dataset=4 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=2 --dataset=4 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=2 --dataset=4 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0

python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=3 --fuzzy_type=1 --dataset=4 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=3 --fuzzy_type=2 --dataset=4 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=3 --fuzzy_type=2 --dataset=4 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=3 --fuzzy_type=2 --dataset=4 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=1 --dataset=4 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=2 --dataset=4 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=2 --dataset=4 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=2 --dataset=4 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0

python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=7 --fuzzy_type=1 --dataset=4 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=7 --fuzzy_type=2 --dataset=4 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=7 --fuzzy_type=2 --dataset=4 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=7 --fuzzy_type=2 --dataset=4 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=4 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=4 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=4 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=4 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0

python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=15 --fuzzy_type=1 --dataset=4 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=15 --fuzzy_type=2 --dataset=4 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=15 --fuzzy_type=2 --dataset=4 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.0  --n_rules=15 --fuzzy_type=2 --dataset=4 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=1 --dataset=4 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=2 --dataset=4 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=2 --dataset=4 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=1 --exp_no=$EXP_NO --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=2 --dataset=4 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0
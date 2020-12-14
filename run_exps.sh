#!/bin/bash

python train.py --exp_id=1 --exp_no=1 --student_temp=1 --teacher_temp=5.0 --alpha=0.5 --n_rules=15
python train.py --exp_id=1 --exp_no=2 --student_temp=1 --teacher_temp=5.0 --alpha=0.5 --n_rules=15
python train.py --exp_id=1 --exp_no=3 --student_temp=1 --teacher_temp=5.0 --alpha=0.5 --n_rules=15
python train.py --exp_id=1 --exp_no=4 --student_temp=1 --teacher_temp=5.0 --alpha=0.5 --n_rules=15
python train.py --exp_id=1 --exp_no=5 --student_temp=1 --teacher_temp=5.0 --alpha=0.5 --n_rules=15

python train.py --exp_id=2 --exp_no=1 --student_temp=1 --teacher_temp=7.5 --alpha=0.5 --n_rules=15
python train.py --exp_id=2 --exp_no=2 --student_temp=1 --teacher_temp=7.5 --alpha=0.5 --n_rules=15
python train.py --exp_id=2 --exp_no=3 --student_temp=1 --teacher_temp=7.5 --alpha=0.5 --n_rules=15
python train.py --exp_id=2 --exp_no=4 --student_temp=1 --teacher_temp=7.5 --alpha=0.5 --n_rules=15
python train.py --exp_id=2 --exp_no=5 --student_temp=1 --teacher_temp=7.5 --alpha=0.5 --n_rules=15
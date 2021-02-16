##  ========== MNIST ============
#
## N_inputs = 15
#
## Pure Cross Entropy with Type1
#python train.py --exp_id=1 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=15
#python train.py --exp_id=1 --exp_no=2 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=15
#python train.py --exp_id=1 --exp_no=3 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=15
#python train.py --exp_id=1 --exp_no=4 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=15
#python train.py --exp_id=1 --exp_no=5 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=15
#
# Pure Cross Entropy with Type 2
#python train.py --exp_id=2 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
#python train.py --exp_id=2 --exp_no=2 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15
#python train.py --exp_id=2 --exp_no=3 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15
#python train.py --exp_id=2 --exp_no=4 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15
#python train.py --exp_id=2 --exp_no=5 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15
#python train.py --exp_id=2 --exp_no=6 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1
#python train.py --exp_id=2 --exp_no=7 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0

# Type 1 Knowledge Distillation
#python train.py --exp_id=3 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=15
#python train.py --exp_id=3 --exp_no=2 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=15
#python train.py --exp_id=3 --exp_no=3 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=15
#python train.py --exp_id=3 --exp_no=4 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=15
#python train.py --exp_id=3 --exp_no=5 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=15

# Type 2 Knowledge Distillation (A)
#python train.py --exp_id=4 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
#python train.py --exp_id=4 --exp_no=2 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
#python train.py --exp_id=4 --exp_no=3 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
#python train.py --exp_id=4 --exp_no=4 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
#python train.py --exp_id=4 --exp_no=5 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1

# Type 2 Knowledge Distillation (B)
#python train.py --exp_id=5 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1
#python train.py --exp_id=5 --exp_no=2 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1
#python train.py --exp_id=5 --exp_no=3 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1
#python train.py --exp_id=5 --exp_no=4 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1
#python train.py --exp_id=5 --exp_no=5 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1

# Type 2 Knowledge Distillation (C)
#python train.py --exp_id=6 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0
#python train.py --exp_id=6 --exp_no=2 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0
#python train.py --exp_id=6 --exp_no=3 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0
#python train.py --exp_id=6 --exp_no=4 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0
#python train.py --exp_id=6 --exp_no=5 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0

## N_inputs 30
#
## Pure Cross Entropy with Type1
#python train.py --exp_id=7 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=30
#python train.py --exp_id=7 --exp_no=2 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=30
#python train.py --exp_id=7 --exp_no=3 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=30
#python train.py --exp_id=7 --exp_no=4 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=30
#python train.py --exp_id=7 --exp_no=5 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=30
#
## Pure Cross Entropy with Type 2
#python train.py --exp_id=8 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
#python train.py --exp_id=8 --exp_no=2 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
#python train.py --exp_id=8 --exp_no=3 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
#python train.py --exp_id=8 --exp_no=4 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
#python train.py --exp_id=8 --exp_no=5 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
#python train.py --exp_id=8 --exp_no=6 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
#python train.py --exp_id=8 --exp_no=7 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
#python train.py --exp_id=8 --exp_no=8 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
#python train.py --exp_id=8 --exp_no=9 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
#python train.py --exp_id=8 --exp_no=10 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
#python train.py --exp_id=8 --exp_no=11 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0
#python train.py --exp_id=8 --exp_no=12 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0
#python train.py --exp_id=8 --exp_no=13 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0
#python train.py --exp_id=8 --exp_no=14 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0
#python train.py --exp_id=8 --exp_no=15 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0

## Type 1 Knowledge Distillation
#python train.py --exp_id=9 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=30
#python train.py --exp_id=9 --exp_no=2 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=30
#python train.py --exp_id=9 --exp_no=3 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=30
#python train.py --exp_id=9 --exp_no=4 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=30
#python train.py --exp_id=9 --exp_no=5 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=30
#
## Type 2 Knowledge Distillation (A)
#python train.py --exp_id=10 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
#python train.py --exp_id=10 --exp_no=2 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
#python train.py --exp_id=10 --exp_no=3 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
#python train.py --exp_id=10 --exp_no=4 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
#python train.py --exp_id=10 --exp_no=5 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
#
## Type 2 Knowledge Distillation (B)
#python train.py --exp_id=11 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
#python train.py --exp_id=11 --exp_no=2 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
#python train.py --exp_id=11 --exp_no=3 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
#python train.py --exp_id=11 --exp_no=4 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
#python train.py --exp_id=11 --exp_no=5 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
#
## Type 2 Knowledge Distillation (C)
#python train.py --exp_id=12 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0
#python train.py --exp_id=12 --exp_no=2 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0
#python train.py --exp_id=12 --exp_no=3 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0
#python train.py --exp_id=12 --exp_no=4 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0
#python train.py --exp_id=12 --exp_no=5 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0
#
## ================ Cifar ==========================
#
## N_inputs = 15
#
## Pure Cross Entropy with Type1
#python train.py --exp_id=13 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=15
#python train.py --exp_id=13 --exp_no=2 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=15
#python train.py --exp_id=13 --exp_no=3 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=15
#python train.py --exp_id=13 --exp_no=4 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=15
#python train.py --exp_id=13 --exp_no=5 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=15
#
## Pure Cross Entropy with Type 2
#python train.py --exp_id=14 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15
#python train.py --exp_id=14 --exp_no=2 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15
#python train.py --exp_id=14 --exp_no=3 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15
#python train.py --exp_id=14 --exp_no=4 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15
#python train.py --exp_id=14 --exp_no=5 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15
#
## Type 1 Knowledge Distillation
#python train.py --exp_id=15 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=15
#python train.py --exp_id=15 --exp_no=2 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=15
#python train.py --exp_id=15 --exp_no=3 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=15
#python train.py --exp_id=15 --exp_no=4 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=15
#python train.py --exp_id=15 --exp_no=5 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=15
#
## Type 2 Knowledge Distillation (A)
#python train.py --exp_id=16 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
#python train.py --exp_id=16 --exp_no=2 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
#python train.py --exp_id=16 --exp_no=3 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
#python train.py --exp_id=16 --exp_no=4 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
#python train.py --exp_id=16 --exp_no=5 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
#
## Type 2 Knowledge Distillation (B)
#python train.py --exp_id=17 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1
#python train.py --exp_id=17 --exp_no=2 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1
#python train.py --exp_id=17 --exp_no=3 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1
#python train.py --exp_id=17 --exp_no=4 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1
#python train.py --exp_id=17 --exp_no=5 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1
#
## Type 2 Knowledge Distillation (C)
#python train.py --exp_id=18 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0
#python train.py --exp_id=18 --exp_no=2 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0
#python train.py --exp_id=18 --exp_no=3 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0
#python train.py --exp_id=18 --exp_no=4 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0
#python train.py --exp_id=18 --exp_no=5 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0
#
## N_inputs 30
#
## Pure Cross Entropy with Type1
#python train.py --exp_id=19 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=1 --dataset=2 --n_inputs=30
#python train.py --exp_id=19 --exp_no=2 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=1 --dataset=2 --n_inputs=60
#python train.py --exp_id=19 --exp_no=3 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=1 --dataset=2 --n_inputs=120
#python train.py --exp_id=19 --exp_no=2 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=30
#python train.py --exp_id=19 --exp_no=3 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=30
#python train.py --exp_id=19 --exp_no=4 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=30
#python train.py --exp_id=19 --exp_no=5 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=30
#
## Pure Cross Entropy with Type 2
#python train.py --exp_id=20 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=2 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
#python train.py --exp_id=20 --exp_no=2 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
#python train.py --exp_id=20 --exp_no=3 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
#python train.py --exp_id=20 --exp_no=4 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
#python train.py --exp_id=20 --exp_no=5 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
#python train.py --exp_id=20 --exp_no=6 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=2 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
#python train.py --exp_id=20 --exp_no=7 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
#python train.py --exp_id=20 --exp_no=8 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
#python train.py --exp_id=20 --exp_no=9 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
#python train.py --exp_id=20 --exp_no=10 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
#python train.py --exp_id=20 --exp_no=12 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0
#python train.py --exp_id=20 --exp_no=13 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0
#python train.py --exp_id=20 --exp_no=14 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0
#python train.py --exp_id=20 --exp_no=11 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=2 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0
#python train.py --exp_id=20 --exp_no=15 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0

## Type 1 Knowledge Distillation
#python train.py --exp_id=21 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=2 --n_inputs=30
#python train.py --exp_id=21 --exp_no=2 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=30
#python train.py --exp_id=21 --exp_no=3 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=30
#python train.py --exp_id=21 --exp_no=4 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=30
#python train.py --exp_id=21 --exp_no=5 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=30
#
## Type 2 Knowledge Distillation (A)
#python train.py --exp_id=22 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=2 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
#python train.py --exp_id=22 --exp_no=2 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
#python train.py --exp_id=22 --exp_no=3 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
#python train.py --exp_id=22 --exp_no=4 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
#python train.py --exp_id=22 --exp_no=5 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
#
## Type 2 Knowledge Distillation (B)
#python train.py --exp_id=23 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=2 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
#python train.py --exp_id=23 --exp_no=2 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
#python train.py --exp_id=23 --exp_no=3 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
#python train.py --exp_id=23 --exp_no=4 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
#python train.py --exp_id=23 --exp_no=5 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
#
## Type 2 Knowledge Distillation (C)
#python train.py --exp_id=24 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=2 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0
#python train.py --exp_id=24 --exp_no=2 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0
#python train.py --exp_id=24 --exp_no=3 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0
#python train.py --exp_id=24 --exp_no=4 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0
#python train.py --exp_id=24 --exp_no=5 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0

#python train.py --exp_id=27 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=7 --fuzzy_type=1 --dataset=2 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
#python train.py --exp_id=28 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=7 --fuzzy_type=2 --dataset=2 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
#python train.py --exp_id=29 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=7 --fuzzy_type=2 --dataset=2 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
#python train.py --exp_id=30 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=7 --fuzzy_type=2 --dataset=2 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0
# python train.py --exp_id=31 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=2 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
# python train.py --exp_id=32 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=2 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
# python train.py --exp_id=33 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=2 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
# python train.py --exp_id=34 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=2 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0
#
# python train.py --exp_id=35 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=7 --fuzzy_type=1 --dataset=2 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
# python train.py --exp_id=36 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=7 --fuzzy_type=2 --dataset=2 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
# python train.py --exp_id=37 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=7 --fuzzy_type=2 --dataset=2 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=1
# python train.py --exp_id=38 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=7 --fuzzy_type=2 --dataset=2 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=0
# python train.py --exp_id=39 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=2 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
# python train.py --exp_id=40 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=2 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
# python train.py --exp_id=41 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=2 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=1
# python train.py --exp_id=42 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=2 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=0
#
#python train.py --exp_id=43 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=3 --fuzzy_type=1 --dataset=2 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
#python train.py --exp_id=44 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=3 --fuzzy_type=2 --dataset=2 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
#python train.py --exp_id=45 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=3 --fuzzy_type=2 --dataset=2 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=1
#python train.py --exp_id=46 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=3 --fuzzy_type=2 --dataset=2 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=0
#python train.py --exp_id=47 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=1 --dataset=2 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
#python train.py --exp_id=48 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=2 --dataset=2 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
#python train.py --exp_id=49 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=2 --dataset=2 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=1
#python train.py --exp_id=50 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=2 --dataset=2 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=0

# python train.py --exp_id=51 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=3 --fuzzy_type=1 --dataset=1 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
# python train.py --exp_id=52 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=3 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
# python train.py --exp_id=53 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=3 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1
# python train.py --exp_id=54 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=3 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0
# python train.py --exp_id=55 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=1 --dataset=1 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
# python train.py --exp_id=56 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
# python train.py --exp_id=57 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1
# python train.py --exp_id=58 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0
#
# python train.py --exp_id=59 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
# python train.py --exp_id=60 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
# python train.py --exp_id=61 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1
# python train.py --exp_id=62 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0
# python train.py --exp_id=63 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
# python train.py --exp_id=64 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
# python train.py --exp_id=65 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1
# python train.py --exp_id=66 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0
#
# python train.py --exp_id=67 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=15 --fuzzy_type=1 --dataset=1 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
# python train.py --exp_id=68 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=15 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
# python train.py --exp_id=69 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=15 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1
# python train.py --exp_id=70 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=15 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0
# python train.py --exp_id=71 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=1 --dataset=1 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
# python train.py --exp_id=72 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
# python train.py --exp_id=73 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1
# python train.py --exp_id=74 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0
#
# python train.py --exp_id=75 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=3 --fuzzy_type=1 --dataset=2 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
# python train.py --exp_id=76 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=3 --fuzzy_type=2 --dataset=2 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
# python train.py --exp_id=77 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=3 --fuzzy_type=2 --dataset=2 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1
# python train.py --exp_id=78 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=3 --fuzzy_type=2 --dataset=2 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0
# python train.py --exp_id=79 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=1 --dataset=2 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
# python train.py --exp_id=80 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=2 --dataset=2 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
# python train.py --exp_id=81 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=2 --dataset=2 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1
# python train.py --exp_id=82 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=2 --dataset=2 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0
#
# python train.py --exp_id=83 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=7 --fuzzy_type=1 --dataset=2 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
# python train.py --exp_id=84 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=7 --fuzzy_type=2 --dataset=2 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
# python train.py --exp_id=85 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=7 --fuzzy_type=2 --dataset=2 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1
# python train.py --exp_id=86 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=7 --fuzzy_type=2 --dataset=2 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0
# python train.py --exp_id=87 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=2 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
# python train.py --exp_id=88 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=2 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
# python train.py --exp_id=89 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=2 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1
# python train.py --exp_id=90 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=2 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0
#
# python train.py --exp_id=91 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=15 --fuzzy_type=1 --dataset=2 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
# python train.py --exp_id=92 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=15 --fuzzy_type=2 --dataset=2 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
# python train.py --exp_id=93 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=15 --fuzzy_type=2 --dataset=2 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1
# python train.py --exp_id=94 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=15 --fuzzy_type=2 --dataset=2 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0
# python train.py --exp_id=95 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=1 --dataset=2 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
# python train.py --exp_id=96 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=2 --dataset=2 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
# python train.py --exp_id=97 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=2 --dataset=2 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1
# python train.py --exp_id=98 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=2 --dataset=2 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0

# ==============
python train.py --exp_id=99 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=3 --fuzzy_type=1 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python train.py --exp_id=100 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=3 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python train.py --exp_id=101 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=3 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
python train.py --exp_id=102 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=3 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0
python train.py --exp_id=103 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=1 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python train.py --exp_id=104 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python train.py --exp_id=105 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
python train.py --exp_id=106 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0

python train.py --exp_id=107 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python train.py --exp_id=108 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python train.py --exp_id=109 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
python train.py --exp_id=110 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0
python train.py --exp_id=111 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python train.py --exp_id=112 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python train.py --exp_id=113 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
python train.py --exp_id=114 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0

python train.py --exp_id=115 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=15 --fuzzy_type=1 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python train.py --exp_id=116 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=15 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python train.py --exp_id=117 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=15 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
python train.py --exp_id=118 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=15 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0
python train.py --exp_id=119 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=1 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python train.py --exp_id=120 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python train.py --exp_id=121 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
python train.py --exp_id=122 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0

python train.py --exp_id=123 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=3 --fuzzy_type=1 --dataset=2 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python train.py --exp_id=124 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=3 --fuzzy_type=2 --dataset=2 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python train.py --exp_id=125 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=3 --fuzzy_type=2 --dataset=2 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
python train.py --exp_id=126 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=3 --fuzzy_type=2 --dataset=2 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0
python train.py --exp_id=127 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=1 --dataset=2 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python train.py --exp_id=128 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=2 --dataset=2 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python train.py --exp_id=129 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=2 --dataset=2 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
python train.py --exp_id=130 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=2 --dataset=2 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0

python train.py --exp_id=131 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=7 --fuzzy_type=1 --dataset=2 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python train.py --exp_id=132 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=7 --fuzzy_type=2 --dataset=2 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python train.py --exp_id=133 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=7 --fuzzy_type=2 --dataset=2 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
python train.py --exp_id=134 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=7 --fuzzy_type=2 --dataset=2 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0
python train.py --exp_id=135 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=2 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python train.py --exp_id=136 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=2 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python train.py --exp_id=137 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=2 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
python train.py --exp_id=138 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=2 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0

python train.py --exp_id=139 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=15 --fuzzy_type=1 --dataset=2 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python train.py --exp_id=140 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=15 --fuzzy_type=2 --dataset=2 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python train.py --exp_id=141 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=15 --fuzzy_type=2 --dataset=2 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
python train.py --exp_id=142 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=15 --fuzzy_type=2 --dataset=2 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0
python train.py --exp_id=143 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=1 --dataset=2 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python train.py --exp_id=144 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=2 --dataset=2 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python train.py --exp_id=145 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=2 --dataset=2 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
python train.py --exp_id=146 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=2 --dataset=2 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0

python train.py --exp_id=145 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=3 --fuzzy_type=1 --dataset=1 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python train.py --exp_id=147 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=3 --fuzzy_type=2 --dataset=1 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python train.py --exp_id=148 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=3 --fuzzy_type=2 --dataset=1 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=1
python train.py --exp_id=149 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=3 --fuzzy_type=2 --dataset=1 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=0
python train.py --exp_id=150 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=1 --dataset=1 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python train.py --exp_id=151 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=2 --dataset=1 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python train.py --exp_id=152 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=2 --dataset=1 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=1
python train.py --exp_id=153 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=2 --dataset=1 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=0

python train.py --exp_id=154 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python train.py --exp_id=155 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python train.py --exp_id=156 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=1
python train.py --exp_id=157 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=0
python train.py --exp_id=158 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python train.py --exp_id=159 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python train.py --exp_id=160 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=1
python train.py --exp_id=161 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=0

python train.py --exp_id=115 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=15 --fuzzy_type=1 --dataset=1 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python train.py --exp_id=116 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=15 --fuzzy_type=2 --dataset=1 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python train.py --exp_id=117 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=15 --fuzzy_type=2 --dataset=1 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=1
python train.py --exp_id=118 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=15 --fuzzy_type=2 --dataset=1 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=0
python train.py --exp_id=119 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=1 --dataset=1 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python train.py --exp_id=120 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=2 --dataset=1 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python train.py --exp_id=121 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=2 --dataset=1 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=1
python train.py --exp_id=122 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=15 --fuzzy_type=2 --dataset=1 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=0




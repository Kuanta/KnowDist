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

python train.py --exp_id=43 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=3 --fuzzy_type=1 --dataset=2 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python train.py --exp_id=44 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=3 --fuzzy_type=2 --dataset=2 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python train.py --exp_id=45 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=3 --fuzzy_type=2 --dataset=2 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=1
python train.py --exp_id=46 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.0 --n_rules=3 --fuzzy_type=2 --dataset=2 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=0
python train.py --exp_id=47 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=1 --dataset=2 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python train.py --exp_id=48 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=2 --dataset=2 --n_inputs=5 --use_sigma_scale=0 --use_height_scale=1
python train.py --exp_id=49 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=2 --dataset=2 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=1
python train.py --exp_id=50 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=3 --fuzzy_type=2 --dataset=2 --n_inputs=5 --use_sigma_scale=1 --use_height_scale=0
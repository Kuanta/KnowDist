#  ========== MNIST ============

# N_inputs = 15

# Pure Cross Entropy with Type1
python test.py --exp_id=1 --exp_no=1 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=15
python test.py --exp_id=1 --exp_no=2 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=15
python test.py --exp_id=1 --exp_no=3 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=15
python test.py --exp_id=1 --exp_no=4 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=15
python test.py --exp_id=1 --exp_no=5 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=15
# Pure Cross Entropy with Type 2
python test.py --exp_id=2 --exp_no=1 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=2 --exp_no=2 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15
python test.py --exp_id=2 --exp_no=3 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15
python test.py --exp_id=2 --exp_no=4 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15
python test.py --exp_id=2 --exp_no=5 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15
python test.py --exp_id=2 --exp_no=6 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=2 --exp_no=7 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0
# Type 1 Knowledge Distillation
python test.py --exp_id=3 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=15
python test.py --exp_id=3 --exp_no=2 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=15
python test.py --exp_id=3 --exp_no=3 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=15
python test.py --exp_id=3 --exp_no=4 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=15
python test.py --exp_id=3 --exp_no=5 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=15
# Type 2 Knowledge Distillation (A)
python test.py --exp_id=4 --exp_no=1 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=4 --exp_no=2 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=4 --exp_no=3 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=4 --exp_no=4 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=4 --exp_no=5 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
# Type 2 Knowledge Distillation (B)
python test.py --exp_id=5 --exp_no=1 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=5 --exp_no=2 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=5 --exp_no=3 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=5 --exp_no=4 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=5 --exp_no=5 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1
# Type 2 Knowledge Distillation (C)
python test.py --exp_id=6 --exp_no=1 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0
python test.py --exp_id=6 --exp_no=2 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0
python test.py --exp_id=6 --exp_no=3 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0
python test.py --exp_id=6 --exp_no=4 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0
python test.py --exp_id=6 --exp_no=5 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0
# N_inputs 30

# Pure Cross Entropy with Type1
python test.py --exp_id=7 --exp_no=1 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=30
python test.py --exp_id=7 --exp_no=2 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=30
python test.py --exp_id=7 --exp_no=3 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=30
python test.py --exp_id=7 --exp_no=4 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=30
python test.py --exp_id=7 --exp_no=5 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=30

# Pure Cross Entropy with Type 2
python test.py --exp_id=8 --exp_no=1 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30
python test.py --exp_id=8 --exp_no=2 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30
python test.py --exp_id=8 --exp_no=3 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30
python test.py --exp_id=8 --exp_no=4 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30
python test.py --exp_id=8 --exp_no=5 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30

# Type 1 Knowledge Distillation
python test.py --exp_id=9 --exp_no=1 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=30
python test.py --exp_id=9 --exp_no=2 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=30
python test.py --exp_id=9 --exp_no=3 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=30
python test.py --exp_id=9 --exp_no=4 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=30
python test.py --exp_id=9 --exp_no=5 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=30

# Type 2 Knowledge Distillation (A)
python test.py --exp_id=10 --exp_no=1 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=10 --exp_no=2 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=10 --exp_no=3 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=10 --exp_no=4 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=10 --exp_no=5 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1

# Type 2 Knowledge Distillation (B)
python test.py --exp_id=11 --exp_no=1 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=11 --exp_no=2 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=11 --exp_no=3 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=11 --exp_no=4 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=11 --exp_no=5 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1

# Type 2 Knowledge Distillation (C)
python test.py --exp_id=12 --exp_no=1 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0
python test.py --exp_id=12 --exp_no=2 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0
python test.py --exp_id=12 --exp_no=3 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0
python test.py --exp_id=12 --exp_no=4 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0
python test.py --exp_id=12 --exp_no=5 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0

# ================ Cifar ==========================

# N_inputs = 15

# Pure Cross Entropy with Type1
python test.py --exp_id=13 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=15
python test.py --exp_id=13 --exp_no=2 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=15
python test.py --exp_id=13 --exp_no=3 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=15
python test.py --exp_id=13 --exp_no=4 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=15
python test.py --exp_id=13 --exp_no=5 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=15

# Pure Cross Entropy with Type 2
python test.py --exp_id=14 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15
python test.py --exp_id=14 --exp_no=2 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15
python test.py --exp_id=14 --exp_no=3 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15
python test.py --exp_id=14 --exp_no=4 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15
python test.py --exp_id=14 --exp_no=5 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15

# Type 1 Knowledge Distillation
python test.py --exp_id=15 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=15
python test.py --exp_id=15 --exp_no=2 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=15
python test.py --exp_id=15 --exp_no=3 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=15
python test.py --exp_id=15 --exp_no=4 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=15
python test.py --exp_id=15 --exp_no=5 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=15

# Type 2 Knowledge Distillation (A)
python test.py --exp_id=16 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=16 --exp_no=2 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=16 --exp_no=3 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=16 --exp_no=4 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=16 --exp_no=5 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=0 --use_height_scale=1

# Type 2 Knowledge Distillation (B)
python test.py --exp_id=17 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=17 --exp_no=2 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=17 --exp_no=3 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=17 --exp_no=4 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=17 --exp_no=5 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=1

# Type 2 Knowledge Distillation (C)
python test.py --exp_id=18 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0
python test.py --exp_id=18 --exp_no=2 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0
python test.py --exp_id=18 --exp_no=3 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0
python test.py --exp_id=18 --exp_no=4 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0
python test.py --exp_id=18 --exp_no=5 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=15 --use_sigma_scale=1 --use_height_scale=0

# N_inputs 30

# Pure Cross Entropy with Type1
python test.py --exp_id=19 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=30
python test.py --exp_id=19 --exp_no=2 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=30
python test.py --exp_id=19 --exp_no=3 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=30
python test.py --exp_id=19 --exp_no=4 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=30
python test.py --exp_id=19 --exp_no=5 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=30

# Pure Cross Entropy with Type 2
python test.py --exp_id=20 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30
python test.py --exp_id=20 --exp_no=2 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30
python test.py --exp_id=20 --exp_no=3 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30
python test.py --exp_id=20 --exp_no=4 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30
python test.py --exp_id=20 --exp_no=5 --student_temp=1 --teacher_temp=2.5 --alpha=0 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30

# Type 1 Knowledge Distillation
python test.py --exp_id=21 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=30
python test.py --exp_id=21 --exp_no=2 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=30
python test.py --exp_id=21 --exp_no=3 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=30
python test.py --exp_id=21 --exp_no=4 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=30
python test.py --exp_id=21 --exp_no=5 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=1 --dataset=1 --n_inputs=30

# Type 2 Knowledge Distillation (A)
python test.py --exp_id=22 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=22 --exp_no=2 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=22 --exp_no=3 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=22 --exp_no=4 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1
python test.py --exp_id=22 --exp_no=5 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=0 --use_height_scale=1

# Type 2 Knowledge Distillation (B)
python test.py --exp_id=23 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=23 --exp_no=2 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=23 --exp_no=3 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=23 --exp_no=4 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1
python test.py --exp_id=23 --exp_no=5 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=1

# Type 2 Knowledge Distillation (C)
python test.py --exp_id=24 --exp_no=1 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0
python test.py --exp_id=24 --exp_no=2 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0
python test.py --exp_id=24 --exp_no=3 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0
python test.py --exp_id=24 --exp_no=4 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0
python test.py --exp_id=24 --exp_no=5 --student_temp=1 --teacher_temp=2.5 --alpha=0.75 --n_rules=7 --fuzzy_type=2 --dataset=1 --n_inputs=30 --use_sigma_scale=1 --use_height_scale=0
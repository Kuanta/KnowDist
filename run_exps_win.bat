Rem Pure cross entropy
python train.py --exp_id=1 --exp_no=1 --student_temp=1 --teacher_temp=5.0 --alpha=0 --n_rules=7
python train.py --exp_id=1 --exp_no=2 --student_temp=1 --teacher_temp=5.0 --alpha=0 --n_rules=7

python train.py --exp_id=2 --exp_no=1 --student_temp=1 --teacher_temp=5.0 --alpha=0.25 --n_rules=7
python train.py --exp_id=2 --exp_no=2 --student_temp=1 --teacher_temp=5.0 --alpha=0.25 --n_rules=7

TRAIN = True
TRAIN_TEACHER = False
TEST = False
COMPARE = False
DATASET = "Mnist"

EXP_NO = 4
ROOT = "./models/{}".format(EXP_NO)

TEACHER_MODEL_PATH = "./models/teacher_lite"
STUDENT_MODEL_PATH = ROOT + "/student_modified_kl"

# Constants
STUDENT_TEMP = 1
TEACHER_TEMP = 2.5
ALPHA = 0.25
N_RULES = 15
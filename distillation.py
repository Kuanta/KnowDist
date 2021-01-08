import torch
import torch.nn.functional as F

class DistillationLoss:
    def __init__(self, student_temp, teacher_temp, alpha):
        '''
        Loss function for knowledge distillation
        :param student_temp: Temperature value for the student logits
        :param teacher_temp: Temperature value for the teacher logits
        :param alpha: Weighting factor between CrossEntropy loss and KL divergence
        '''
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.alpha = alpha
        self.kl_loss = torch.nn.KLDivLoss(reduction="batchmean")

    def __call__(self, logits, labels):
        '''
        Forward function of the loss
        :param logits (tuple): An array containig the logits for student and teacher
        :param labels: One-hot encoded hard labels for the batch
        :return: Distillation loss
        '''
        student_logits = logits[0]
        teacher_logits = logits[1]
        hard_student_probs = F.softmax(student_logits, dim=1)
        hard_teacher_probs = F.softmax(teacher_logits, dim=1)

        soft_student_probs = F.log_softmax(student_logits/self.student_temp, dim=1)
        soft_teacher_probs = F.softmax(teacher_logits/self.teacher_temp, dim=1)

        # soft_student_probs = F.softmax(hard_student_probs / self.student_temp, dim=1)
        # soft_teacher_probs = F.softmax(hard_teacher_probs / self.teacher_temp, dim=1)

        # Cross-Entropy loss
        cross_loss = F.cross_entropy(hard_student_probs, labels)
        #kl_loss = -(soft_student_probs*soft_teacher_probs).sum(dim=1).mean()
        kl_loss = self.kl_loss(soft_student_probs, soft_teacher_probs)
        loss = (1-self.alpha)*cross_loss + self.alpha*self.student_temp*self.teacher_temp*kl_loss
        return loss

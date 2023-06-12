from django.core.exceptions import ValidationError
from django.db import models
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager
import numpy as np

class UserManager(BaseUserManager):
    def create_user(self, username, email, password, **extra_fields):
        extra_fields.setdefault('is_staff', False)
        extra_fields.setdefault('is_superuser', False)
        extra_fields.setdefault('face_encoding', None)
        return self._create_user(username, email, password, **extra_fields)

    def create_superuser(self, username, email, password, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        extra_fields.setdefault('face_encoding', None)
        return self._create_user(username, email, password, **extra_fields)
class Student(models.Model):
    level=models.CharField(max_length=5)
class PROFESSOR(models.Model):
    pass


    # fields for student information   
class Parent(models.Model):
  id = models.AutoField(primary_key=True)
  students = models.ManyToManyField(Student)

class User(AbstractBaseUser):
    username = models.CharField(max_length=50, unique=True)
    face_encoding = models.BinaryField(null=True, blank=True)
    student = models.ForeignKey(Student, on_delete=models.CASCADE, null=True, blank=True)
    parent = models.ForeignKey(Parent, on_delete=models.CASCADE, null=True, blank=True)
    professor = models.ForeignKey(PROFESSOR, on_delete=models.CASCADE, null=True, blank=True)




    

    objects = UserManager()

    USERNAME_FIELD = 'username'
    REQUIRED_FIELDS = ['email']
    STUDENT = 'student'
    PROFESSOR = 'professor'
    PARENT = 'parent'
    
    ROLE_CHOICES = [
        (STUDENT, 'Student'),
        (PROFESSOR, 'Professor'),
        (PARENT, 'Parent'),
    ]
    
    role = models.CharField(max_length=10, choices=ROLE_CHOICES, default=STUDENT)
    def __str__(self):
        return self.username

    def has_perm(self, perm, obj=None):
        return True

    def has_module_perms(self, app_label):
        return True
    def set_face_encoding(self, encoding):
        if encoding is not None:
            # Convert the numpy array to bytes
            encoding_bytes = encoding.tobytes()
            self.face_encoding = encoding_bytes

    def get_face_encoding(self):
        if self.face_encoding:
            # Convert the bytes back to a numpy array
            encoding = np.frombuffer(self.face_encoding, dtype=np.float64)
            return encoding
        else:
            return None
    @property
    def is_staff(self):
        return self.is_admin

    @property
    def is_admin(self):
        return self.is_superuser

    is_active = models.BooleanField(default=True)
    is_superuser = models.BooleanField(default=False)


class Comment(models.Model):
    source_user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='sent_comments', null=True, blank=True)
    destination_user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='received_comments', null=True, blank=True)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Comment from {self.source_user.username} to {self.destination_user.username}"
class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    # Add additional fields for profile information (e.g., bio, profile picture, etc.)

    def __str__(self):
        return self.user.username
from django.db.models.signals import post_save
from django.dispatch import receiver

@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        Profile.objects.create(user=instance)
class Performance(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    date = models.DateField()
    score = models.IntegerField()
    question = models.CharField(max_length=255, null=True)
    solution_provided = models.CharField(max_length=255, null=True)

    def __str__(self):
        return f"Performance for {self.student.user.username} - {self.date}, Question: {self.question}"
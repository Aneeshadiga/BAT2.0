# models.py
from django.db import models
import uuid

class BatSpecies(models.Model):
    name = models.CharField(max_length=255, unique=True)
    # Add other fields if needed

class UserData(models.Model):
    token = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    user_data = models.JSONField()

    def __str__(self):
        return str(self.token)
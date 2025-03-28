from django.db import models

class Battle(models.Model):
    first_pokemon = models.IntegerField()
    second_pokemon = models.IntegerField()
    winner = models.IntegerField()

    def __str__(self):
        return f"{self.first_pokemon} vs {self.second_pokemon} -> Winner: {self.winner}"

# test_montyhall_manual.py

# Assurez-vous d'importer la classe MontyHall
# from environments import MontyHall  # Utilisez cette ligne si la classe est définie dans environments.py
from environments import MontyHall  # Utilisez cette ligne si la classe est définie dans env.py

# Créer une instance de l'environnement Monty Hall
env = MontyHall()

# Réinitialiser l'environnement pour commencer un nouvel épisode
env.reset()

# Afficher les actions disponibles après la réinitialisation
print("Available actions after reset:", env.available_actions())  # Devrait afficher [0, 1, 2] pour les trois portes

# Choisir la première porte (action 0)
state, reward = env.step(0)
print("State and reward after choosing a door:", state, reward)

# Décider de changer de porte (action 4)
state, reward = env.step(4)
print("State and reward after switching:", state, reward)

import numpy as np

np.random.seed(2)

letters = ['A', 'B', 'C', 'D']
probabilities = [0.5, 0.25, 0.125, 0.125]

# Generate 1000 letters based on probabilities
generated_letters = np.random.choice(letters, 1000, p=probabilities)
generated_letters[:10]


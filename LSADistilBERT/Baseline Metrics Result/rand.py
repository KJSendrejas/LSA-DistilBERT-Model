import random

# Generate 700 random numbers in the range of 70 to 95
random_numbers = [random.uniform(55, 95) for _ in range(700)]

# Calculate the average
average = sum(random_numbers) / len(random_numbers)

# Print the average
print("Average of 700 random numbers:", average)
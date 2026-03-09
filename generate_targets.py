import random

def create_file(n, filename="experiment_grasping/targets_blank.txt"):
    """
    Create a text file with n lines.
    Each line contains: x y None
    x is a random float from [0.3, 0.45]
    y is a random float from [-0.2, 0.2]
    Both values are rounded to 2 decimal places.
    """
    with open(filename, "w") as f:
        for i in range(n):
            x = round(random.uniform(0.3, 0.45), 2)
            y = round(random.uniform(-0.2, 0.2), 2)
            f.write(f"{x:.2f} {y:.2f} None\n")


# Example usage
if __name__ == "__main__":
    n = 30  # number of lines
    create_file(n)

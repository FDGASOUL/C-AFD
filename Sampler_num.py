import math


def calculate_full_formula(v, p, delta, d):
    # Calculate the common logarithmic term log(p * sqrt(2 * pi))
    log_term = math.log(p * math.sqrt(2 * math.pi))

    # Calculate the numerator
    numerator = math.sqrt(-16 * v * log_term) - 8 * log_term

    # Calculate the denominator
    denominator = 1.69 * delta * (d - 1) * v ** (-0.071)

    # Final result: numerator / denominator
    result = numerator / denominator
    return result


# Example usage:
p = 0.000001
delta = 0.005
d1 = 48
d2 = 48
d = min(d1, d2)
v = (d1-1) * (d2-1)
result = calculate_full_formula(v, p, delta, d)
print("Result:", result)

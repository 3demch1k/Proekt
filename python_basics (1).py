"""Заготовки задач на базовый Python."""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    text = data.value
    vowels = "aeiou"
    count = 0
    for char in text.lower():
        if char in vowels:
            count += 1
    return count


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    return len(set(text)) == len(text)


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    binary_str = bin(number)
    return binary_str.count("1")


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    steps = 0
    while number >= 10:
        product = 1
        for digit in str(number):
            product *= int(digit)
        number = product
        steps += 1
    return steps


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    n = len(predicted)
    total_diff = 0
    for i in range(n):
        total_diff += (predicted[i] - expected[i]) ** 2
    return total_diff / n


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    factors = [] 
    d = 2
    temp = number

    while d * d <= temp:
        if temp % d == 0:
            count = 0
            while temp % d == 0:
                count += 1
                temp //= d
            factors.append((d, count))
        d += 1

    if temp > 1:
        factors.append((temp, 1))

    result = ""
    for prime, power in factors:
        if power == 1:
            result += f"({prime})"
        else:
            result += f"({prime}**{power})"

    return result


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    current_sum = 0
    k = 1

    while current_sum < cube_count:
        current_sum += k * k
        if current_sum == cube_count:
            return k
        k += 1

    return "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number_str = str(data.value)
    length = len(number_str)

    if length <= 2:
        return True

    if length % 2 != 0:
        mid = length // 2
        left_part = number_str[:mid]
        right_part = number_str[mid + 1:]
    else:
        mid = length // 2
        left_part = number_str[:mid - 1]
        right_part = number_str[mid + 1:]

    sum_left = sum(int(digit) for digit in left_part)
    sum_right = sum(int(digit) for digit in right_part)

    return sum_left == sum_right

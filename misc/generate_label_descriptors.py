import os

# List of symbols and their descriptions
symbols = {
    '!': "In mathematics, the exclamation mark denotes the factorial operation, where for a positive integer n, n! = n × (n - 1) × (n - 2) × ... × 1. Factorials are commonly used in combinatorics, algebra, and calculus.",
    '(': "The left parenthesis is used to group terms in mathematical expressions, particularly in operations like addition, subtraction, multiplication, and division, ensuring the correct order of operations.",
    ')': "The right parenthesis is used to close a group of terms in mathematical expressions, indicating that the enclosed operations should be evaluated first according to the order of operations.",
    '+': "The plus sign is used to represent addition in mathematics. It is a binary operator used to combine two numbers into their sum.",
    ',': "The comma is used in mathematics to separate elements in a list, set, or vector, and also to separate terms in some mathematical functions.",
    '-': "The minus sign is used to represent subtraction or a negative value in mathematics. It can also be used for denoting the opposite of a number.",
    '0': "The digit zero represents the additive identity in mathematics. It is the integer that, when added to any number, leaves the number unchanged.",
    '1': "The digit one represents the multiplicative identity in mathematics. It is the number that, when multiplied by any other number, leaves the number unchanged.",
    '2': "The digit two is the smallest and first prime number in mathematics, often used in binary systems, powers of two, and various mathematical sequences.",
    '3': "The digit three is the second prime number and often appears in geometric shapes (triangles) and other mathematical contexts.",
    '4': "The digit four is the smallest composite number and is often used in geometry, particularly in quadrilaterals.",
    '5': "The digit five is the third prime number, often appearing in various mathematical structures, such as pentagons and the Fibonacci sequence.",
    '6': "The digit six is the smallest perfect number in mathematics, as it is the sum of its divisors: 1, 2, and 3.",
    '7': "The digit seven is a prime number and often appears in various number theory and combinatorics problems.",
    '8': "The digit eight is a power of two, used in binary systems, and represents the first cubic number in mathematics.",
    '9': "The digit nine is the square of three and often appears in various mathematical patterns, such as magic squares.",
    '=': "The equals sign represents equality between two expressions in mathematics, indicating that both sides of an equation are equal.",
    'A': "The letter 'A' is often used to represent variables, constants, matrices, and sets in mathematics, particularly in algebra and set theory.",
    'C': "The letter 'C' commonly represents the set of complex numbers in mathematics, or it is used for constants in equations and formulas.",
    'Delta': "Delta (Δ) represents a change or difference in a quantity. It is commonly used in calculus, physics, and engineering.",
    'G': "The letter 'G' is often used to denote constants such as the gravitational constant, or to represent groups in abstract algebra.",
    'H': "The letter 'H' can represent the Hamiltonian operator in physics or a Hilbert space in functional analysis.",
    'M': "The letter 'M' is often used to represent matrices, sets, or various constants in mathematics.",
    'N': "The letter 'N' typically represents the set of natural numbers, which includes all positive integers starting from 1.",
    'R': "The letter 'R' represents the set of real numbers, including both rational and irrational numbers.",
    'S': "The letter 'S' can represent sets, sequences, or other mathematical structures, often in relation to sums or series.",
    'T': "The letter 'T' is commonly used to represent time, transformations, or a particular type of matrix in linear algebra.",
    'X': "The letter 'X' is often used as a variable to represent unknown quantities or in vector spaces and transformations.",
    '[': "The left square bracket is often used to denote intervals, or as part of the notation for lists, matrices, or vectors.",
    ']': "The right square bracket is used to close intervals or lists, matrices, or vectors, and to denote the end of array-like structures.",
    'alpha': "Alpha (α) is used to represent angles in trigonometry, coefficients in algebra, and is often used as a symbol for small quantities.",
    'b': "The letter 'b' is often used as a variable or constant in algebra, representing a base or a parameter in functions and equations.",
    'beta': "Beta (β) is often used to represent angles in trigonometry, as well as parameters in statistical distributions and functions.",
    'cos': "Cos (cosine) is a trigonometric function that represents the ratio of the adjacent side to the hypotenuse of a right triangle.",
    'd': "The letter 'd' often represents differentials in calculus, denoting small changes in variables.",
    'div': "The symbol 'div' denotes division in mathematics, representing the operation of dividing one number by another.",
    'e': "The letter 'e' represents Euler's number, approximately 2.718, which is the base of the natural logarithm and plays a crucial role in calculus.",
    'exists': "The 'exists' symbol (∃) is used in logic and mathematics to indicate that at least one element satisfies a given condition.",
    'f': "The letter 'f' is commonly used to represent functions in mathematics, where f(x) denotes the value of the function at x.",
    'forall': "The 'forall' symbol (∀) is used in logic and set theory to indicate that a statement holds for all elements of a set.",
    'forward_slash': "The forward slash (/) is used for division, indicating the division of one quantity by another.",
    'gamma': "Gamma (γ) is often used to represent certain constants, angles in geometry, or the gamma function in calculus.",
    'geq': "The 'geq' symbol (≥) is used to indicate that one quantity is greater than or equal to another.",
    'gt': "The 'gt' symbol (>) indicates that one quantity is greater than another.",
    'i': "The letter 'i' represents the imaginary unit in complex numbers, where i^2 = -1.",
    'in': "The 'in' symbol (∈) indicates that an element belongs to a set or group.",
    'infty': "The symbol for infinity (∞) represents an unbounded quantity in mathematics.",
    'int': "The 'int' symbol represents integration in calculus, where an integral is used to find areas under curves, among other applications.",
    'j': "The letter 'j' is commonly used in complex numbers, particularly in electrical engineering where j represents the imaginary unit.",
    'k': "The letter 'k' is used in mathematics to represent a constant, a variable, or an index in sequences.",
    'l': "The letter 'l' is used to represent lengths, linear functions, or other mathematical concepts.",
    'lambda': "Lambda (λ) is often used to represent eigenvalues in linear algebra or to denote functions in lambda calculus.",
    'ldots': "The 'ldots' symbol (…) represents a series or continuation of terms in mathematical notation.",
    'leq': "The 'leq' symbol (≤) indicates that one quantity is less than or equal to another.",
    'lim': "The 'lim' symbol represents limits in calculus, where the value of a function approaches a specific point.",
    'log': "The 'log' function represents the logarithm, which is the inverse of exponentiation, commonly used in algebra and calculus.",
    'lt': "The 'lt' symbol (<) indicates that one quantity is less than another.",
    'mu': "Mu (μ) is commonly used to represent a mean value or a coefficient in various branches of mathematics and physics.",
    'neq': "The 'neq' symbol (≠) indicates that two quantities are not equal.",
    'o': "The letter 'o' is used to represent a function or operation in various contexts, such as composition or small-o notation.",
    'p': "The letter 'p' often represents probability, a prime number, or a variable in various mathematical contexts.",
    'phi': "Phi (φ) represents the golden ratio, approximately 1.618, which appears in many areas of mathematics and art.",
    'pi': "Pi (π) is the mathematical constant representing the ratio of the circumference of a circle to its diameter, approximately 3.14159.",
    'pm': "The 'pm' symbol (±) indicates plus or minus, often used to show the range of possible values for an uncertain quantity.",
    'q': "The letter 'q' is commonly used to represent rational numbers, variables, or specific constants in mathematics.",
    'rightarrow': "The right arrow symbol (→) represents a function or mapping from one set to another.",
    'sigma': "Sigma (σ) is used to represent summation in mathematics, often used in statistics to denote standard deviation.",
    'sin': "Sin (sine) is a trigonometric function that represents the ratio of the opposite side to the hypotenuse in a right triangle.",
    'sqrt': "Sqrt (square root) is the operation that finds a number that, when squared, gives the original number.",
    'sum': "The sum symbol (∑) represents the addition of a sequence of numbers, often used in series and summations.",
    'tan': "Tan (tangent) is a trigonometric function that represents the ratio of the opposite side to the adjacent side of a right triangle.",
    'theta': "Theta (θ) is often used to represent angles in trigonometry and geometry.",
    'times': "The times symbol (×) represents multiplication in mathematics.",
    'u': "The letter 'u' is often used as a variable in mathematics, representing a function, sequence, or other quantity.",
    'v': "The letter 'v' is often used to represent vectors, variables, or other quantities in mathematics.",
    'vert_bar': "The vertical bar (|) represents absolute value, or in set notation, the separator for conditions in set-builder notation.",
    'w': "The letter 'w' is used as a variable in various mathematical contexts, such as representing weights in optimization problems.",
    'y': "The letter 'y' is commonly used as a variable in algebra and calculus, often representing the dependent variable in functions.",
    'z': "The letter 'z' is often used to represent complex numbers or variables in mathematics.",
    '{': "The left curly brace is used in set notation to enclose elements of a set.",
    '}': "The right curly brace is used to close a set notation and indicate the end of the set elements."
}

# Specify directory to save text files
directory = "definitions"

# Create the directory if it doesn't exist
os.makedirs(directory, exist_ok=True)

# Write descriptions to text files
for symbol, description in symbols.items():
    # Create a file for each symbol
    with open(os.path.join(directory, f"{symbol}.txt"), 'w') as file:
        file.write(f"Character: {symbol}\n")
        file.write(f"Description: {description}\n")

print("Text files have been generated in the specified directory.")

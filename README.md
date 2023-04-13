# esbmf
Error-Shaping Based Boolean Matrix Factorization.

## Usage

```python
import numpy as np
from esbmf.bmf import BMF


def generate_original(size, p):
    original = np.random.choice([False, True], size=size, p=[1 - p, p])
    while np.all(original == 0, axis=0).any():
        original = np.random.choice([False, True], size=size, p=[1 - p, p])
    return original


# generate a boolean matrix with size 2^4 x 4, P(1) = 0.5
original = generate_original((2**4, 4), 0.5)
# BMF with factorization degree of 3
bmf = BMF(original, 3)
bmf.run_esbmf()


print(f'the original:\n{original}')
print(f'the approx:\n{bmf.approx}')


```

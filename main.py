from scipy.stats.qmc import LatinHypercube
import numpy as np

d = 1 # 차원 수
n = 100 # 샘플 수

lhd = LatinHypercube(d)
samples = lhd.random(n)

# 예시 함수 (실제 문제에 맞게 변경)
def target_function(x):
    return x**2

y_samples = np.array([target_function(x) for x in samples])

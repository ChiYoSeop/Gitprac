import numpy as np
from pandas import Series
import matplotlib.pyplot as plt

from lmfit import Model, Parameter, report_fit

def decay(t):
    return (10^(-1))*(np.exp(t/0.026)-1)

t = np.linspace(0, 5, num=1000)
np.random.seed(2021)
data = decay(t) + np.random.randn(t.size)

model = Model(decay, independent_vars=['t'])
result = model.fit(data, t=t)

print(result.values)

result.plot()
plt.show()
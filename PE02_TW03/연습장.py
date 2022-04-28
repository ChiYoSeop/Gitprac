import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
from lmfit.models import ExponentialModel
from lmfit import parameter
from sklearn.metrics import r2_score

tree = ET.parse('HY202103_D08_(0,2)_LION1_DCM_LMZC.xml')
root = tree.getroot()

def snf(a):
    splt = a.text.split(',')
    flst = list(map(float,splt))
    return flst


for data in root.iter('Voltage'):
    # x = np.array(snf(data))
    x = np.array(snf(data))
v1 = x[:9]
v2 = x[9:]

for data in root.iter('Current'):
    crt = list(map(abs,snf(data)))
    # b = np.array(snf(data)[6:])
    y = np.array(crt)
i1 = y[:9]
i2 = y[9:]

dp1 = np.polyfit(v1, i1, 3)
f1 = np.poly1d(dp1)

regressor = ExponentialModel()
initial_guess = dict(amplitude=1, decay=-1)



results = regressor.fit(i1, x=v1, **initial_guess)
y_fit = results.best_fit
results1 = regressor.fit(i2, x=v2, **initial_guess)
y_fit1 = results1.best_fit

# plt.subplot(1,2,1)

plt.plot(v1, i1, "o", label="Data")
plt.plot(v1, f1(v1), "r--", label="Fit")
# plt.legend()
# plt.yscale("log")


#
# plt.subplot(1,2,2)

plt.plot(v2, i2, "o", label="Data")
plt.plot(v2, y_fit1, "r--", label="Fit")
plt.legend()
plt.yscale("log")


plt.show()

results1.params.pretty_print()
print(f"R square: {r2_score(i2, y_fit1)}")
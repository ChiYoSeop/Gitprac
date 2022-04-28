import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
from lmfit.models import ExponentialModel
from lmfit import Model
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

# dp1 = np.polyfit(v1, i1, 3)
# f1 = np.poly1d(dp1)

def IV(v1, ir, vt):
    return(ir*(-1*np.exp(v1/vt)+1))

def Skley(v2, ir, vt):
    return(ir*(np.exp(v2/vt)-1))

gmodel = Model(Skley)
result = gmodel.fit(i2, v2 = v2, ir = 10e-10, vt = 0.026)

gmodel = Model(IV)
result1 = gmodel.fit(i1, v1 = v1, ir = 10e-10, vt = 0.026)

print(result.fit_report())


# plt.plot(v1, i1, "o", label="Data")
# plt.plot(v1, f1(v1), "r--", label="Fit")
plt.plot(v1, i1, "o", label="Data")
# plt.plot(v1, result1.init_fit, "r--", label="init1_Fit")
plt.plot(v1, result1.best_fit, "y--", label="Best1_Fit")


plt.plot(v2, i2, "o", label="Data")
# plt.plot(v2, result.init_fit, "r--", label="init_Fit")
plt.plot(v2, result.best_fit, "y--", label="Best_Fit")
plt.legend()
plt.yscale("log")


plt.show()

result.params.pretty_print()
print(f"R square: {r2_score(i2, result.best_fit)}")
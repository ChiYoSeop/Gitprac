import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
from lmfit import Model
from sklearn.metrics import r2_score

tree = ET.parse('HY202103_D08_(0,2)_LION1_DCM_LMZC.xml')
root = tree.getroot()

def spfl(a):
    splt = a.text.split(',')
    flst = list(map(float,splt))
    return flst


for data in root.iter('Voltage'):
    x = np.array(spfl(data))
v1 = x[:9]
v2 = x[9:]

for data in root.iter('Current'):
    crt = list(map(abs,spfl(data)))
    y = np.array(crt)
i1 = y[:9]
i2 = y[9:]

#좌측 polyfit을 활용한 fiiting
dp1 = np.polyfit(v1, i1, 3)
f1 = np.poly1d(dp1)

plt.plot(v1, i1, "o", label="Left Data")
plt.plot(v1, f1(v1), "r--", label="Fitting by polyfit")

#우측 lmfit 함수를 직접 정의하여 fitting
def Skly(v2, ir, vt):
    return(ir*(np.exp(v2/vt)-1))

gmodel = Model(Skly)
result = gmodel.fit(i2, v2 = v2, ir = 10e-10, vt = 0.026)

plt.plot(v2, i2, "o", label="Right Data")
plt.plot(v2, result.best_fit, "y--", label="Best_Fit by lmfit")

plt.legend()
plt.yscale("log")
plt.show()

#좌측 R square
print(f"Left R square: {r2_score(i1, f1(v1))}")

#우측 Best fit 함수의 파라미터 결과 및 R square
print(result.fit_report())
print(f"Right R square: {r2_score(i2, result.best_fit)}")
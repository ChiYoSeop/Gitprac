import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import lmfit

tree = ET.parse('HY202103_D08_(0,2)_LION1_DCM_LMZC.xml')
root = tree.getroot()

def snf(a):
    splt = a.text.split(',')
    flst = list(map(float,splt))
    return flst


for data in root.iter('Voltage'):
    x = np.array(snf(data))

for data in root.iter('Current'):
    crt = list(map(abs,snf(data)))
    y = np.array(crt)

p = lmfit.Parameters()
p.add_many(('a1', 10e-9), ('a2', 0.026))

def residual(p):
    v = p.valuesdict()
    return v['a1']*(np.exp(x/v['a2'])-1)

mi = lmfit.minimize(residual, p)

lmfit.printfuncs.report_fit(mi.params)


plt.plot(x, residual(mi.params)+y, 'r--', label='best fit')
plt.plot(x,y,'o',label = 'Raw data')
plt.legend(loc='best')
plt.yscale('log')
plt.show()
import xml.etree.ElementTree as ET

import lmfit
import matplotlib.pyplot as plt
import numpy as np
from lmfit.models import ExponentialModel
from sklearn.metrics import r2_score
from lmfit import Model

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


mi =lmfit.minimize






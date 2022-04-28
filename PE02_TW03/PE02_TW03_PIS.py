import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np

tree = ET.parse('HY202103_D08_(0,2)_LION1_DCM_LMZC.xml')
root = tree.getroot()

def snf(a):
    splt = a.text.split(',')
    flst = list(map(float,splt))
    return flst

wvlen = []
itst = []

for data in root.iter('L'):
    L = snf(data)
    wvlen.append(L)

for data in root.iter('IL'):
    IL = snf(data)
    itst.append(IL)

lgds = []

for data in root.iter("WavelengthSweep"):
    lgds.append(data.get("DCBias"))

# Fit "REF' Graph to Polynomial
dp1 = np.polyfit(wvlen[6], itst[6], 4)
f1 = np.poly1d(dp1)

plt.title("Transmission spectra-as measured")
plt.xlabel("Wavelength [nm]")
plt.ylabel("Measured transmission [dB]")
# plt.plot(wvlen[6], itst[6],label="Raw")
# plt.plot(wvlen[6], f1(wvlen[6]), 'r--', label='Fit')

# Subtract the fitted "REF" value corresponding to each wavelength from the measured data.
for n in range(len(wvlen)-1):
    plt.plot(wvlen[n], itst[n]-f1(wvlen[n]), label=f"DCBias ={lgds[n]}V")


plt.rc("legend", fontsize = 7)
plt.legend(loc = 'best' , ncol = 3)
plt.show()


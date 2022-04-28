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

    for n in range(len(wvlen)):
        if n == 6:
            plt.plot(wvlen[n], itst[n], label="REF")

        else:
            # plt.plot(wvlen[n], itst[n], label=f"DCBias ={lgds[n]}V")
            continue

dp1 = np.polyfit(wvlen[6], itst[6], 3)
f1 = np.poly1d(dp1)
plt.plot(wvlen[6], f1(wvlen[6]),'b--', label = 'fit')
plt.plot(wvlen[5],f1(wvlen[5]))

plt.title("Transmission spectra-as measured")
plt.xlabel("Wavelength [nm]")
plt.ylabel("Measured transmission [dB]")
plt.rc("legend", fontsize = 7)
plt.legend(loc = 'best' , ncol = 3)

plt.show()
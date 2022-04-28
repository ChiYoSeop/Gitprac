import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
tree = ET.parse('HY202103_D08_(0,2)_LION1_DCM_LMZC.xml')
root = tree.getroot()

def snf(a):
    splt = a.text.split(',')
    flst = list(map(float,splt))
    return flst

# # I-V Graph
#
# for data in root.iter('Voltage'):
#     vlt = snf(data)
#
# for data in root.iter('Current'):
#     crt = list(map(abs,snf(data)))
#     crt2 = snf(data)


# plt.figure(figsize = (10,5))
# # plt.subplot(1,2,1)

# plt.plot(vlt,crt, 'bo--', label = 'I-V curve')
# plt.title("I-V analysis")
# plt.xlabel("Voltage [V]")
# plt.ylabel("Current [A]")
# plt.legend(loc = ('best'))
# plt.yscale('log')



# I-IL Graph
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


# plt.subplot(1, 2, 2)



for n in range(len(wvlen)):
    if n == 6:
        plt.plot(wvlen[n], itst[n], label="REF")

    else:
        # plt.plot(wvlen[n], itst[n], label=f"DCBias ={lgds[n]}V")
        continue
# data fitting
dp1 = np.polyfit(wvlen[6], itst[6], 3)
f1 = np.poly1d(dp1)
plt.plot(wvlen[6], f1(wvlen[6]),'b--', label = 'fit')
plt.plot(wvlen[5],f1(wvlen[5]))



# def polyfit(x, y, degree):
#     results = {}
#     coeffs = np.polyfit(x, y, degree)
# # 다항 상관 계수
#     results['polynomial'] = coeffs.tolist()
# # r-제곱
#     p = np.poly1d(coeffs)
# # 적합(fit) 값들과 평균
#     yhat = p(x)     # or [p(z) for z in x]
#     ybar = np.sum(y)/len(y)  # or sum(y)/len(y)
#     ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
#     sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
#     results['determination'] = ssreg / sstot
#     return results

# print(polyfit(wvlen[6],f1(wvlen[6]),2))



plt.title("Transmission spectra-as measured")
plt.xlabel("Wavelength [nm]")
plt.ylabel("Measured transmission [dB]")
plt.rc("legend", fontsize = 7)
plt.legend(loc = 'best' , ncol = 3)



# plt.savefig("PE2_TW02_Pic.png",dpi = 300, bbox_inches = 'tight')

plt.show()

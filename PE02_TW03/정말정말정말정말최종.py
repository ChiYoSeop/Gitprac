import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
from numpy import exp
from lmfit import Model
from sklearn.metrics import r2_score
import time

tree = ET.parse('HY202103_D08_(0,2)_LION1_DCM_LMZC.xml')
root = tree.getroot()

def spfl(a):    # spfl 함수 정의
    sp = a.text.split(',')  # ,를 기준으로 나누고 값 가져오기
    fl = list(map(float, sp))   # 가져온 값을 실수로 바꾸고 리스트에 넣기
    return fl   # fl 반환

def eq(x, a, b, c, d, e):
    return a * (x**4) + b * (x**3) + c * (x**2) + d * x + e

def IV(x, Is, q, n, k):
    return Is * (exp((q * x) / (n * k)) - 1)

# 3차
# I-V Graph
for data in root.iter('Voltage'):
    vlt = spfl(data)  # 'Voltage'안에 있는 값을 spfl함수를 사용해 v에 저장
    v = np.array(vlt)
for data in root.iter('Current'):
    crt = list(map(abs, spfl(data)))  #'Current'안에 있는 값을 spfl함수를 사용하고, 절댓값을 사용해 리스트 안에 넣어
    i = np.array(crt)

plt.figure(figsize = (12, 8))
plt.subplot(2, 3, 4)
plt.plot(v, i, 'o', label = 'I-V curve')
plt.title("IV analysis")
plt.xlabel("Voltage [V]")
plt.ylabel("Current [A]")
plt.yscale('logit')
plt.legend(loc = 'best')

plt.subplot(2,3,5)
plt.plot(v, i, 'o', label = 'I-V curve')

v1 = v[:10]
v2 = v[9:]

i1 = i[:10]
i2 = i[9:]

start_time1 = time.time()
lmodel = Model(eq)
params1 = lmodel.make_params(a=1, b=1, c=1, d=1, e=1)
result1 = lmodel.fit(i1, params1, x = v1)
plt.plot(v1, result1.best_fit, '--', label = 'Fit-l')
end_time1 = time.time()
print(f'left fitting time : {end_time1 - start_time1}')

start_time2 = time.time()
rmodel = Model(IV)
params2 = rmodel.make_params(Is=1, q=1, n=1, k=1)
result2 = rmodel.fit(i2, params2, x = v2)
plt.plot(v2, result2.best_fit, '--', label = 'Fit-r')
end_time2 = time.time()
print(f'right fitting time : {end_time2 - start_time2}')

plt.title("IV analysis")
plt.xlabel("Voltage [V]")
plt.ylabel("Current [A]")
plt.yscale('logit')
plt.legend(loc='best')
print(f'Left R Squared : {r2_score(i1, result1.best_fit)}')
print(f'Right R Squared: {r2_score(i2, result2.best_fit)}')

# Raw data
wvl = []
itst = []
for data in root.iter('L'):
    L = spfl(data)
    wvl.append(L)
for data in root.iter('IL'):
    IL = spfl(data)
    itst.append(IL)

lgds = []
for data in root.iter("WavelengthSweep"):
    lgds.append(data.get("DCBias"))

plt.subplot(2, 3, 1)
for n in range(len(wvl)):
    plt.title("Transmission spectra-as measured")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Measured transmission [dB]")
    plt.rc("legend", fontsize=7)
    if n == 6:
        plt.plot(wvl[6], itst[6], label = 'DCBias = REF')
        plt.rc("legend", fontsize=5)
    else:
        plt.plot(wvl[n], itst[n], label = f'DCBias = {lgds[n]}V')
    plt.legend(loc = 'best', ncol = 3)

# Fitting
plt.subplot(2, 3, 2)
for n in range(len(wvl)):
    if n == 6:
        plt.plot(wvl[n], itst[n], label="REF")
    else:
        continue
for b in range(2,7):
    dp1 = np.polyfit(wvl[6], itst[6], b)
    f1 = np.poly1d(dp1)
    plt.plot(wvl[6], f1(wvl[6]), 'r--', label = f'{b} th R^2:{round(r2_score(itst[6], f1(wvl[6])),4)}')
    plt.rc("legend", fontsize=8)
print(f'I-IL R Squared : {r2_score((itst[6]),f1(wvl[6]))}') # 피팅 제곱의 값 즉, 1에 가까울 수록 정확하다.
plt.xlabel('Wavelength[nm]')
plt.ylabel('Transmissions[dB]')
plt.title('Transmission spectra - fitted')
plt.legend(loc='best')


plt.subplot(2, 3, 3)
for k in range(len(wvl)-1):
    plt.title("Fitting Function")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Measured transmission [dB]")
    plt.plot(wvl[k], itst[k] - f1(wvl[k]), label = f'DCBias = {lgds[k]}V')
    plt.legend(loc = 'best', ncol = 3)
    plt.rc("legend", fontsize=5)
    # 각 DC Bias에서 빛의 세기의 최대값
    print(f'Max intensity[dB] at DCBias = {lgds[k]} :{max(itst[k] - f1(wvl[k]))}')
    # 빛의 세기가 최대일 때 파장
    print(f'at {wvl[k][(np.argmax(itst[k] - f1(wvl[k])))]}nm')
    # 각 DC Bias에서 빛의 세기의 최소값
    print(f'Min intensity[dB] at DCBias = {lgds[k]} :{min(itst[k] - f1(wvl[k]))}')
    # 빛의 세기가 최소일 때 파장
    print(f'at {wvl[k][(np.argmin(itst[k] - f1(wvl[k])))]}nm')
plt.tight_layout()
plt.savefig("graph.png",dpi = 300, bbox_inches = 'tight')


# print(f'max :{max(itst[k]-f1(wvl[k]))}')


fiti = result1.best_fit
# h = [-2.0,-1.5,-1,-0.5,0.0]
# for t in range(len(fiti)):
#     print(f'{}: {fiti[t]}')

# 좌측 best_fit일 때의 parameters 값
print(f'Left best Parameters: {result1.best_values}')
# 좌측 best_fit일 때의 Current 값들
print(f'Left best Currents[A]: {result1.best_fit}')
# 우측 best_fit일 때의 parameters 값
print(f'Right best Parameters: {result2.best_values}')
# 우측 best_fit일 때의 Current 값들
print(f'Right best Currents[A]: {result2.best_fit}')
plt.show()
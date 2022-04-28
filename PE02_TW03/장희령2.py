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

def IV(x, Is, q, n, k):
    return Is * (exp((q*x) / (k*n)) - 1)

def eq(x, a, b, c, d, e):
    return a * (x**4) + b * (x**3) + c * (x**2) + d * x + e

for data in root.iter('Voltage'):
    vlt = spfl(data)  # 'Voltage'안에 있는 값을 spfl함수를 사용해 v에 저장
    v = np.array(vlt)
for data in root.iter('Current'):
    crt = list(map(abs, spfl(data)))  #'Current'안에 있는 값을 spfl함수를 사용하고, 절댓값을 사용해 리스트 안에 넣어
    i = np.array(crt)

''' TIME, Y축값, LMFIT 변수'''

v1 = v[:10]
v2 = v[9:]

i1 = i[:10]
i2 = i[9:]
# 좌측 계산 시간 측정
start_time1 = time.time()
lmodel = Model(eq)
params1 = lmodel.make_params(a=1, b=1, c=1, d=1, e=1)
result1 = lmodel.fit(i1, params1, x = v1)
plt.plot(v1, result1.best_fit, '--', label = 'Fit-l')
end_time1 = time.time()
print(f'left fitting time : {end_time1 - start_time1}')

#우측 계산 시간 측정
start_time2 = time.time()
rmodel = Model(IV)
params2 = rmodel.make_params(Is = 1, q = 1, n=1, k=1)
result2 = rmodel.fit(i2, params2, x = v2)
plt.plot(v2, result2.best_fit, '--', label = 'Fit-r')
end_time2 = time.time()
print(f'right fitting time : {end_time2 - start_time2}')

print(result2.fit_report())
plt.plot(v, i, 'o', label = 'I-V curve')
plt.title("IV analysis")
plt.xlabel("Voltage [V]")
plt.ylabel("Current [A]")
plt.yscale('log')
plt.legend(loc='best')
print(f'left : {r2_score(i1, result1.best_fit)}')
print(f'right : {r2_score(i2, result2.best_fit)}')

plt.show()
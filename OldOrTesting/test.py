import numpy as np

def sig(x):
    return (np.exp(x)/(np.exp(x)+1))

def gomp(x):
    a = np.exp(1)
    b = 4
    c = 0.736
    return (a*np.exp(1)**(-b*(np.exp(1))**(-(c*x))))

def testSig(w,c):
	return np.exp(1) * (np.log(1.5*w)/(sig(c)))

def testGomp(w,c):
	return np.exp(1) * (np.log(1.5*w)/(gomp(c)))

def testLog10(w,c):
	return np.exp(1) * (np.log(1.5*w)/(np.exp(np.log10(c*4))))

for i in range(10):
	j = i + 1
	for k in range(8):
		l = k + 1
		print(j,l,testSig(j,l),testGomp(j,l),testLog10(j,l))
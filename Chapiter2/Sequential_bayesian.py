import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

a=1
b=1#Hyper Parameter

X=np.linspace(0,1,100)
Y=beta.pdf(X,a,b)

plt.plot(X,Y)
plt.legend()
plt.title("prior disribution")
plt.xlabel("mu")
plt.show()

while True:
    x=int(input("0か1を入力してください: "))    
    
    if x==0:
        b+=1
    elif x==1:
        a+=1
    else:
        break
            
    Y=beta.pdf(X, a, b)
    plt.plot(X,Y)
    plt.legend()
    plt.title("posterior disribution")
    plt.xlabel("mu")
    plt.show()

    
    

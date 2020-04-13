import matplotlib.pyplot as plt
import numpy as np

class Polynomial_regression(object):
    def __init__(self,alpha=0.1,beta=0.4,M=10):
        self.alpha=alpha
        self.beta=beta
        self.M=M
    def phi(self,x):
        return np.array([x**i for i in range(self.M+1)])# (M+1)xlen(x) matrix
    def matrix_S(self,x_train):
        I=np.identity(self.M+1)
        S=self.alpha*I+self.beta*np.dot(self.phi(x_train),self.phi(x_train).T)
        S=np.linalg.inv(S) 
        return S #(M+1)x(M+1) matrix      
    def prediction(self,x,x_train,t_train):
        mean=self.beta*self.phi(x).T.dot(self.matrix_S(x_train)).dot(np.dot(self.phi(x_train),t_train))
        var=np.diag(1/self.beta+self.phi(x).T.dot(self.matrix_S(x_train)).dot(self.phi(x)))
        std=np.sqrt(var)
        return mean,std#N vector

def samples(x):
    return np.sin(2*np.pi*x)+np.random.normal(0,1,len(x))

def main():
    alpha=0.1#Hyperparameter
    beta=0.4
    M=10#Order
    N=20#Number of observed data
    model=Polynomial_regression(alpha,beta,M)       
    x_train=np.random.rand(N)
    t_train=samples(x_train)
    x=np.linspace(0,1,100)
    y,y_std=model.prediction(x,x_train,t_train)
    
    
    plt.scatter(x_train,t_train,label="observed value")
    plt.plot(x,np.sin(2*np.pi*x),color="red",label="sin($2\pi x$)")
    plt.plot(x,y,color="green",label="predictive distribution")
    plt.fill_between(x, y-y_std, y+y_std, color='pink', alpha=0.5, label="predict_std")
    plt.legend()
    plt.title("Polynoomial prediction")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.xlim(0,1.1)
    plt.ylim(-3, 3)
    plt.show()
    
    
if __name__ == '__main__':
    main()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_multilabel_classification
import scipy.linalg

"""
Performs update for the latent variable z. See equations 24 and 25
"""
def update_z(d1,d2,m,W1_mu,W1_sigma,W2_mu,W2_sigma,
             EPhi1,EPhi2,X,Y,Mu1,Mu2,n):
    z_precision = np.zeros((m,m))
    z_sigma = np.zeros((m,m))
    z_sigma = np.zeros((m,n))
    
    z_precision = np.eye(m) + np.dot(np.dot(W1_mu.T, EPhi1),W1_mu) + \
                    np.dot(np.dot(W2_mu.T, EPhi2),W2_mu)
    for i in range(d1):
        z_precision = z_precision + EPhi1[i,i] * W1_sigma[i]
    for i in range(d2):
        z_precision = z_precision + EPhi2[i,i]*W2_sigma[i]
    
    z_sigma = np.linalg.inv(z_precision)

    z_mu = np.dot(np.dot(W1_mu.T,EPhi1),(X-Mu1)) + \
        np.dot(np.dot(W2_mu.T,EPhi2),(Y-Mu2))
    z_mu = np.linalg.solve(z_precision,z_mu)
    #z_mu = np.dot(z_sigma,z_mu)
    
    return z_sigma, z_mu,z_precision

"""
Performs update for W which is the latent variable for the number of
optimal components for the model
See eqn 28 and eqn 29
"""
def update_W(z_mu,n,W_mu,Ealpha,EPhi,Ezz,X,d,Mu,m):
    W_precision = np.zeros((d,m,m))
    W_sigma = np.zeros((d,m,m))
    #W_mu = np.zeros(W_mu.shape)
    
    for i in range(d):
        W_precision[i,:,:] = np.diag(Ealpha) + EPhi[i,i]*Ezz
        W_sigma[i,:,:] = np.linalg.inv(W_precision[i,:,:])
    W_mu_new = np.dot(z_mu,np.dot((X-Mu).T,EPhi)).T
    #W_mu_test=W_mu
    #print "This is for dimension ", d
    for i in range(d):
        #W_sum = np.reshape(W_mu[-i,:]* EPhi[i,-i],(2,1))
        W_mu_mask = np.ma.array(W_mu,mask=False)
        W_mu_mask.mask[i,:] = True
        #print W_mu_mask
        EPhi_mask = np.ma.array(EPhi,mask = False)
        EPhi_mask.mask[:,i] = True
        #print np.ma.dot(W_mu_mask.T,EPhi_mask[i,:]).shape
        W_sum = np.reshape(
            np.array(np.ma.dot(W_mu_mask.T,EPhi_mask[i,:])),(2,1))
        #print W_sum
        W_mu_new[i,:] = np.reshape(
            np.reshape(W_mu_new[i,:],(2,1)) - np.dot(Ezz,W_sum),(2))
        #W_mu[i,:] = np.dot(W_sigma[i],W_mu_new[i,:])
        W_mu[i,:] = np.linalg.solve(W_precision[i],W_mu_new[i,:])
    return W_sigma,W_mu,W_precision

"""
Performs update on mu. See equations 30 and 31
"""
def update_mu(beta,d,n,EPhi,X,W_mu,z_mu):
    
    mu_precision = beta*np.eye(d) + n*EPhi
    mu_sigma = np.linalg.inv(mu_precision)
    
    mu_mu = np.linalg.solve(mu_precision,
                            np.dot(EPhi,np.sum(X-np.dot(W_mu,z_mu),axis = 1)))
    #mu_mu = np.dot(mu_sigma,
                #np.dot(EPhi,np.sum(X-np.dot(W_mu,z_mu),axis = 1)))
    
    return mu_sigma,mu_mu

"""
Performs update on a and b. See eqn 32 and 33
"""
def update_a_b(a,b,W_mu,W_sigma,d):
    a_new = a+d/2
    b_new = b + np.sum(W_mu*W_mu,axis=0)/2
    for i in range(d):
        b_new = b_new + np.diag(W_sigma[i])/2
    
    return a_new,b_new

"""
Performs update on Phi. See eqns 26 and 27
"""
def update_phi(nu_prior,n,mu_sigma,mu_mu,d,W_precision,W_sigma,
               W_mu,Ezz,X,XX,z_mu,Mu,K):
    nu = nu_prior + n
    
    Emumu = mu_sigma + np.outer(mu_mu,mu_mu)
    
    temp = np.zeros((d,d))
    for i in range(d):
        #temp[i,i] = np.trace(np.dot(W_sigma[i],Ezz))
        temp[i,i] = np.trace(np.linalg.solve(W_precision[i],Ezz))
         
    deltaK = XX + np.dot(np.dot(W_mu,Ezz),W_mu.T) + temp + n*Emumu
    XZW = np.dot(np.dot(X,z_mu.T),W_mu.T)
    XMu = np.dot(X,Mu.T)
    WZMu = np.dot(np.dot(W_mu,z_mu),Mu.T)
    deltaK = deltaK - XZW - np.transpose(XZW)
    deltaK = deltaK - XMu - XMu.T
    deltaK = deltaK + WZMu + WZMu.T
    K_new = K*np.eye(d) + deltaK
    invK_new = np.linalg.pinv(K_new)
    
    return nu, invK_new,K_new,deltaK

def digamma_d(nu, d):
    return np.sum(scipy.special.digamma((nu+1-np.array(range(d)))/2))

def logdet(x):
    return np.log(np.linalg.det(x))

def compElogPhi(nu,d,K):
    return d * np.log(2) - logdet(K) + digamma_d(nu,d)

def lgamma_d(nu,d):
    return np.log(np.pi)*d*(d-1)/4 + \
    np.sum(scipy.special.gammaln((nu+1-np.array(range(d))/2)))

def log_partition(nu,d,K):
    return 0.5*nu*d*np.log(2) - 0.5*nu*logdet(K) + lgamma_d(nu,d)

"""
We initialise CCA using MLE results
"""
def initialise_CCA(X,Y,m):
    # we will create the sample covariance matrix
    z = np.concatenate((X,Y)) #combine the tranposed data. 
    #Each row represents a variable and column represents sample
    # X & Y must have the same number of samples
    C = np.cov(z) 
        
    sx = X.shape[0] #find the dimensions of X and Y
    sy = Y.shape[0]
    n = X.shape[1]
        
    #we partition the covariance matrix into the respective elements
    Cxx = C[0:sx,0:sx]
    Cxy = C[0:sx,sx:sx+sy]
    Cyx = Cxy.T
    Cyy = C[sx:,sx:]
    
    u1,s1,v1 = np.linalg.svd(Cxx)
    s1_half_inv = np.dot(np.dot(u1,np.diag(1/np.sqrt(s1))),u1.T)
    u2,s2,v2 = np.linalg.svd(Cyy)
    s2_half_inv = np.dot(np.dot(u2,np.diag(1/np.sqrt(s2))),u2.T)
    sm = np.dot(np.dot(s1_half_inv,Cxy),s2_half_inv)
    u12,s12,v12 = np.linalg.svd(sm)

    W1_mu = np.dot(np.dot(np.dot(Cxx,s1_half_inv),u12[:,:m]),
                   np.diag(np.sqrt(s12[:m])))
    W2_mu = np.dot(np.dot(np.dot(Cyy,s2_half_inv),v12.T[:,:m]),
                   np.diag(np.sqrt(s12[:m])))
    
    EPhi1 = np.linalg.pinv(Cxx-np.dot(W1_mu,W1_mu.T))
    EPhi2 = np.linalg.pinv(Cyy-np.dot(W2_mu,W2_mu.T))
    mu1_mu = np.mean(X,axis = 1)
    mu2_mu = np.mean(Y,axis = 1)

    Mu1 = np.array([mu1_mu]*n).T
    Mu2 = np.array([mu2_mu]*n).T

    return W1_mu, W2_mu, EPhi1, EPhi2, Mu1, Mu2

def probabilistic_CCA(X,Y,max_iter = 100,a=0.001,b=0.001,beta=0.001,K=0.001,em_converge = 0.0001):
    
    #define some constants
    d1 = X.shape[1]
    d2 = Y.shape[1]
    d = min(d1,d2)
    n = X.shape[0]
    m = d
    elbo = -1e300
    #initialise some variables
    #z prior eqn 1.5
    z_sigma = np.eye(m)
    z_mu = np.zeros((m,n))
    
    #phi prior eqn 14
    nu1_prior = d1
    nu2_prior = d2
    nu1 = d1
    nu2 = d2
    K1 = K*np.eye(d1)
    K2 = K*np.eye(d2)

    #W prior eqn 10 and 11
    W1_sigma = np.zeros((d1,m,m))
    for i in range(d1):
        W1_sigma[i,:,:] = b/a*np.eye(m)
    W2_sigma = np.zeros((d2,m,m))
    for i in range(d2):
        W2_sigma[i,:,:] = b/a*np.eye(m)
    W1_mu = np.zeros((d1,m))
    W2_mu = np.zeros((d2,m))
    #mu prior eqn 13
    mu1_sigma = 1/beta*np.eye(d1)
    mu1_mu = np.mean(X,axis = 1)

    mu2_sigma = 1/beta*np.eye(d2)
    mu2_mu = np.mean(Y,axis = 1)
    #initialise through MLE
    X = X.T
    Y = Y.T 
    W1_mu,W2_mu,EPhi1,EPhi2,Mu1,Mu2 = initialise_CCA(X,Y,m)
    
    a1 = a
    a2 = a
    b1 = np.array([b] * m)
    b2 = np.array([b] * m)
    XX = np.dot(X,X.T)
    YY = np.dot(Y,Y.T)
    

    for k  in range(max_iter):
        print "This is iteration number ", k
        z_sigma,z_mu,z_precision = update_z(d1,d2,m,W1_mu,W1_sigma,
                                    W2_mu,W2_sigma,EPhi1,EPhi2,X,Y,Mu1,Mu2,n)

        #if k %200 == 0:
           # print z_sigma,z_mu
        #Ezz = n*z_sigma + np.dot(z_mu,z_mu.T)
        Ezz_minus_z_mu_square = np.linalg.solve(z_precision,n*np.eye(m))
        Ezz = Ezz_minus_z_mu_square + np.dot(z_mu,z_mu.T)
        
        Ealpha1 = a1/b1
        Ealpha2 = a2/b2
        
        W1_sigma,W1_mu,W1_precision= update_W(z_mu,n,W1_mu,
                              Ealpha1,EPhi1,Ezz,X,d1,Mu1,m)
        W2_sigma,W2_mu,W2_precision= update_W(z_mu,n,W2_mu,
                             Ealpha2,EPhi2,Ezz,Y,d2,Mu2,m)
        
        #print W1_mu_test
        
        mu1_sigma,mu1_mu = update_mu(beta,d1,n,EPhi1,X,W1_mu,z_mu)
        mu2_sigma,mu2_mu = update_mu(beta,d2,n,EPhi2,Y,W2_mu,z_mu)
        
        Mu1 = np.array([mu1_mu]*n).T
        Mu2 = np.array([mu2_mu]*n).T
        
        a1,b1 = update_a_b(a,b,W1_mu,W1_sigma,d1)
        a2,b2 = update_a_b(a,b,W2_mu,W2_sigma,d2)
        
        Ealpha1 = a1/b1
        Ealpha2 = a2/b2
        nu1, invK1,K1,deltaK1 = update_phi(nu1_prior,n,mu1_sigma,mu1_mu,d1,
                    W1_precision,W1_sigma,W1_mu,Ezz,X,XX,z_mu,Mu1,K)
        nu2, invK2, K2,deltaK2 = update_phi(nu2_prior,n,mu2_sigma,mu2_mu,
                    d2,W2_precision,W2_sigma,W2_mu,Ezz,Y,YY,z_mu,Mu2,K)
        
        EPhi1 = nu1 * invK1
        EPhi2 = nu2 * invK2
        """
        ElogdetPhi1 = compElogPhi(nu1,d1,K1)
        ElogdetPhi2 = compElogPhi(nu2,d2,K2)
        
        
        #we start computing the lower bound
        elbo_old = elbo
        
        # X
        elbo = -0.5*n*(d1+d2) * np.log(2*np.pi)
        elbo = elbo+ 0.5*n*(ElogdetPhi1 + ElogdetPhi2)
        elbo = elbo -0.5 * np.trace(np.dot(EPhi1,deltaK1)) - \
                0.5*np.trace(np.dot(EPhi2,deltaK2))


            
        #Z
        elbo = elbo+0.5*n*(logdet(z_sigma) - np.trace(z_sigma)+m)-\
                0.5*np.sum(np.sum(z_mu*z_mu))


            
        #mu
        elbo = elbo- 0.5*(-logdet(beta*mu1_sigma) + \
                          np.trace(beta*mu1_sigma)+\
                         beta*np.sum(mu1_mu*mu1_mu)-d1)
        elbo = elbo - 0.5*(-logdet(beta*mu2_sigma) +\
                           np.trace(beta*mu2_sigma)+\
                         beta*np.sum(mu2_mu*mu2_mu)-d2)

        #alpha and W
        Elog_alpha = scipy.special.digamma(a1) - np.log(b1)
        Ealpha = a1/b1
        
        for i in range(d1):
            elbo = elbo - 0.5*(-np.sum(Elog_alpha)-logdet(W1_sigma[i])+\
                    np.trace(np.dot(np.diag(Ealpha),W1_sigma[i])) +\
                    np.sum(Ealpha*W1_mu[i,:]*W1_mu[i,:])-d1)
        elbo = elbo+m*a*np.log(b) + (a-a1)*np.sum(Elog_alpha) -\
                a1*np.sum(np.log(b1))-np.sum((b-b1)*Ealpha) - \
            m*scipy.special.gammaln(a) + m*scipy.special.gammaln(a1)
            
        Elog_alpha = scipy.special.digamma(a2) - np.log(b2)
        Ealpha = a2/b2
        for i in range(d2):
            elbo = elbo - 0.5*(-np.sum(Elog_alpha)-logdet(W2_sigma[i])+\
                    np.trace(np.dot(np.diag(Ealpha),W2_sigma[i])) +\
                    np.sum(Ealpha*W2_mu[i,:]*W2_mu[i,:])-d2)
        elbo = elbo+m*a*np.log(b) + (a-a2)*np.sum(Elog_alpha) -\
                a2*np.sum(np.log(b2))-np.sum((b-b2)*Ealpha) - \
            m*scipy.special.gammaln(a) + m*scipy.special.gammaln(a2)

        
        # Phi
        elbo = elbo + \
        0.5*(nu1_prior-d1-1)*compElogPhi(nu1_prior,d1,K*np.eye(2))
        elbo = elbo - 0.5 *(nu1-d1-1)*ElogdetPhi1
        elbo = elbo + 0.5*nu1*d1 - 0.5*nu1*np.trace(K*invK1)-\
                log_partition(nu1_prior,d1,K*np.eye(d1)) +\
                log_partition(nu1,d1,K1)
        
        elbo = elbo + \
        0.5*(nu2_prior-d2-1)*compElogPhi(nu2_prior,d2,K*np.eye(2))
        elbo = elbo - 0.5 *(nu2-d2-1)*ElogdetPhi2
        elbo = elbo + 0.5*nu2*d2 - 0.5*nu2*np.trace(K*invK2)-\
                log_partition(nu2_prior,d2,K*np.eye(d2)) +\
                log_partition(nu2,d2,K2)
        print elbo  
        """ 
    return z_sigma,z_mu,Ealpha1,Ealpha2,W1_sigma,W2_sigma,W1_mu,\
W2_mu,mu1_sigma,mu1_mu,mu2_sigma,mu2_mu,nu1,K1,nu2,K2,EPhi1,EPhi2,elbo


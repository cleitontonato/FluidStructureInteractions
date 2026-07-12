# Beam functions

import numpy as np


def mode_clafre(lamb,sig,x,n=0):
    """
    Calculates the modes of the clamped-free beam

    Parameters:
    -----------
    lamb : numpy 1D array (same length L1 as sig)
        An array that contains the values of lambda of each mode to be calculated
    sig : numpy 1D array (same size L1 as lamb)
        The values of sigma of each mode to be calculated
    x : numpy 1D array of length L2
        The values of x \in [0,1] where the modes are calculated
    n : int, defaults to 0
        The order of the derivation of the returned modes

    Returns:
    --------
        A 2D numpy array of size L2xL1
        The array of shape (L2, L1) represents each of the L1
        eigenmodes, optionally derived, as columns of length L2.
    """
    lambdax = np.outer(x, lamb)
    if n == 0:
        return np.cos(lambdax)-sig*np.sin(lambdax) + (sig*np.sinh(lambdax)-np.cosh(lambdax))
    elif n == 1:
        return -sig*lamb*np.cos(lambdax) - lamb*np.sin(lambdax) + (sig*lamb*np.cosh(lambdax) - lamb*np.sinh(lambdax))
    elif n == 2:
        return -lamb**2*np.cos(lambdax)  + sig*lamb**2*np.sin(lambdax) + (sig*lamb**2*np.sinh(lambdax)- lamb**2*np.cosh(lambdax))
    elif n == 3:
        return lamb**3*np.sin(lambdax) + sig*(lamb**3*np.cos(lambdax) + (lamb**3*np.cosh(lambdax)) - lamb**3*np.sinh(lambdax))
    elif n == 4:
        return lamb**4*mode_clacla(lamb, sig, x, n=0)
    else:
        return None

def lambdasigma_clafre(n):
    """
    Returns the values of lambda and sigma of the first n
    clamped-free beam modes.

    The eignenmode i of a beam being:
        phi_i(x) = cos(lambda_i x) - sigma_i sin(lambda_i)
                    + sigma_i sinh(lambda_i x) - cosh(lambda_i x)
        
    lambda_i is the ith root of :
        cosh(lambda_i) cos(lambda_i) + 1 = 0

    sigma_i satisfies :
        sigma_i = ( sinh (lambda_i) - sin (lambda_i) )
                    / ( cosh(lambda_i) + cos(lambda_i) )

    Parameters:
    -----------
    n : int
        The number of modes whose values of lambda and sigma
        have to be returned
        
    Returns:
    --------
        A tuple (lamb, sig) of two 1D numpy array, containing the values
        of lambda signma we sought for
    """

    valeurs = np.loadtxt('coefs_clafre.txt')
    lamb = valeurs[0:n,0]
    sig = valeurs[0:n,1]
    return lamb, sig


def coefs_clafre(n,a,b,c):
    """
    Compute matrix of coefficients of clamped free beam operators

    In pratice, it computes the integral of
            phi_i(x) derived a times
            * phi_j(x) derived b times
            * x**c
            * dx
        on the range x \in [0,1]

    Parameters:
    -----------
    n: int
        Number of modes
    a: int
        Order of derivative of first mode
    b: int
        Order of derivative of second mode
    c: int
        Order of power of multiplicative function (x**c)

    Returns:
    --------
        A numpy 2D array of shape (n,n), element (i,j)
        being the integral of
            phi_i(x) derived a times
            * phi_j(x) derived b times
            * x**c
    """
    nx = 10000
    l, s = lambdasigma_clafre(n)
    x = np.linspace(0,1,nx+1)
    m = np.zeros((n,n))
    dx = x[2]-x[1]

    integrateur = dx*np.ones(nx+1).T
    integrateur[0] = dx/2
    integrateur[-1] = dx/2

    fun3 = x**c
    
    for i in range(n):
        fun1 = mode_clafre(l[i],s[i],x,a)[:,0]*fun3
        for j in range(n):
            if (a==0 and b==1 and c==0):
                if i==j:
                    m[i,i] = 2
                else:
                    m[i,j] = 4/( ( l[i]/l[j] )**2 + (-1)**(i+j) )
            elif (a==0 and b==2 and c==0):
                if i==j:
                    m[i,i] = l[i]*s[i]*(2-l[i]*s[i])
                else:
                    m[i,j] = (4*(l[j]*s[j]-l[i]*s[i])) / ((-1)**(i+j)-(l[i]/l[j])**2)
            elif (a==0 and b==4 and c==0):
                if i==j:
                    m[i,i] = l[i]**4
                else:
                    m[i,j] = 0
            elif (a==0 and b==0 and c==0):
                if i==j:
                    m[i,i] = 1
                else:
                    m[i,j] = 0
            else:
                fun2 = mode_clafre(l[j],s[j],x,b)[:,0]
                m[i,j] = np.dot(fun1*fun2,integrateur)

    return m

def mode_clacla(lamb, sig, x, n=0):
    """
    Calculates the modes of the clamped-clamped beam
    lamb : numpy 1D array (same length L1 as sig)
        An array that contains the values of lambda of each mode to be calculated
    sig : numpy 1D array (same size L1 as lamb)
        The values of sigma of each mode to be calculated
    x : numpy 1D array of length L2
        The values of x \in [0,1] where the modes are calculated
    n : int, defaults to 0
        The order of the derivation of the returned modes

    returns :
        A 2D numpy array of size L2xL1
        The array of shape (L2, L1) represents each of the L1
        eigenmodes, optionally derived, as columns of length L2.
    """
    lambdax = np.outer(x, lamb)
    if n == 0:
        return np.cos(lambdax)-sig*np.sin(lambdax) + (sig*np.sinh(lambdax)-np.cosh(lambdax))
    elif n == 1:
        return -sig*lamb*np.cos(lambdax) - lamb*np.sin(lambdax) + (sig*lamb*np.cosh(lambdax) - lamb*np.sinh(lambdax))
    elif n == 2:
        return -lamb**2*np.cos(lambdax)  + sig*lamb**2*np.sin(lambdax) + (sig*lamb**2*np.sinh(lambdax)- lamb**2*np.cosh(lambdax))
    elif n == 3:
        return lamb**3*np.sin(lambdax) + sig*(lamb**3*np.cos(lambdax) + (lamb**3*np.cosh(lambdax)) - lamb**3*np.sinh(lambdax))
    elif n == 4:
        return lamb**4*mode_clacla(lamb, sig, x, n=0)
    else:
        return None

def lambdasigma_clacla(n):
    """
    Returns the values of lambda and sigma of the first n
    clamped-clamped beam modes.

    The eignenmode i of a beam being:
        phi_i(x) = cos(lambda_i x) - sigma_i sin(lambda_i)
                    + sigma_i sinh(lambda_i x) - cosh(lambda_i x)
        
    lambda_i is the ith root of :
        cosh(lambda_i) cos(lambda_i) + 1 = 0

    sigma_i satisfies :
        sigma_i = ( sinh (lambda_i) - sin (lambda_i) )
                    / ( cosh(lambda_i) + cos(lambda_i) )

    Parameters:
    -----------
    n : int
        The number of modes whose values of lambda and sigma
        have to be returned
        
    Returns:
    --------
        A tuple (lamb, sig) of two 1D numpy array, containing the values
        of lambda signma we sought for
    """
    l = np.zeros(n)
    s = np.zeros(n)
    l[0:5] = [4.73004074, 7.85320462, 10.9956079, 14.1371655, 12.2787597]
    s[0:5] = [0.982502215, 1.000777312, 0.999966450, 1.000001450, 0.999999937]
    for i in range(5, n):
        l[i] = (2*(i+1)+1)*np.pi/2
        s[i] = 1
    return l, s

def coefs_clacla(n, a, b, c):
    """
    Compute matrix of coefficients of clamped clamped beam operators

    In pratice, it computes the integral of
            phi_i(x) derived a times
            * phi_j(x) derived b times
            * x**c
            * dx
        on the range x \in [0,1]

    Parameters:
    -----------
    n: int
        Number of modes
    a: int
        Order of derivative of first mode
    b: int
        Order of derivative of second mode
    c: int
        Order of power of multiplicative function (x**c)

    Returns:
    --------
        A numpy 2D array of shape (n,n), element (i,j)
        being the integral of
            phi_i(x) derived a times
            * phi_j(x) derived b times
            * x**c
    """
    nx = 10000
    l, s = lambdasigma_clacla(n)
    x = np.linspace(0, 1, nx+1)
    m = np.zeros((n, n))
    dx = x[2]-x[1]

    integrateur = dx*np.ones(nx+1).T
    integrateur[0] = dx/2
    integrateur[-1] = dx/2

    fun3 = x**c
    
    for i in range(n):
        fun1 = mode_clacla(l[i], s[i], x, a)[:, 0]*fun3
        for j in range(n):
            if (a==0 and b==4 and c==0):
                if i==j:
                    m[i,i] = l[i]**4
                else:
                    m[i,j] = 0
            elif (a==0 and b==0 and c==0):
                if i==j:
                    m[i,i] = 1
                else:
                    m[i,j] = 0
            elif (a==0 and b==1 and c==0):
                if i==j:
                    m[i,i] = 0
                else:
                    m[i,j] = ( 4*(l[j]**2)*(l[i]**2) / ( l[j]**4 - l[i]**4 ) ) * ( (-1)**(i+j) - 1 )
            elif (a==0 and b==2 and c==0):
                if i==j:
                    m[i,i] = l[i]*s[i]*(2-l[i]*s[i])
                else:
                    m[i,j] = ( 4*(l[j]**2)*(l[i]**2) / ( l[j]**4 - l[i]**4 ) ) * (l[j]*s[j]-l[i]*s[i]) * ( (-1)**(i+j) + 1 )
            else:
                fun2 = mode_clacla(l[j], s[j], x, b)[:, 0]
                m[i, j] = np.dot(fun1*fun2, integrateur)

    return m

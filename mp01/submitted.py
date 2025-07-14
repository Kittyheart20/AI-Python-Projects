'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import numpy as np
from collections import Counter

def marginal_distribution_of_word_counts(texts, word0):
    '''
    Parameters:
    texts (list of lists) - a list of texts; each text is a list of words
    word0 (str) - the word that you want to count

    Output:
    Pmarginal (numpy array of length cX0) - Pmarginal[x0] = P(X0=x0), where
      X0 is the number of times that word0 occurs in a document
      cX0-1 is the largest value of X0 observed in the provided texts
    '''

    numcount = []
    sum = 0

    for doc in texts :
      numcount.append(doc.count(word0))

    Pmarginal = np.zeros(np.asarray(numcount).max() + 1)

    for i in range(len(Pmarginal)):
      count = numcount.count(i)
      Pmarginal[i] = count
      sum = sum + count
        

    Pmarginal = Pmarginal/sum

    #raise RuntimeError("You need to write this part!")
    return Pmarginal
    
def conditional_distribution_of_word_counts(texts, word0, word1):
    '''
    Parameters:
    texts (list of lists) - a list of texts; each text is a list of words
    word0 (str) - the first word that you want to count
    word1 (str) - the second word that you want to count

    Outputs: 
    Pcond (numpy array, shape=(cX0,cX1)) - Pcond[x0,x1] = P(X1=x1|X0=x0), where
      X0 is the number of times that word0 occurs in a document
      cX0-1 is the largest value of X0 observed in the provided texts
      X1 is the number of times that word1 occurs in a document
      cX1-1 is the largest value of X0 observed in the provided texts
      CAUTION: If P(X0=x0) is zero, then P(X1=x1|X0=x0) should be np.nan.
    '''
    
    count0 = []
    count1 = []

    for doc in texts:
      c0 = doc.count(word0)
      count0.append(c0)

      c1 = doc.count(word1)
      count1.append(c1)

    max_c0 = np.max(count0) + 1
    max_c1 = np.max(count1) + 1
    count01 = np.zeros((max_c0, max_c1))

    for c0, c1 in zip(count0, count1):
      count01[c0][c1] += 1


    Pmarginal0 = np.zeros(max_c0)    

    for i in range(len(Pmarginal0)):
      count = count0.count(i)
      Pmarginal0[i] = count

    #Pmarginal0 = Pmarginal0/len(texts)


    Pcond = np.zeros((max_c0, max_c1))

    for i in range(len(Pcond)):
      for j in range(len(Pcond[i])):
        if Pmarginal0[i] > 0:
            Pcond[i][j] = count01[i][j] / Pmarginal0[i]
        else:
            Pcond[i][j] = np.nan

    
    #raise RuntimeError("You need to write this part!")
    return Pcond

def joint_distribution_of_word_counts(Pmarginal, Pcond):
    '''
    Parameters:
    Pmarginal (numpy array of length cX0) - Pmarginal[x0] = P(X0=x0), where
    Pcond (numpy array, shape=(cX0,cX1)) - Pcond[x0,x1] = P(X1=x1|X0=x0)

    Output:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
      X0 is the number of times that word0 occurs in a given text,
      X1 is the number of times that word1 occurs in the same text.
      CAUTION: if P(X0=x0) then P(X0=x0,X1=x1)=0, even if P(X1=x1|X0=x0)=np.nan.
    '''

    Pjoint = np.zeros((len(Pcond), len(Pcond[0])))

    for i in range(len(Pcond)):
      for j in range(len(Pcond[i])):
            if not np.isnan(Pcond[i][j]):
              Pjoint[i][j] = Pcond[i][j] * Pmarginal[i]
            else:
               Pjoint[i][j] = 0


    #raise RuntimeError("You need to write this part!")
    return Pjoint

def mean_vector(Pjoint):
    '''
    Parameters:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
    
    Outputs:
    mu (numpy array, length 2) - the mean of the vector [X0, X1]
    '''

    mu = np.zeros(2)
    for i in range(len(Pjoint)):
      for j in range(len(Pjoint[i])):
        if not np.isnan(Pjoint[i][j]):
           mu[0] += i*Pjoint[i][j]
           mu[1] += j*Pjoint[i][j]

    

    #raise RuntimeError("You need to write this part!")
    return mu

def covariance_matrix(Pjoint, mu):
    '''
    Parameters:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
    mu (numpy array, length 2) - the mean of the vector [X0, X1]
    
    Outputs:
    Sigma (numpy array, shape=(2,2)) - matrix of variance and covariances of [X0,X1]
    '''
    Sigma = np.zeros((2, 2))
    variance0 = 0
    variance1 = 0
    covariance = 0

    for i in range(len(Pjoint)):
      for j in range(len(Pjoint[i])):
        if not np.isnan(Pjoint[i][j]):
           variance0 += Pjoint[i][j] * pow(i - mu[0], 2)
           variance1 += Pjoint[i][j] * pow(j - mu[1], 2)
           covariance += Pjoint[i][j] * (i - mu[0]) *  (j - mu[1])

    Sigma[0,0] = variance0
    Sigma[0,1] = covariance
    Sigma[1,0] = covariance
    Sigma[1,1] = variance1
  
    #raise RuntimeError("You need to write this part!")
    return Sigma

def distribution_of_a_function(Pjoint, f):
    '''
    Parameters:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
    f (function) - f should be a function that takes two
       real-valued inputs, x0 and x1.  The output, z=f(x0,x1),
       may be any hashable value (number, string, or even a tuple).

    Output:
    Pfunc (Counter) - Pfunc[z] = P(Z=z)
       Pfunc should be a collections.defaultdict or collections.Counter, 
       so that previously unobserved values of z have a default setting
       of Pfunc[z]=0.
    '''
    Pfunc = Counter()

    for i in range(len(Pjoint)):
      for j in range(len(Pjoint[i])):
        if not np.isnan(Pjoint[i][j]):
           Pfunc[f(i, j)] += Pjoint[i][j]
    
    #raise RuntimeError("You need to write this part!")
    return Pfunc
    

import numpy as np

def estimate_geometric(PX):
    '''
    @param:
    PX (numpy array of length cX): PX[x] = P(X=x), the observed probability mass function

    @return:
    p (scalar): the parameter of a matching geometric random variable
    PY (numpy array of length cX): PY[x] = P(Y=y), the first cX values of the pmf of a
      geometric random variable such that E[Y]=E[X].
    '''
    EX = 0
    PY = np.zeros(len(PX))
    
    for x in range(len(PX)):
        EX += (x + 1) * PX[x]

    p = 1 / EX

    for x in range(len(PX)):
        # Adjusting the index to match the 1-indexed formula in the 0-indexed array
        PY[x] = p * (1 - p) ** x

    #raise RuntimeError("You need to write this")
    return p, PY

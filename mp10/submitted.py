'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
import numpy as np

epsilon = 1e-3

def compute_transition(model):
    '''
    Parameters:
    model - the MDP model returned by load_MDP()

    Output:
    P - An M x N x 4 x M x N numpy array. P[r, c, a, r', c'] is the probability that the agent will move from cell (r, c) to (r', c') if it takes action a, where a is 0 (left), 1 (up), 2 (right), or 3 (down).
    '''
    M, N = model.M, model.N
    P = np.zeros((M, N, 4, M, N))
    
    for r in range(M):
        for c in range(N):
                if (model.TS[r,c]):
                    P[r, c, :, :, :] = 0
                else:
                    moves0 = [(r, c-1, model.D[r, c, 0]), (r-1, c, model.D[r, c, 2]), (r+1, c, model.D[r, c, 1])]
                    moves1 = [(r-1, c, model.D[r, c, 0]), (r, c+1, model.D[r, c, 2]), (r, c-1, model.D[r, c, 1])]
                    moves2 = [(r, c+1, model.D[r, c, 0]), (r+1, c, model.D[r, c, 2]), (r-1, c, model.D[r, c, 1])]
                    moves3 = [(r+1, c, model.D[r, c, 0]), (r, c-1, model.D[r, c, 2]), (r, c+1, model.D[r, c, 1])]

                    for nr, nc, p in moves0:
                        if ((0 <= nr) and (nr < M) and (0 <= nc) and (nc < N) and (not model.W[nr, nc])):
                            P[r, c, 0, nr, nc] = p
                        else:
                            P[r, c, 0, r, c] += p

                    for nr, nc, p in moves1:
                        if ((0 <= nr) and (nr < M) and (0 <= nc) and (nc < N) and (not model.W[nr, nc])):
                            P[r, c, 1, nr, nc] = p
                        else:
                            P[r, c, 1, r, c] += p

                    for nr, nc, p in moves2:
                        if ((0 <= nr) and (nr < M) and (0 <= nc) and (nc < N) and (not model.W[nr, nc])):
                            P[r, c, 2, nr, nc] = p
                        else:
                            P[r, c, 2, r, c] += p

                    for nr, nc, p in moves3:
                        if ((0 <= nr) and (nr < M) and (0 <= nc) and (nc < N) and (not model.W[nr, nc])):
                            P[r, c, 3, nr, nc] = p
                        else:
                            P[r, c, 3, r, c] += p
    return P

    raise RuntimeError("You need to write this part!")

def compute_utility(model, U_current, P):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()
    U_current - The current utility function, which is an M x N array
    P - The precomputed transition matrix returned by compute_transition()

    Output:
    U_next - The updated utility function, which is an M x N array
    '''
    M, N = model.M, model.N
    g = model.gamma
    U_next = np.zeros((M, N))

    for r in range(M):
        for c in range(N):
            if (model.TS[r,c]):
                U_next[r, c] = model.R[r, c]
            else:
                left = 0
                up = 0
                right = 0
                down = 0

                for rn in range(M):
                    for cn in range(N):
                        left += P[r, c, 0, rn, cn] * U_current[rn, cn]
                
                for rn in range(M):
                    for cn in range(N):
                        up += P[r, c, 1, rn, cn] * U_current[rn, cn]

                for rn in range(M):
                    for cn in range(N):
                        right += P[r, c, 2, rn, cn] * U_current[rn, cn]

                for rn in range(M):
                    for cn in range(N):
                        down += P[r, c, 3, rn, cn] * U_current[rn, cn]

                U_next[r, c] = model.R[r, c] + g * max(left, up, right, down)

    return U_next

    raise RuntimeError("You need to write this part!")


def value_iterate(model):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()

    Output:
    U - The utility function, which is an M x N array
    '''
    M, N = model.M, model.N
    U = np.zeros((M, N))
    P = compute_transition(model)

    while True:
        U_next = compute_utility(model, U, P)
        delta = np.max(np.abs(U_next - U))
        U = U_next
        if delta < epsilon:
            break
    print("U: ", U)
    return U

    raise RuntimeError("You need to write this part!")

def policy_evaluation(model):
    '''
    Parameters:
    model - The MDP model returned by load_MDP();
    
    Output:
    U - The converged utility function, which is an M x N array
    '''

    M, N = model.M, model.N
    U = np.zeros((M, N))  
    g = model.gamma
    delta = epsilon  

    while delta >= epsilon:
        delta = 0  
        U_next = np.zeros((M, N))  

        for r in range(M):
            for c in range(N):
                if (model.TS[r,c]):
                    U_next[r, c] = model.R[r, c] 

                else:
                    util = 0
                    
                    for rn in range(M):
                        for cn in range(N):
                            util += model.FP[r, c, rn, cn] * U[rn, cn]

                    U_next[r, c] = model.R[r, c] + g * util

        delta = np.max(np.abs(U_next - U))  
        
        U = U_next  

    return U
    raise RuntimeError("You need to write this part!")

import numpy as np

def A_generation(n, prob_p, prob_n):
    def generate():
        A = np.zeros((n, n))
        edge = []
        for i in range(n):
            for j in range(n):
                temp = np.random.random()
                if temp < prob_p:
                    sign = 1
                elif temp < prob_p + prob_n:
                    sign = -1
                else:
                    sign = 0
                if sign != 0:
                    edge.append([i, j])
                A[i, j ] = sign
        return A, edge
    
    def connect(edge, n):
        def find(arr, i):
            if arr[i] != i:
                arr[i] = find(arr, arr[i])
            return arr[i]
        def union(arr, i, j):
            p = find(arr, i)
            q = find(arr, j)
            arr[p] = q
        arr = [i for i in range(n)]
        for [x, y] in edge:
            union(arr, x, y)
        res = True
        for i in range(n):
            if find(arr, i) != find(arr, 0):
                res = False
        return res
    
    rand_A = np.zeros((n, n))
    while True:
        A, edge = generate()
        if connect(edge, n):
            rand_A = A
            break
    return rand_A

def generate_data(A, n, time_points, cell_num):
    time_step = 0.01 
    tp_num = len(time_points) # number of time points
    V = 30
    beta = 1
    theta = 0.2
    sigma = 0.1
    
    gene_num = n
    data = np.zeros((tp_num, cell_num, gene_num))
    curr_time = 0.0
    curr_exp = np.random.rand(cell_num, gene_num)
    while curr_time <= time_points[-1] + 1e-6:
        for i in range(tp_num):
            if abs(curr_time - time_points[i]) < 1e-6:
                data[i, :, :] = curr_exp
        curr_time += time_step
        diff = np.zeros((cell_num, gene_num))
        for sim in range(cell_num):
            for j in range(gene_num):
                pro = beta
                for i in range(gene_num):
                    pro *= 1 + A[i, j] * curr_exp[sim, i] / (1 + curr_exp[sim, i])
                diff[sim, j] = V * time_step * pro
        diff -= V * time_step * theta * curr_exp
        diff += np.multiply(curr_exp, np.random.normal(0, \
                sigma * np.sqrt(time_step), (cell_num, gene_num)))
        curr_exp += diff 
    return data
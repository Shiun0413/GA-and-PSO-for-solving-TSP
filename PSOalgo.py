#粒子群算法TSP问题完整代码：
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

#計算距離矩陣
def clac_distance(X, Y):
    """
    計算兩個城市之間的距離，
    :param X: 城市X的坐標.np.array數組
    :param Y: 城市Y的坐標.np.array數組
    :return:
    """
    distance_matrix = np.zeros((city_num, city_num))
    for i in range(city_num):
        for j in range(city_num):
            if i == j:
                continue

            distance = np.sqrt((X[i] - X[j]) ** 2 + (Y[i] - Y[j]) ** 2)
            distance_matrix[i][j] = distance

    return distance_matrix

#定義總距離(路程即適應度值)
def fitness_func(distance_matrix, x_i):
    """
    適應度函數
    :param distance_matrix: 城市距離矩陣
    :param x_i: PSO的一個解（路徑序列）
    :return:
    """
    total_distance = 0
    for i in range(1, city_num):
        start_city = x_i[i - 1]
        end_city = x_i[i]
        total_distance += distance_matrix[start_city][end_city]
    total_distance += distance_matrix[x_i[-1]][x_i[0]]  # 從最後的城市返回出發的城市

    return total_distance

#定義速度更新函數
def get_ss(x_best, x_i, r):
    """
    計算交換序列，即x2結果交換序列ss得到x1，對應PSO速度更新公式中的 r1(pbest-xi) 和 r2(gbest-xi)
    :param x_best: pbest or gbest
    :param x_i: 粒子當前的解
    :param r: 隨機因子
    :return:
    """
    velocity_ss = []
    for i in range(len(x_i)):
        if x_i[i] != x_best[i]:
            j = np.where(x_i == x_best[i])[0][0]
            so = (i, j, r)  # 得到交换子
            velocity_ss.append(so)
            x_i[i], x_i[j] = x_i[j], x_i[i]  # 執行交換操作

    return velocity_ss

# 定義位置更新函數
def do_ss(x_i, ss):
    """
    執行交換操作
    :param x_i:
    :param ss: 由交換子組成的交換序列
    :return:
    """
    for i, j, r in ss:
        rand = np.random.random()
        if rand <= r:
            x_i[i], x_i[j] = x_i[j], x_i[i]
    return x_i

def draw(best):
    result_x = [0 for col in range(city_num+1)]
    result_y = [0 for col in range(city_num+1)]
    
    for i in range(city_num):
        result_x[i] = city_x[best[i]]
        result_y[i] = city_y[best[i]]
    result_x[city_num] = result_x[0]
    result_y[city_num] = result_y[0]
    plt.xlim(0, 100)  # 限定橫軸的範圍
    plt.ylim(0, 100)  # 限定縱軸的範圍
    plt.plot(result_x, result_y, marker='>', mec='r', mfc='w',label=u'路线')
    plt.legend()  # 讓圖例生效
    plt.margins(0)
    plt.subplots_adjust(bottom=0.15)
    for i in range(len(best)):
        plt.text(result_x[i] + 0.05, result_y[i] + 0.05, str(best[i]+1), color='red')
    plt.xlabel('橫坐標')
    plt.ylabel('縱坐標')
    plt.title('軌跡圖')
    plt.show()
     
def print_route(route):
    result_cur_best=[]
    for i in route:
        result_cur_best+=[i]
    for i in range(len(result_cur_best)):
        result_cur_best[i] += 1
    result_path = result_cur_best
    result_path.append(result_path[0])
    return result_path

if __name__=="__main__":
    
    #讀取城市坐標
    coord = []
    with open("data.txt", "r") as lines:
        lines = lines.readlines()
    for line in lines:
        xy = line.split()
        coord.append(xy)
    coord = np.array(coord)
    w, h = coord.shape
    coordinates = np.zeros((w, h), float)
    for i in range(w):
        for j in range(h):
            coordinates[i, j] = float(coord[i, j])
    city_x=coordinates[:,0]
    city_y=coordinates[:,1]
    #城市數量
    city_num = coordinates.shape[0]
    
    # 參數設置
    size = 40       #粒子数量
    r1 = 0.4         #個體學習因子
    r2 = 0.4         #社會學習因子
    iter_max_num = 50    #迭代次數
    fitness_value_lst = []

    distance_matrix = clac_distance(city_x, city_y)

    # 初始化種群各個粒子的位置，作為個體的歷史最優pbest
    pbest_init = np.zeros((size, city_num), dtype=np.int64)
    for i in range(size):
        pbest_init[i] = np.random.choice(list(range(city_num)), size=city_num, replace=False)

    # 計算每個粒子對應的適應度
    pbest = pbest_init
    pbest_fitness = np.zeros((size, 1))
    for i in range(size):
        pbest_fitness[i] = fitness_func(distance_matrix, x_i=pbest_init[i])

    # 計算全局適應度和對應的gbest
    gbest = pbest_init[pbest_fitness.argmin()]
    gbest_fitness = pbest_fitness.min()

    # 記錄算法迭代效果
    fitness_value_lst.append(gbest_fitness)

    # 迭代過程
    for i in range(iter_max_num):
        # 控制迭代次數
        for j in range(size):
            # 遍歷每個粒子
            pbest_i = pbest[j].copy()
            x_i = pbest_init[j].copy()

            # 計算交換序列，即 v = r1(pbest-xi) + r2(gbest-xi)
            ss1 = get_ss(pbest_i, x_i, r1)
            ss2 = get_ss(gbest, x_i, r2)
            ss = ss1 + ss2
            x_i = do_ss(x_i, ss)

            fitness_new = fitness_func(distance_matrix, x_i)
            fitness_old = pbest_fitness[j]
            if fitness_new < fitness_old:
                pbest_fitness[j] = fitness_new
                pbest[j] = x_i

            gbest_fitness_new = pbest_fitness.min()
            gbest_new = pbest[pbest_fitness.argmin()]
            if gbest_fitness_new < gbest_fitness:
                gbest_fitness = gbest_fitness_new
                gbest = gbest_new           
        fitness_value_lst.append(gbest_fitness)

    # 輸出迭代結果
    print("最佳路線：", print_route(gbest))
    print("路徑長：", gbest_fitness)
    
    #繪圖   
    sns.set_style('whitegrid')
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False
    draw(gbest)

    plt.show()


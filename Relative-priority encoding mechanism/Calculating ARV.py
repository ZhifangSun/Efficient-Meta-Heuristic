"""
类别: 算法
名称: 基于退火算子和差分算子的灰狼优化算法
作者: 孙质方
邮件: zf_sun@vip.hnist.edu.cn
日期: 2021年12月26日
说明:
"""
import random
import numpy
import math
import matplotlib.pyplot as plt
from time import *
import tracemalloc
import numpy as np


# 种群初始化
def initialtion(NP,len_x,value_down_range,value_up_range):
    np_list = []  # 种群，染色体
    for i in range(0, NP):
        x_list = []  # 个体，基因
        for j in range(0, len_x):
            x_list.append(value_down_range + random.random() * (value_up_range - value_down_range))
        np_list.append(x_list)
    return np_list


# 列表相减
def substract(a_list, b_list,lb,ub):
    a = len(a_list)
    new_list = []
    for i in range(0, a):
        temp=a_list[i] - b_list[i]
        if temp < lb:
            temp = lb
        elif temp > ub:
            temp = ub
        new_list.append(temp)
    return new_list


# 列表相加
def add(a_list, b_list,lb,ub):
    a = len(a_list)
    new_list = []
    for i in range(0, a):
        temp = a_list[i] + b_list[i]
        if temp < lb:
            temp = lb
        elif temp > ub:
            temp = ub
        new_list.append(temp)
    return new_list


# 列表的数乘
def multiply(a, b_list,lb,ub):
    b = len(b_list)
    new_list = []
    for i in range(0, b):
        temp = a * b_list[i]
        if temp < lb:
            temp = lb
        elif temp > ub:
            temp = ub
        new_list.append(temp)
    return new_list

#灰狼
def init():
    Positions = numpy.zeros((SearchAgents_no, dim))
    for i in range(dim):  # 形成SearchAgents_no*30个数[-100，100)以内
        Positions[:, i] = numpy.random.uniform(0, 1, SearchAgents_no) * (ub - lb) + lb
    return Positions

def PSO_GWO_init():
    Positions = numpy.zeros((SearchAgents_no, dim))
    for i in range(SearchAgents_no):
        x0=random.random()
        x0_list=[-1,-1,-1,x0]
        while x0==0.2 or x0==0.4 or x0==0.6 or x0==0.8:
            x0 = random.random()
        Positions[i,0]=x0
        for j in range(1,dim):
            if Positions[i, j - 1] <= 0.5:
                Positions[i, j] = Positions[i, j - 1] * 2
                if Positions[i, j]==0 or Positions[i, j]==0.25 or Positions[i, j]==0.5 or Positions[i, j]==0.75 or (Positions[i, j] in x0_list):
                    x0=(x0+random.random()) % 1
                    x0_list.append(x0)
                    x0_list.remove(x0_list[0])
                    Positions[i, j]=x0
            else:
                Positions[i, j] = (1 - Positions[i, j - 1]) * 2
                if Positions[i, j]==0 or Positions[i, j]==0.25 or Positions[i, j]==0.5 or Positions[i, j]==0.75 or (Positions[i, j] in x0_list):
                    x0=(x0+random.random()) % 1
                    x0_list.append(x0)
                    x0_list.remove(x0_list[0])
                    Positions[i, j]=x0
    for i in range(SearchAgents_no):
        for j in range(dim):
            Positions[i,j]=Positions[i,j]* (ub - lb) + lb
    return Positions

def beta_init():
    Positions = numpy.zeros((SearchAgents_no, dim))
    for i in range(SearchAgents_no):
        for j in range(dim):
            Positions[i, j] = random.betavariate(1.2,1.2) * (ub - lb) + lb
    return Positions

def re_gene(Positions,objf,lenth):
    init_Positions = numpy.zeros((SearchAgents_no +lenth, dim))
    # positions = numpy.zeros((SearchAgents_no, dim))
    for i in range(dim):  # 形成SearchAgents_no*30个数[-100，100)以内
        init_Positions[:SearchAgents_no, i] = Positions[:, i]
        for j in range(lenth):
            if init_Positions[j][i]>=(lb + ub)/2:
                init_Positions[SearchAgents_no + j][i] = (lb + ub)/2 - (init_Positions[j][i]-(lb + ub)/2)
            else:
                init_Positions[SearchAgents_no + j][i] = (lb + ub) / 2 + ((lb + ub) / 2-init_Positions[j][i] )
    init_Positions=numpy.array(sorted(init_Positions,key=lambda x:objf(x)))
    for i in range(SearchAgents_no):
        Positions[i, :] = init_Positions[i,:]
    return Positions

def SMGWO(Positions,objf, lb, ub, dim, SearchAgents_no, Max_iter,k,Tmax,Tmin,t,begin_time):
    #F = 0.6  # 缩放因子
    tim=time()-begin_time
    T = Tmax
    T0 = Tmax
    K = k
    EPS = Tmin
    ccc = []
    ddd = []
    Positions=re_gene(Positions,objf,SearchAgents_no)
    Convergence_curve_1 = []
    best_ans=objf(Positions[0])
    #迭代寻优
    l=0
    while T > EPS and tim<t:
        # Positions=numpy.array(sorted(Positions,key=lambda x:objf(x)))
        # Alpha_score=objf(Positions[0])
        Alpha_pos =Positions[0]
        Beta_pos =Positions[1]
        Delta_pos =Positions[2]
        # 以上的循环里，Alpha、Beta、Delta
        Gamma=math.log(EPS/T0,K)
        a = 2 - l * ((2) / Gamma)  #   a从2线性减少到0
        # a = 2-(math.log(1+1.3*math.tan((l/Gamma)**3),2))**6
        # print(a)
        for i in range(0, SearchAgents_no):
            r1=np.random.rand(dim)
            r2=np.random.rand(dim)
            A1 = 2 * a * r1 - a  # (-a.a)
            C1 = 2 * r2  # (0,2)
            D_alpha = np.abs(
                C1 * Alpha_pos - Positions[i])  # abs() 函数返回数字的绝对值。Alpha_pos[j]表示Alpha位置，Positions[i,j])候选灰狼所在位置
            X1 = Alpha_pos - A1 * D_alpha  # X1表示根据alpha得出的下一代灰狼位置向量

            r3=np.random.rand(dim)
            r4=np.random.rand(dim)
            A2 = 2 * a * r3 - a
            C2 = 2 * r4
            D_beta = np.abs(C2 * Beta_pos - Positions[i])
            X2 = Beta_pos - A2 * D_beta

            r5=np.random.rand(dim)
            r6=np.random.rand(dim)
            A3 = 2 * a * r5 - a
            C3 = 2 * r6
            D_delta = np.abs(C3 * Delta_pos - Positions[i])
            X3 = Delta_pos - A3 * D_delta

            temp = (2 / 3) * (l / Gamma) * X1 + (1 / 3) * X2 + ((2 / 3) - (2 / 3) * (l / Gamma)) * X3
            temp[temp < lb] = lb
            temp[temp > ub] = ub
            Positions[i] = temp

        #差分进化算子
        r1 = random.randint(0, SearchAgents_no - 1)
        r2 = random.randint(0, SearchAgents_no - 1)
        while r2 == r1:
            r2 = random.randint(0, SearchAgents_no - 1)
        r3 = random.randint(0, SearchAgents_no - 1)
        while r3 == r2 | r3 == r1:
            r3 = random.randint(0, SearchAgents_no - 1)
        # 在DE中常见的差分策略是随机选取种群中的两个不同的个体，将其向量差缩放后与待变异个体进行向量合成
        # F为缩放因子F越小，算法对局部的搜索能力更好，F越大算法越能跳出局部极小点，但是收敛速度会变慢。此外，F还影响种群的多样性。
        F=1.2*(T0-T)/(T0-EPS)
        v_list = add(Positions[r1], multiply(F, substract(Positions[r2], Positions[r3],lb,ub),lb,ub),lb,ub)
        v_list_ans = objf(numpy.array(v_list))
        # Positions=numpy.array(sorted(Positions, key=lambda x: objf(x)))
        ant = 0
        lab=[]
        for key in range(SearchAgents_no):
            jf=objf(Positions[key])
            lab.append(jf)
            if jf > v_list_ans:
                ant += 1
        lab=np.array(lab)
        arrIndex = np.array(lab).argsort()
        Positions = Positions[arrIndex]
        p=1/(1+math.exp((ant/SearchAgents_no)*T))#退火算子
        # p = math.exp(((ant / SearchAgents_no) - 1))
        T*=K
        l+=1
        if random.random() <= p:
            for j in range(dim):
                Positions[len(Positions)-1,j]=v_list[j]
        re_lenth = SearchAgents_no-SearchAgents_no * (Gamma - l) / Gamma  # re_lenth从pop_size线性减小到1
        # print(re_lenth)
        if re_lenth < 1:
            re_lenth = 1
        Positions=re_gene(Positions,objf,math.ceil(re_lenth))
        Alpha_score = objf(Positions[0])
        if Alpha_score<best_ans:
            best_ans=Alpha_score
        Convergence_curve_1.append(best_ans)

        tim=time()-begin_time
    return Convergence_curve_1


def GWO(Positions,objf, lb, ub, dim, SearchAgents_no, Max_iter):
    Convergence_curve_2 = []
    #迭代寻优
    for l in range(0, Max_iter):  # 迭代1000
        Positions = numpy.array(sorted(Positions, key=lambda x: objf(x)))
        Alpha_score = objf(Positions[0])
        Alpha_pos = list(Positions[0])
        Beta_pos = list(Positions[1])
        Delta_pos = list(Positions[2])
        a = 2 - l * ((2) / Max_iter);  #   a从2线性减少到0

        for i in range(0, SearchAgents_no):
            for j in range(0, dim):
                r1 = random.random()  # r1 is a random number in [0,1]主要生成一个0-1的随机浮点数。
                r2 = random.random()  # r2 is a random number in [0,1]

                A1 = 2 * a * r1 - a;  #  (-a.a)
                C1 = 2 * r2;  #  (0.2)
                # D_alpha表示候选狼与Alpha狼的距离
                D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j]);  # abs() 函数返回数字的绝对值。Alpha_pos[j]表示Alpha位置，Positions[i,j])候选灰狼所在位置
                X1 = Alpha_pos[j] - A1 * D_alpha;  # X1表示根据alpha得出的下一代灰狼位置向量
                #if i==0 and l==0:
                    #print(f'A1:{A1};C1:{C1};D_alpha:{D_alpha};X1:{X1}')

                r1 = random.random()
                r2 = random.random()

                A2 = 2 * a * r1 - a;  #
                C2 = 2 * r2;

                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j]);
                X2 = Beta_pos[j] - A2 * D_beta;
                #if i==0 and l==0:
                    #print(f'A2:{A2};C2:{C2};D_beta:{D_beta};X2:{X2}')

                r1 = random.random()
                r2 = random.random()

                A3 = 2 * a * r1 - a;
                C3 = 2 * r2;

                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j]);
                X3 = Delta_pos[j] - A3 * D_delta;
                temp = (X1 + X2 + X3) / 3
                if temp < lb:
                    temp = (ub + lb) / 2 - (-temp) % ((ub - lb) / 2)
                elif temp > ub:
                    temp = (ub + lb) / 2 + (temp) % ((ub - lb) / 2)
                Positions[i, j] = temp
                # 候选狼的位置更新为根据Alpha、Beta、Delta得出的下一代灰狼地址。

        Convergence_curve_2.append(Alpha_score)
    return Convergence_curve_2

def CMAGWO(Positions,objf, lb, ub, dim, SearchAgents_no, Max_iter):
    Convergence_curve_2 = []
    Positions = numpy.array(sorted(Positions, key=lambda x: objf(x)))
    Alpha_score = objf(Positions[0])
    Alpha_pos = list(Positions[0])
    Beta_pos = list(Positions[1])
    Delta_pos = list(Positions[2])
    #迭代寻优
    for l in range(0, Max_iter//2):  # 迭代1000
        a = 2 - l * ((2) / Max_iter//2);  #   a从2线性减少到0

        for i in range(0, SearchAgents_no):
            for j in range(0, dim):
                r1 = random.random()  # r1 is a random number in [0,1]主要生成一个0-1的随机浮点数。
                r2 = random.random()  # r2 is a random number in [0,1]

                A1 = 2 * a * r1 - a;  #  (-a.a)
                C1 = 2 * r2;  #  (0.2)
                # D_alpha表示候选狼与Alpha狼的距离
                D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j]);  # abs() 函数返回数字的绝对值。Alpha_pos[j]表示Alpha位置，Positions[i,j])候选灰狼所在位置
                X1 = Alpha_pos[j] - A1 * D_alpha;  # X1表示根据alpha得出的下一代灰狼位置向量
                #if i==0 and l==0:
                    #print(f'A1:{A1};C1:{C1};D_alpha:{D_alpha};X1:{X1}')

                r1 = random.random()
                r2 = random.random()

                A2 = 2 * a * r1 - a;  #
                C2 = 2 * r2;

                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j]);
                X2 = Beta_pos[j] - A2 * D_beta;
                #if i==0 and l==0:
                    #print(f'A2:{A2};C2:{C2};D_beta:{D_beta};X2:{X2}')

                r1 = random.random()
                r2 = random.random()

                A3 = 2 * a * r1 - a;
                C3 = 2 * r2;

                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j]);
                X3 = Delta_pos[j] - A3 * D_delta;
                temp = (X1 + X2 + X3) / 3
                if temp < lb:
                    temp = (ub + lb) / 2 - (-temp) % ((ub - lb) / 2)
                elif temp > ub:
                    temp = (ub + lb) / 2 + (temp) % ((ub - lb) / 2)
                Positions[i, j] = temp
                # 候选狼的位置更新为根据Alpha、Beta、Delta得出的下一代灰狼地址。
        Positions = numpy.array(sorted(Positions, key=lambda x: objf(x)))
        Alpha_score = objf(Positions[0])
        Alpha_pos = list(Positions[0])
        Beta_pos = list(Positions[1])
        Delta_pos = list(Positions[2])
        Convergence_curve_2.append(Alpha_score)

#CMA-ES
    # lambd=4+math.floor(3*math.log(dim))#种群大小
    mu=3#精英种群大小
    weights=[1,0.1,0.01]#精英种群权重
    weights=np.array(weights)
    weights=weights/numpy.sum(weights)#精英集中成员的权重归一化Normalize recombination weights array
    mueff=(numpy.sum(weights))**2/numpy.sum(weights**2)#精英集中成员权重的方差-有效性Variance-effectiveness of sum w_i x_i
    pc=numpy.zeros(dim)#协方差的演化路径Evolution paths for C
    ps = numpy.zeros(dim)#sigma步长的演化路径p_sigma
    cc=(4+mueff/dim)/(dim+4+2*mueff/dim)#协方差路径的系数，cc为alph_cp,Time constant for cumulation for C
    cs = (mueff + 2) / (dim + mueff + 5)#alpa_sigma，t-const for cumulation for sigma control
    c1 = 2 / ((dim + 1.3) ** 2 + mueff)#alpha_c1，Learning rate for rank-one update of C
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))#alpha_c_lambda，and for rank-mu update
    damps = 1 + 2 * max(0.0, math.sqrt((mueff - 1) / (dim + 1)) - 1) + cs#d_sigma，Damping for sigma
    chiN = dim ** 0.5 * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))#高斯分布二范数的期望Expectation of ||N(0,I)|| == norm(randn(N,1))


    mean_alpha=Positions[0]
    mean_beta=Positions[1]
    mean_delta=Positions[2]
    xmean = (mean_alpha + mean_beta + mean_delta) / 3
    CM=np.cov(Positions,rowvar=False)
    ave_x=Positions.mean(axis=0)
    sum_d=0
    for i in range(SearchAgents_no):
        sum_d+=numpy.linalg.norm(Positions[i] - ave_x)
    theta_alpha=numpy.linalg.norm(mean_alpha - ave_x)/sum_d
    theta_beta=numpy.linalg.norm(mean_beta - ave_x)/sum_d
    theta_delta=numpy.linalg.norm(mean_delta - ave_x)/sum_d

    eigeneval=0
    l=0
    while l< Max_iter//2:
        # print(CM)
        # print("***")
        sigma = (theta_alpha + theta_beta + theta_delta) / 3
        # 协方差矩阵CM的特征值矩阵和特征向量矩阵(正交基矩阵)eigen_vals, eigen_vecs
        eigen_vals, eigen_vecs = np.linalg.eigh(CM)
        eigen_vals=np.sqrt(eigen_vals)
        # print(eigen_vals, eigen_vecs)
        invsqrtC = np.dot(np.dot(np.transpose(eigen_vecs) , numpy.diag(1/eigen_vals)) , eigen_vecs)
        for i in range(0, SearchAgents_no):
            X1=mean_alpha+theta_alpha*np.dot(eigen_vals*numpy.random.randn(dim),eigen_vecs)
            X2 = mean_beta + theta_beta * 0.1*np.dot(eigen_vals*numpy.random.randn(dim),eigen_vecs)
            X3 = mean_delta + theta_delta * 0.01*np.dot(eigen_vals*numpy.random.randn(dim),eigen_vecs)

            temp = (X1 + X2 + X3) / 3
            for j in range(0, dim):
                if temp[j] < lb:
                    temp[j] = (ub + lb) / 2 - (-temp[j]) % ((ub - lb) / 2)
                elif temp[j] > ub:
                    temp[j] = (ub + lb) / 2 + (temp[j]) % ((ub - lb) / 2)
            Positions[i] = temp
            l+=1
        Positions = numpy.array(sorted(Positions, key=lambda x: objf(x)))

        # xmean=weights*Positions[0:mu]
        mean_alpha = Positions[0]
        mean_beta = Positions[1]
        mean_delta = Positions[2]

        #mean_alpha
        xold=xmean
        # print(f'11{xold}')
        xmean=np.dot(weights,Positions[0:mu])
        # print(f'22{xmean}')
        # print(f'33{invsqrtC}')
        #更新进化路径，与理论不一致没有用精英集大小mu，mueff与mu有关
        ps = (1 - cs) * ps + np.dot((xmean - xold),math.sqrt(cs * (2 - cs) * mueff) * invsqrtC) / sigma
        # print("!")
        # print(np.linalg.norm(ps))
        hsig = np.linalg.norm(ps) / math.sqrt(1 - (1 - cs) ** (2 * l /SearchAgents_no)) / chiN <1.4 + 2 / (dim + 1)
        # print("@")
        # print(hsig)
        #cc为alph_cp。hsig为记录的符号信息
        pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * (xmean - xold) / sigma
        # print("#")
        # print(pc)
        artmp = (1 / sigma) * (Positions[0:mu]-np.tile(xold, (mu, 1)))
        # print("$")
        # print(artmp)
        CM = (1 - c1 - cmu) * CM+ c1 * (np.dot(np.transpose(pc),pc)+ (1-hsig) * cc*(2-cc) * CM)+ \
             np.dot(np.dot(cmu * np.transpose(artmp) , numpy.diag(weights)) , artmp)

        # print((cs/damps)*(np.linalg.norm(ps)/chiN - 1))
        theta_alpha=theta_alpha* math.exp((cs/damps)*(np.linalg.norm(ps)/chiN - 1))
        theta_beta = theta_beta * math.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
        theta_delta = theta_delta * math.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
        if l-eigeneval>SearchAgents_no/(c1+cmu)/dim/10:
            eigeneval=l
            CM=numpy.triu(CM)+np.transpose(numpy.triu(CM,1))

        Alpha_score = objf(Positions[0])
        Convergence_curve_2.append(Alpha_score)
    return Convergence_curve_2

def PSO_GWO(Positions,objf, lb, ub, dim, SearchAgents_no, Max_iter):
    c1=2.05
    c2=2.05
    Convergence_curve_2 = []
    #迭代寻优
    Positions = numpy.array(sorted(Positions, key=lambda x: objf(x)))
    pbest = Positions
    Alpha_pos = list(Positions[0])
    Beta_pos = list(Positions[1])
    Delta_pos = list(Positions[2])
    for l in range(0, Max_iter):  # 迭代1000
        Alpha_score = objf(numpy.array(Alpha_pos))
        a = 0 - (0-2)*(l/Max_iter)**2  #   a从2线性减少到0

        for i in range(0, SearchAgents_no):
            for j in range(0, dim):
                r1 = random.random()  # r1 is a random number in [0,1]主要生成一个0-1的随机浮点数。
                r2 = random.random()  # r2 is a random number in [0,1]

                A1 = 2 * a * r1 - a;  #  (-a.a)
                C1 = 2 * r2;  #  (0.2)
                # D_alpha表示候选狼与Alpha狼的距离
                D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j]);  # abs() 函数返回数字的绝对值。Alpha_pos[j]表示Alpha位置，Positions[i,j])候选灰狼所在位置
                X1 = Alpha_pos[j] - A1 * D_alpha;  # X1表示根据alpha得出的下一代灰狼位置向量
                #if i==0 and l==0:
                    #print(f'A1:{A1};C1:{C1};D_alpha:{D_alpha};X1:{X1}')

                r1 = random.random()
                r2 = random.random()

                A2 = 2 * a * r1 - a;  #
                C2 = 2 * r2;

                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j]);
                X2 = Beta_pos[j] - A2 * D_beta;
                #if i==0 and l==0:
                    #print(f'A2:{A2};C2:{C2};D_beta:{D_beta};X2:{X2}')

                r1 = random.random()
                r2 = random.random()

                A3 = 2 * a * r1 - a;
                C3 = 2 * r2;

                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j]);
                X3 = Delta_pos[j] - A3 * D_delta;

                w1=X1/(X1 + X2 + X3)
                w2 = X2 / (X1 + X2 + X3)
                w3 = X3 / (X1 + X2 + X3)
                r1 = random.random()
                r2 = random.random()
                temp = c1*r1*(w1*X1+w2*X2+w3*X3)+c2*r2*(pbest[i,j]-Positions[i, j])
                # print(temp)
                # temp = (X1 + X2 + X3) / 3
                if temp < lb:
                    temp = (ub + lb) / 2 - (-temp) % ((ub - lb) / 2)
                elif temp > ub:
                    temp = (ub + lb) / 2 + (temp) % ((ub - lb) / 2)
                Positions[i, j] = temp
                # 候选狼的位置更新为根据Alpha、Beta、Delta得出的下一代灰狼地址。
            ans=objf(Positions[i])
            if ans<objf(numpy.array(Alpha_pos)):
                Delta_pos=Beta_pos
                Beta_pos=Alpha_pos
                Alpha_pos=list(Positions[i])
            elif ans<objf(numpy.array(Beta_pos)):
                Delta_pos=Beta_pos
                Beta_pos=list(Positions[i])
            elif ans<objf(numpy.array(Delta_pos)):
                Delta_pos = list(Positions[i])

            if ans<objf(pbest[i]):
                pbest[i]=Positions[i]

        Convergence_curve_2.append(Alpha_score)
    Convergence_curve_2.append(objf(numpy.array(Alpha_pos)))
    return Convergence_curve_2

def vm_GWO(Positions,objf, lb, ub, dim, SearchAgents_no, Max_iter):
    Convergence_curve_2 = []
    #迭代寻优
    for l in range(0, Max_iter):  # 迭代1000
        Positions = numpy.array(sorted(Positions, key=lambda x: objf(x)))
        Alpha_score = objf(Positions[0])
        Alpha_pos = list(Positions[0])
        Beta_pos = list(Positions[1])
        Delta_pos = list(Positions[2])
        # a = 2 - l * ((2) / Max_iter);  #   a从2线性减少到0
        a = 1.6*math.exp(-l/Max_iter)
        for i in range(0, SearchAgents_no):
            for j in range(0, dim):
                r1 = random.random()  # r1 is a random number in [0,1]主要生成一个0-1的随机浮点数。
                r2 = random.random()  # r2 is a random number in [0,1]

                A1 = 2 * a * r1 - a;  #  (-a.a)
                C1 = 2 * r2;  #  (0.2)
                # D_alpha表示候选狼与Alpha狼的距离
                D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j]);  # abs() 函数返回数字的绝对值。Alpha_pos[j]表示Alpha位置，Positions[i,j])候选灰狼所在位置
                X1 = Alpha_pos[j] - A1 * D_alpha;  # X1表示根据alpha得出的下一代灰狼位置向量
                #if i==0 and l==0:
                    #print(f'A1:{A1};C1:{C1};D_alpha:{D_alpha};X1:{X1}')

                r1 = random.random()
                r2 = random.random()

                A2 = 2 * a * r1 - a;  #
                C2 = 2 * r2;

                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j]);
                X2 = Beta_pos[j] - A2 * D_beta;
                #if i==0 and l==0:
                    #print(f'A2:{A2};C2:{C2};D_beta:{D_beta};X2:{X2}')

                r1 = random.random()
                r2 = random.random()

                A3 = 2 * a * r1 - a;
                C3 = 2 * r2;

                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j]);
                X3 = Delta_pos[j] - A3 * D_delta;
                #if i==0 and l==0:
                    #print(f'A3:{A3};C3:{C3};D_delta:{D_delta};X3:{X3}')
                fai=0.5*math.atan(l)
                ceita=(2/math.pi)*math.acos(1/3)*math.atan(l)
                w1=math.cos(ceita)
                w2=0.5*math.sin(ceita)*math.cos(fai)
                temp = w1*X1 + w2*X2 + (1-w1-w2)*X3
                if temp < lb:
                    temp = (ub+lb)/2-(-temp)%((ub-lb)/2)
                elif temp > ub:
                    temp = (ub+lb)/2+(temp)%((ub-lb)/2)
                Positions[i, j] = temp
                #候选狼的位置更新为根据Alpha、Beta、Delta得出的下一代灰狼地址。

        Convergence_curve_2.append(Alpha_score)
    return Convergence_curve_2

#适应度函数

def F1(x):#[-100,100]
    dim=30
    lb=-100
    ub=100
    cc = 0
    for i in range(len(x)):
        cc+=(i+1)*(x[i]**4)
    s=cc+random.random()
    return s

def F2(x):#[-100,100]
    dim=30
    lb=-100
    ub=100
    cc=0
    c=1
    for i in range(len(x)):
        cc+=abs(x[i])
        c*=abs(x[i])
    s=cc+c
    return s

def F3(x):#[-100,100]
    dim=30
    lb=-100
    ub=100
    cc=0
    for i in range(1,len(x)+1):
        c=0
        for j in range(0,i):
            c+=x[j]
        cc+=c**2
    s=cc
    return s

def F4(x):#[-100,100]
    dim=30
    lb=-100
    ub=100
    cc=-99999999
    for i in range(0,len(x)):
        if abs(x[i])>cc:
            cc=abs(x[i])
    s=cc
    return s

def F5(x):#[-100,100]
    dim=30
    lb=-100
    ub=100
    ss = numpy.sum(x ** 2)
    s = ss**2
    return s

def F6(x):#[-100,100]
    dim=30
    lb=-100
    ub=100
    ss = 0
    for i in range(len(x)):
        ss+=abs(x[i])
    s = ss
    return s

def F7(x):
    dim=30
    lb=-32
    ub=32
    ss=numpy.sum(x**2)
    cc=0
    for i in range(len(x)):
        cc+=math.cos(2*math.pi*x[i])
    s=-20*math.exp(-0.2*math.sqrt(ss/len(x)))-math.exp(cc/len(x))+20+math.exp(1)
    return s

def F8(x):#[-100,100]
    dim=30
    lb=-100
    ub=100
    ss = numpy.sum(x**2/4000)+1
    c=1
    for i in range(1,len(x)+1):
        c*=math.cos(x[i-1]/math.sqrt(i))
    s = ss-c
    return s

def F9(x):#[-100,100]
    dim=30
    lb=-100
    ub=100
    ss = numpy.sum(x**2)
    s = 1-math.cos(2*math.pi*math.sqrt(ss))+0.1*math.sqrt(ss)
    return s

def F10(x):
    dim=30
    lb=-50
    ub=50
    a=5
    k=100
    m=4
    c=(math.sin(3*math.pi*x[0]))**2+((x[29]-1)**2)*(1+(math.sin(2*math.pi*x[29]))**2)
    for i in range(len(x)):
        c += ((x[i]-1)**2)*(1+(math.sin(3*math.pi*x[i])+1)**2)
    c=c*0.1
    for i in range(len(x)):
        if x[i]>a:
            c+=k*(x[i]-a)**m
        elif x[i]<-a:
            c+=k*(-x[i]-a)**m
    s=c
    return s

def F11(x):#[-1,1]
    dim=30
    lb=-5.12
    ub=5.12
    cc = 0
    for i in range(0, len(x)):
        cc += (x[i] ** 2-10*math.cos(2*math.pi*x[i])+10)
    s = cc
    return s

def F12(x):#[-1,1]
    dim = 30
    lb = -50
    ub = 50
    a = 10
    k = 100
    m = 4
    c = 10*(math.sin(math.pi * ((x[0]+1)/4+1)))**2 + ((x[29] + 1)/4) ** 2
    for i in range(len(x)-1):
        c += (((x[i] + 1)/4) ** 2)*(1+10*(math.sin(math.pi * ((x[i+1]+1)/4+1)))**2)
    c = c * (math.pi/30)
    for i in range(len(x)):
        if x[i] > a:
            c += k * (x[i] - a) ** m
        elif x[i] < -a:
            c += k * (-x[i] - a) ** m
    s = c
    return s

def F13(x):#[-5,10]
    dim = 30
    lb = -100
    ub = 100
    c=0
    for i in range(0, len(x)):
        c += (abs(x[i]+0.5))**2
    s=c
    return s

def F14(x):#[-30,30]
    dim = 30
    lb = -30
    ub = 30
    c=0
    for i in range(0, len(x)-1):
        c += (100*(x[i+1]-x[i]**2)**2+(x[i]-1)**2)
    s=c
    return s

def F15(x):#[-30,30]
    dim = 6
    lb = 0
    ub = 1
    A=[[10,3,17,3.5,1.7,8],
       [0.05,10,17,0.1,8,14],
       [3,3.5,1.7,10,17,8],
       [17,8,0.05,10,0.1,14]]
    C=[1,1.2,3,3.2]
    P=[[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5586],
       [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
       [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
       [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]]
    c=0
    for i in range(4):
        cc=0
        for j in range(6):
            cc+=A[i][j]*((x[j]-P[i][j])**2)
        c += C[i]*math.exp(-cc)
    s=-c+3.32236
    return s

def F16(x):#[-30,30]
    dim = 4
    lb = 0
    ub = 10
    A=[[4,4,4,4],
       [1,1,1,1],
       [8,8,8,8],
       [6,6,6,6],
       [3,7,3,7],
       [2,9,2,9],
       [5,5,3,3],
       [8,1,8,1],
       [6,2,6,2],
       [7,3.6,7,3.6]]
    C=[0.1,0.2,0.2,0.4,0.4,0.6,0.3,0.7,0.5,0.5]
    c=0
    for i in range(10):
        cc = 0
        for j in range(4):
            cc += (x[j]-A[i][j])**2
        c += 1/(C[i] + cc)
    s=-c+10.5319
    return s

def F17(x):#[-30,30]
    dim = 4
    lb = 0
    ub = 10
    A=[[4,4,4,4],
       [1,1,1,1],
       [8,8,8,8],
       [6,6,6,6],
       [3,7,3,7]]
    C=[0.1,0.2,0.2,0.4,0.4]
    c=0
    for i in range(5):
        cc = 0
        for j in range(4):
            cc += (x[j]-A[i][j])**2
        c += 1/(C[i] + cc)
    s=-c+10.1499
    return s

def F18(x):#[-30,30]
    dim = 2
    lb = -100
    ub = 100
    s=x[0]**2+2*(x[1]**2)-0.3*math.cos(3*math.pi*x[0]+4*math.pi*x[1])+0.3
    return s


def F19(x):#[-30,30]
    dim = 30
    lb = -10
    ub = 10
    c=0
    cc=0
    ccc=0
    for i in range(0, len(x)):
        c += math.sin(x[i])**2
        cc+=x[i]**2
        ccc+=math.sin(math.sqrt(abs(x[i])))**2
    s=(c-math.exp(-cc))*math.exp(-ccc)+1
    return s


tracemalloc.start()
#主程序

tn = 5
func_details = [[F1, -100, 100, 30],[F2, -100, 100, 30],[F3, -100, 100, 30],
                [F4, -100, 100, 30],[F5, -100, 100, 30],[F6, -100, 100, 30],
                [F7, -32, 32, 30],[F8, -100, 100, 30],[F9, -100, 100, 30],
                [F10, -50, 50, 30],[F11, -5.12, 5.12, 30],[F12, -50, 50, 30],
                [F13, -100, 100, 30],[F14, -30, 30, 30],[F15, 0, 1, 6],
                [F16, 0, 10, 4],[F17, 0, 10, 4],[F18, -100, 100, 2],
                [F19, -10, 10, 30]]
stla=[7.941861248,7.947866774,18.24994609,6.933228576,6.467637587,7.095196462,7.540277255,
7.661875534,6.720435369,12.41295357,9.060303926,14.94295622,8.294937444,11.2097746,
3.669719827,4.950158608,2.916625667,0.632087612,9.120521414]
orthogonal=[[0.99,2.5,0.1],[0.99,2,0.001],[0.99,1.5,0.00001],[0.99,1,0.01],[0.99,0.5,0.0001],
            [0.96,2.5,0.00001],[0.96,2,0.01],[0.96,1.5,0.0001],[0.96,1,0.1],[0.96,0.5,0.001],
            [0.93,2.5,0.0001],[0.93,2,0.1],[0.93,1.5,0.001],[0.93,1,0.00001],[0.93,0.5,0.01],
            [0.9,2.5,0.001],[0.9,2,0.00001],[0.9,1.5,0.01],[0.9,1,0.0001],[0.9,0.5,0.1],
            [0.87,2.5,0.01],[0.87,2,0.0001],[0.87,1.5,0.1],[0.87,1,0.001],[0.87,0.5,0.00001]]
for i in orthogonal:
    ARV=[]
    print(i)
    for j in range(tn):
        print(j)
        arv=[]
        for fun in range(len(func_details)):
            Fx = func_details[fun][0]
            # output_path = 'out1.txt'
            # with open(output_path, 'a', encoding='utf-8') as file1:
            #     print(f'F{fun + 1}',end=" ",file=file1)
            print(f'F{fun+1}')
            Max_iter = 300#迭代次数
            lb = func_details[fun][1]#下界-10000000000000000000000
            ub = func_details[fun][2]#上届10000000000000000000000
            dim = func_details[fun][3]#狼的寻值范围30
            SearchAgents_no = 50#寻值的狼的数量
            positions=init()

            # X=[]
            # for i in range(tn):
            # positions_1=positions.copy()
            positions_1=beta_init()
            k=i[0]
            Tmax=i[1]
            Tmin=i[2]
            begin_time = time()
            x = SMGWO(positions_1,Fx, lb, ub, dim, SearchAgents_no, Max_iter,k,Tmax,Tmin,stla[fun],begin_time)#改进灰狼
            # X.append(x[len(x)-1])
            end_time = time()
            run_time = end_time - begin_time
            # print(run_time)
            print(x[len(x)-1])
            arv.append(x[len(x)-1])
        ARV.append(arv)
    print(ARV)
    output_path = 'result of Calculating ARV.txt'
    with open(output_path, 'a', encoding='utf-8') as file1:
        print(ARV, file=file1)

out1=[
    [302, 325, 331, 303, 365, 288, 401, 421, 384, 399, 414, 380, 485, 527, 572],
    [298, 333, 322, 303, 366, 283, 386, 425, 388, 403, 459, 375, 497, 497, 596],
    [308, 328, 314, 296, 379, 258, 396, 428, 410, 395, 444, 365, 526, 505, 613],
    [293, 319, 324, 310, 343, 276, 384, 415, 391, 423, 434, 380, 493, 514, 593],
    [313, 330, 315, 313, 358, 279, 393, 408, 399, 407, 430, 373, 505, 482, 583],
    [307, 321, 307, 313, 329, 273, 377, 412, 372, 405, 438, 391, 487, 487, 578],
    [318, 299, 305, 313, 333, 269, 408, 412, 393, 401, 447, 387, 484, 478, 596],
    [292, 317, 318, 315, 331, 283, 385, 381, 374, 404, 435, 352, 495, 503, 598],
    [307, 338, 305, 300, 364, 280, 395, 418, 368, 397, 442, 395, 512, 501, 607],
    [299, 326, 321, 306, 337, 276, 396, 421, 395, 410, 436, 358, 501, 469, 599],
    [296, 340, 312, 319, 347, 275, 388, 427, 410, 395, 438, 376, 501, 507, 587],
    [326, 326, 324, 312, 358, 270, 395, 407, 395, 412, 433, 374, 506, 494, 594],
    [305, 321, 305, 317, 350, 271, 395, 439, 400, 424, 413, 378, 474, 485, 580],
    [300, 329, 317, 302, 361, 273, 385, 420, 410, 414, 426, 373, 495, 510, 608],
    [314, 309, 313, 314, 357, 280, 356, 423, 392, 394, 415, 373, 469, 516, 604],
    [311, 342, 312, 291, 347, 268, 396, 418, 386, 394, 423, 372, 512, 498, 601],
    [316, 338, 324, 317, 336, 271, 391, 409, 401, 398, 435, 368, 516, 510, 621],
    [295, 322, 309, 310, 363, 287, 405, 408, 408, 405, 425, 374, 492, 491, 618],
    [317, 330, 298, 312, 356, 271, 392, 402, 394, 404, 428, 368, 504, 482, 585],
    [307, 327, 312, 307, 366, 274, 412, 407, 374, 402, 464, 395, 509, 477, 627],
    [294, 335, 311, 305, 343, 269, 405, 401, 370, 398, 439, 386, 497, 478, 589],
    [295, 330, 315, 319, 362, 281, 408, 425, 399, 404, 418, 386, 496, 482, 597],
    [285, 317, 315, 317, 366, 275, 387, 412, 387, 406, 434, 387, 480, 501, 615],
    [297, 320, 300, 318, 369, 282, 399, 407, 398, 384, 438, 384, 498, 505, 583],
    [305, 329, 317, 314, 351, 267, 380, 403, 400, 398, 430, 384, 505, 490, 593],
    [295, 311, 298, 299, 340, 275, 385, 399, 360, 391, 427, 371, 466, 489, 587],
    [273, 323, 303, 303, 331, 276, 375, 409, 400, 379, 428, 378, 502, 508, 599],
    [293, 300, 314, 304, 349, 269, 386, 407, 372, 409, 395, 378, 484, 489, 589],
    [289, 315, 314, 309, 332, 269, 355, 391, 379, 397, 425, 375, 501, 484, 570],
    [302, 317, 311, 297, 340, 262, 370, 404, 383, 393, 430, 378, 488, 491, 558],
    [291, 310, 321, 296, 348, 259, 381, 409, 368, 397, 438, 350, 487, 482, 590],
    [294, 292, 324, 293, 357, 280, 382, 406, 371, 386, 434, 367, 472, 488, 586],
    [297, 310, 303, 305, 343, 278, 389, 396, 387, 383, 411, 365, 493, 484, 565],
    [293, 323, 306, 315, 341, 277, 385, 399, 386, 383, 422, 362, 489, 483, 593],
    [294, 321, 309, 294, 350, 258, 392, 403, 357, 379, 403, 377, 475, 474, 584],
    [298, 317, 321, 291, 360, 272, 396, 387, 381, 391, 410, 353, 476, 496, 594],
    [288, 304, 317, 303, 350, 272, 382, 409, 376, 391, 439, 377, 460, 490, 572],
    [307, 315, 292, 304, 354, 269, 383, 409, 367, 397, 425, 367, 502, 477, 553],
    [306, 321, 299, 298, 341, 270, 371, 407, 366, 392, 397, 371, 474, 493, 571],
    [291, 329, 320, 308, 329, 264, 389, 405, 381, 389, 422, 363, 479, 476, 567],
    [296, 301, 308, 310, 337, 271, 391, 405, 361, 395, 421, 361, 492, 492, 585],
    [285, 325, 318, 309, 352, 282, 402, 407, 384, 391, 408, 374, 483, 491, 602],
    [292, 329, 297, 297, 346, 265, 381, 397, 387, 383, 447, 373, 463, 497, 575],
    [300, 323, 300, 303, 348, 268, 377, 399, 377, 399, 434, 376, 481, 488, 567],
    [302, 331, 305, 286, 337, 270, 379, 409, 398, 391, 418, 375, 495, 488, 550],
    [308, 320, 312, 294, 343, 260, 376, 409, 381, 396, 395, 376, 475, 467, 592],
    [286, 332, 305, 304, 339, 266, 382, 389, 382, 372, 423, 374, 476, 477, 585],
    [295, 311, 319, 304, 339, 274, 392, 406, 387, 404, 414, 365, 493, 501, 557],
    [296, 311, 299, 293, 340, 263, 372, 404, 379, 382, 446, 355, 482, 497, 556],
    [302, 316, 303, 300, 336, 266, 378, 389, 386, 388, 424, 358, 466, 477, 571],
    [298, 298, 301, 300, 346, 266, 384, 408, 367, 385, 422, 369, 465, 473, 587],
    [284, 315, 309, 287, 346, 270, 367, 404, 389, 380, 426, 356, 480, 478, 570],
    [292, 308, 306, 282, 350, 269, 380, 404, 351, 387, 417, 352, 493, 460, 596],
    [296, 314, 291, 308, 341, 269, 390, 405, 367, 387, 419, 369, 463, 495, 574],
    [297, 306, 292, 296, 347, 266, 373, 395, 380, 397, 411, 362, 472, 482, 564],
    [298, 319, 309, 307, 335, 268, 367, 386, 383, 374, 424, 356, 461, 491, 570],
    [279, 321, 309, 284, 331, 254, 376, 405, 386, 376, 413, 360, 489, 464, 580],
    [304, 316, 304, 299, 349, 266, 388, 399, 382, 395, 420, 374, 488, 479, 581],
    [291, 299, 300, 302, 332, 267, 380, 398, 372, 377, 418, 370, 471, 474, 576],
    [293, 322, 302, 292, 339, 268, 380, 398, 383, 386, 428, 367, 468, 465, 571],
    [293, 324, 308, 294, 347, 272, 381, 401, 380, 386, 410, 373, 476, 483, 557],
    [278, 310, 312, 294, 356, 259, 375, 401, 380, 375, 420, 351, 482, 481, 578],
    [294, 321, 300, 301, 341, 264, 377, 392, 376, 379, 418, 367, 480, 481, 574],
    [289, 318, 304, 289, 350, 255, 365, 392, 359, 380, 417, 350, 458, 450, 554],
    [277, 327, 312, 290, 334, 271, 369, 410, 365, 391, 415, 364, 473, 466, 555],
    [292, 318, 305, 289, 343, 263, 387, 404, 378, 394, 405, 355, 465, 488, 570],
    [297, 309, 309, 298, 343, 259, 372, 397, 383, 373, 418, 369, 481, 478, 572],
    [304, 314, 310, 296, 339, 275, 377, 387, 376, 391, 399, 372, 491, 489, 587],
    [289, 314, 316, 300, 344, 259, 369, 406, 379, 381, 423, 362, 475, 478, 593],
    [295, 323, 309, 298, 340, 267, 382, 392, 380, 377, 423, 365, 478, 484, 578],
    [303, 298, 287, 300, 351, 266, 368, 410, 367, 391, 397, 363, 479, 484, 565],
    [295, 319, 299, 305, 352, 270, 376, 381, 384, 384, 404, 372, 463, 472, 582],
    [296, 310, 303, 303, 336, 269, 375, 394, 378, 369, 424, 366, 479, 479, 578],
    [291, 314, 298, 292, 332, 267, 384, 389, 378, 388, 418, 363, 472, 471, 584],
    [279, 324, 301, 296, 338, 268, 381, 407, 370, 398, 413, 375, 486, 489, 596],
    [283, 318, 301, 301, 339, 266, 358, 392, 364, 377, 416, 362, 470, 462, 561],
    [286, 305, 304, 294, 318, 262, 384, 395, 377, 386, 409, 359, 481, 473, 570],
    [293, 319, 301, 289, 326, 259, 373, 388, 370, 385, 410, 359, 449, 479, 563],
    [280, 304, 302, 284, 340, 263, 380, 401, 380, 384, 410, 358, 468, 482, 574],
    [290, 317, 314, 296, 343, 262, 390, 381, 367, 387, 397, 362, 461, 465, 567],
    [290, 311, 287, 305, 350, 273, 369, 392, 374, 383, 377, 366, 489, 473, 569],
    [289, 306, 301, 289, 341, 262, 368, 369, 368, 378, 410, 367, 462, 464, 567],
    [288, 319, 308, 296, 341, 265, 372, 404, 376, 397, 417, 354, 479, 471, 583],
    [285, 316, 296, 296, 334, 259, 384, 400, 376, 371, 398, 376, 473, 486, 560],
    [300, 308, 292, 299, 322, 271, 371, 411, 372, 377, 418, 358, 460, 476, 551],
    [288, 325, 303, 295, 348, 265, 381, 385, 371, 369, 421, 364, 463, 474, 562],
    [288, 312, 315, 286, 342, 261, 370, 412, 373, 389, 413, 358, 448, 468, 577],
    [289, 312, 307, 300, 334, 268, 368, 398, 361, 377, 422, 362, 479, 469, 561],
    [288, 301, 292, 303, 335, 264, 380, 381, 363, 381, 409, 365, 476, 484, 552],
    [291, 296, 307, 294, 341, 270, 376, 383, 373, 377, 414, 365, 482, 486, 573],
    [287, 312, 304, 294, 334, 264, 376, 366, 388, 371, 423, 350, 489, 479, 578],
    [288, 316, 310, 308, 329, 263, 374, 404, 357, 385, 405, 362, 475, 474, 564],
    [287, 317, 303, 301, 331, 260, 378, 393, 372, 384, 404, 355, 467, 475, 563],
    [291, 309, 306, 295, 333, 266, 378, 412, 367, 384, 400, 365, 456, 447, 555],
    [293, 315, 301, 288, 328, 264, 375, 401, 377, 391, 404, 371, 487, 477, 569],
    [283, 310, 267, 298, 337, 263, 378, 415, 373, 380, 429, 367, 467, 489, 556],
    [287, 319, 302, 306, 340, 264, 371, 372, 345, 380, 409, 360, 473, 483, 563],
    [293, 303, 308, 305, 347, 262, 365, 392, 376, 394, 413, 357, 469, 481, 569],
    [286, 327, 302, 293, 318, 254, 384, 408, 379, 378, 417, 364, 470, 493, 543],
    [288, 311, 319, 294, 335, 261, 371, 390, 369, 386, 425, 359, 476, 473, 540],
    [298, 310, 303, 294, 334, 265, 364, 379, 381, 376, 391, 369, 467, 472, 559],
    [298, 304, 307, 297, 321, 259, 374, 392, 377, 368, 419, 349, 472, 467, 549],
    [282, 312, 305, 288, 340, 260, 376, 392, 352, 376, 412, 353, 472, 465, 582],
    [285, 308, 295, 288, 345, 267, 361, 404, 366, 387, 407, 371, 473, 472, 563],
    [283, 310, 292, 297, 329, 257, 369, 407, 374, 383, 409, 360, 483, 467, 580],
    [281, 286, 308, 287, 329, 257, 381, 386, 368, 371, 416, 355, 464, 459, 571],
    [279, 308, 305, 286, 341, 257, 385, 406, 368, 383, 409, 370, 476, 476, 567],
    [300, 310, 302, 285, 354, 262, 368, 386, 376, 381, 416, 357, 460, 478, 566],
    [281, 314, 292, 290, 328, 264, 373, 398, 366, 383, 408, 356, 476, 479, 563],
    [280, 311, 305, 293, 339, 263, 374, 393, 359, 389, 406, 352, 459, 449, 567],
    [281, 306, 299, 293, 343, 268, 378, 387, 348, 370, 400, 367, 456, 471, 576],
    [285, 319, 314, 287, 333, 264, 364, 390, 382, 368, 381, 354, 471, 481, 560],
    [295, 310, 286, 293, 334, 255, 369, 401, 376, 377, 419, 353, 488, 469, 558],
    [285, 299, 307, 305, 337, 254, 367, 395, 374, 379, 404, 362, 445, 452, 555],
    [287, 296, 295, 290, 345, 264, 384, 398, 372, 375, 407, 357, 478, 475, 572],
    [285, 304, 293, 294, 346, 261, 365, 392, 380, 375, 401, 358, 454, 463, 571],
    [269, 321, 295, 294, 328, 259, 372, 379, 380, 383, 392, 358, 483, 475, 563],
    [284, 304, 297, 300, 332, 264, 371, 381, 371, 385, 408, 361, 477, 479, 578],
    [281, 316, 282, 274, 337, 253, 368, 372, 356, 389, 391, 367, 478, 459, 574],
    [282, 312, 317, 295, 333, 261, 361, 389, 359, 383, 405, 349, 495, 466, 553],
    [286, 300, 293, 298, 345, 262, 357, 391, 364, 388, 404, 368, 471, 481, 551],
    [289, 318, 304, 282, 340, 272, 368, 399, 378, 391, 413, 366, 473, 473, 572],
    [281, 319, 306, 290, 333, 262, 369, 399, 383, 375, 396, 349, 471, 470, 555],
    [290, 315, 310, 292, 343, 262, 387, 405, 358, 382, 407, 345, 461, 459, 573],
    [285, 300, 298, 304, 333, 262, 374, 400, 373, 384, 415, 357, 466, 481, 559],

]

ARV=[]
m=(np.array(out1)).min(0)
j=0
for i in range(25):
    s=0
    for ra in range(j,j+5):
        print(ra)
        for la in range(len(out1[ra])):
            s+=(out1[ra][la]-m[la])
    print(s)
    ARV.append(s/(5*15))
    j+=5
print(ARV)

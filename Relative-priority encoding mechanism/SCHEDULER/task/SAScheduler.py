#!/usr/bin/env python3
# *-* coding:utf8 *-*

"""
类别: 调度器
名称: 改进遗传算法调度器 -- 针对车间的调度
作者: 孙质方
邮件: zf_sun@vip.hnist.edu.cn
日期: 2022年4月7日
说明:
"""

import sys
import os
import math
from SCHEDULER.scheduler import Scheduler
from system.computingsystem import ComputingSystem
from UTIL.schedulerutils import SchedulerUtils
from COMPONENT.runningspan import RunningSpan
from COMPONENT.assignment import Assignment
from COMPONENT.sequence import Sequence
from COMPONENT.OR import ORnode
from COMPONENT.schedulinglist import SchedulingList
from UTIL.genericutils import *
from UTIL.logger import Logger
from CONFIG.config import *
from itertools import permutations, product
from datetime import datetime
from copy import *
import random
import numpy as np
from time import *


class SAIPPSScheduler(Scheduler):

    # sys.stdout = Logger('./result/result_%d.html' % (random.randint(1000, 9999)))
    sys.stdout = Logger('E:/pycpp/GABUDGET/result/result_task.html')

    def schedule(self, app,targetms):
        ComputingSystem.reset(app)  # 重置计算系统. !!!VERY IMPORTANT!!!

        tInitial = 1000.0  # 设定初始退火温度(initial temperature)
        T=tInitial
        tFinal = 1  # 设定终止退火温度(stop temperature)
        alfa = 0.98  # 设定降温参数，T(k)=alfa*T(k-1)

        pop_size = 100
        population = self.init_population(app, pop_size)
        k = 0
        gBest = pBest = population[0].makespan  # 全局最优值、当前最优值
        gLine = pLine = population  # 全局最优解、当前最优解
        best_ans=[]
        # print(len(population))
        tabu_list=[]
        while True:
            T*=alfa
            # print(k)
            half_population = []
            # half_population.extend(self.select(population))#每次选择种群里适应度值靠前的一半
            population = self.OptimizationSSA(app,[population[0].chromosome])#前后交叉
            # mutation_chromosomes = self.mutate(app,crossover_chromosomes)#个体变异
            # population = population[:len(population)//2]

            if population[0].makespan < gBest:    #概率判断
                gBest = population[0].makespan
                gLine=population
            if pBest>population[0].makespan or math.exp((pBest - population[0].makespan) / T) > random.random(): #热力学定理
                pBest = population[0].makespan
                pLine=population

            population=gLine


            # population.sort(key=lambda seq: seq.makespan)
            # pBest, pLine = population[0].makespan, population[0]  # 更新当前最优值、最优解
            # if pBest <= gBest:
            #     gBest, gLine = population[0].makespan, population[0]  # 更新全局最优值、最优解
            # population.sort(key=lambda seq: seq.min_sigma_load)
            # print("<br/>generation = %d, makespan = %.2f, cost = %.2f, time = %s" % (k, population[0].makespan, population[0].cost, datetime.now().strftime('%Y-%m-%d %H:%M:%S %f')))
            k = k + 1
            best_ans.append(gLine[0])
            if population[0].makespan<targetms:
                break

        # print("-" * 100)
        # print("<br/>pop_size = %d<br/>" % pop_size)
        elite_sequence = gLine[0]
        makespan = elite_sequence.makespan#调度序列完工时间
        min_sigma_load=elite_sequence.min_sigma_load
        print(elite_sequence.processor_load)
        complete_time={}
        for i in range(len(elite_sequence.tsk_sequence)):
            start_time=elite_sequence.temp_list[i].running_span.start_time
            finish_time=elite_sequence.temp_list[i].running_span.finish_time
            span=elite_sequence.temp_list[i].running_span.span
            id=(elite_sequence.tsk_sequence[i].name,elite_sequence.prossor_sequence[i].id, elite_sequence.tsk_sequence[i].id)
            complete_time[id]=(start_time,finish_time,span)
        print(complete_time)

        # cost = elite_sequence.cost#调度序列总成本
        # tradeoff = ALPHA * makespan + BETA * cost
        for i in range(len(elite_sequence.tsk_sequence)):
            print(elite_sequence.tsk_sequence[i].id,end=" ")
        print()
        for i in range(len(elite_sequence.prossor_sequence)):
            print(elite_sequence.prossor_sequence[i].id,end=" ")
        print()
        for i in range(len(best_ans)):
            print(best_ans[i].makespan,end=' ')
        print()
        print("The scheduler = %s, makespan = %.2f, min_sigma_load = %.2f" % (self.scheduler_name,makespan,min_sigma_load))

        return makespan,complete_time

    def init_population(self, app, pop_size):  # 两条工艺线路
        l = len(ComputingSystem.processors)
        chromosomes = []
        tasks = app.tasks
        OR = list(app.OR.items())
        # print(OR)
        temp1 = []
        for i in range(len(tasks)):
            temp1.append(i)
        # for i in range(len(tasks)-app.ORnum,len(tasks)-app.ORnum+len(OR)):
        #     temp1.append(i)
        # print(temp1)
        # print(temp)
        chromosome = []
        ant = 0  # 用来记录OR节点内被调度的任务数量
        for j in range(len(OR)):
            a = random.randint(0, 1)
            temp_or = ORnode(a)
            chromosome.append(temp_or)
            # print(f'{j}     {a}')
            ant += len(OR[j][a])

        # temp=temp1[0:len(tasks)-app.ORnum+ant]  #len(tasks)-app.ORnum+ant为所有被调度任务的个数
        temp = temp1[0:len(tasks)]
        for j in range(len(OR)):
            for k in OR[j][chromosome[j].id]:
                a = random.randint(0, len(temp) - 1)
                chromosome[j].lis.append(random.randint(temp[a] * l, (temp[a] + 1) * l - 1))
                temp.remove(temp[a])
            for k in OR[j][not chromosome[j].id]:
                a = random.randint(0, len(temp) - 1)
                chromosome[j]._lis.append(random.randint(temp[a] * l, (temp[a] + 1) * l - 1))
                temp.remove(temp[a])
        for j in range(len(tasks) - app.ORnum):
            # print(len(temp))
            a = random.randint(0, len(temp) - 1)
            # print(a)
            # print(len(temp))
            chromosome.append(random.randint(temp[a] * l, (temp[a] + 1) * l - 1))
            temp.remove(temp[a])

        chromosomes.append(chromosome)
        # if i==0:
        #     print(chromosome)

        population = self.create_population(app, chromosomes)  # 1000*len(tasks)
        # population.sort(key=lambda seq: seq.makespan)
        # population.sort(key=lambda seq: seq.min_sigma_load)
        return population

    def create_population(self, app, chromosomes):
        k = 0
        not_in_OR = app.not_in_OR
        OR = list(app.OR.items())  # 两条规划线路的OR字典存储方法
        # OR = app.OR  # 多条规划线路
        population = []
        candidate_tasks = []
        processor_set = ComputingSystem.processors  # 处理器
        while len(chromosomes) > 0:
            ComputingSystem.reset(app)  # 重置计算系统. !!!VERY IMPORTANT!!!

            candidate_tasks.clear()
            tsk_sequence = []
            prossor_sequence = []
            # candidate_tasks+=app.IPPS_entry_task#添加入口任务
            entry_task_id_list = []  # 入口任务编号
            for i in app.IPPS_entry_task:
                entry_task_id_list.append(i.id)
                if i.id in not_in_OR:
                    candidate_tasks.append(i)
            ctm = app.ctm
            computation_time_matrix = app.computation_time_matrix
            chromosome1 = chromosomes.pop(0)  # 取出chromosomes种群中第一个个体
            chromosome = []  # 保存优先级
            chromosome_dex = []  # 当前OR情况下可调度的所有任务编号
            for i in range(len(OR)):
                if len(app.OR_in) > 0 and not self.judge_OR(i + 1, app.OR_in, chromosome1):
                    continue
                # 还需判断压入的OR节点的任务是否为入口任务
                for j in OR[i][chromosome1[i].id]:
                    chromosome_dex.append(j)
                    app.tasks[j - 1].is_decoded = False
                    if j in entry_task_id_list:
                        candidate_tasks.append(app.tasks[j - 1])
                # if chromosome1[i].id==0:
                #     for j in OR[i][1]:
                #         app.tasks[j-1].is_decoded=True
                # else:
                #     for j in OR[i][0]:
                #         app.tasks[j-1].is_decoded=True

                chromosome += chromosome1[i].lis  # 两条规划线路的操作优先级
                # print("@@@")
                # print(chromosome1[i].id)
                # print(chromosome1[i].lis)
                # chromosome+=chromosome1[i].lis[chromosome1[i].id]   #多条规划线路的操作优先级
            chromosome += chromosome1[len(OR):]
            chromosome_dex += not_in_OR
            # self.reset_tasks(app.tasks,not_in_OR)
            self.reset_tasks(app.tasks, chromosome_dex)
            # for i in not_in_OR:
            #     app.tasks[i-1].is_decoded=False

            """RPGA处理器选择"""
            # print(len(chromosome))
            # chromosome_dex=sorted(chromosome_dex)
            # for i in chromosome_dex:
            #     print(app.tasks[i-1].name)
            # for j in range(0, len(chromosome)):#chromosome整除2
            #     candidate_tasks.sort(key=lambda tsk: tsk.id)
            #     size = len(candidate_tasks)  #入度为0的任务个数
            #     tsk_index,prossor_index = self.get_index(chromosome, chromosome_dex, candidate_tasks, size, ctm)#???
            #     # for p in candidate_tasks:
            #     #     print(p.name, end=' ')
            #     # print()
            #     # print(f'{candidate_tasks[tsk_index].name}!!!!!')
            #     task = candidate_tasks[tsk_index]
            #     task.is_decoded = True
            #     tsk_sequence.append(task)
            #     candidate_tasks.remove(task)
            #     # for p in candidate_tasks:
            #     #     print(p.id,end=' ')
            #     # print()
            #     for successor in task.successors:
            #         if (successor.id in chromosome_dex) and self.is_ready(successor):
            #             candidate_tasks.append(successor)
            #     # print(task.id)
            #     # print(computation_time_matrix[task.id-1])
            #     # print(prossor_index)
            #     processor = processor_set[list(computation_time_matrix[task.id-1].keys())[prossor_index]-1]
            #     prossor_sequence.append(processor)
            # makespan, temp_list, processor_load = self.calculate_response_time_and_cost(app, chromosome_dex, k, tsk_sequence, prossor_sequence)

            """HEFT处理器选择法"""
            temp_list = []
            processor_load = {}
            scheduling_list = self.scheduling_lists.setdefault(k)  # 判断字典scheduling_lists里是否有键counter，没有自动添加
            # print(list(idd.id for idd in app.entry_task))
            if not scheduling_list:
                scheduling_list = SchedulingList("Scheduling_List_%d" % k)
            for j in range(0, len(chromosome)):  # 遍历排序
                candidate_tasks.sort(key=lambda tsk: tsk.id)
                size = len(candidate_tasks)  # 入度为0的任务个数
                # print(size)
                tsk_index, prossor_index = self.get_index(chromosome, chromosome_dex, candidate_tasks, size, ctm)  # ???
                # print(tsk_index, prossor_index)
                # print(len(chromosome))
                # print(len(chromosome_dex))
                task = candidate_tasks[tsk_index]
                task.is_decoded = True
                tsk_sequence.append(task)
                candidate_tasks.remove(task)
                for successor in task.successors:
                    if (successor.id in chromosome_dex) and self.is_ready(successor):
                        candidate_tasks.append(successor)

                # print(task.name)
                earliest_start_time = 0.0  # 初始化全局最早启动时间为0
                earliest_finish_time = float("inf")  # 初始化全局最早结束时间为无穷大
                for p1 in list(computation_time_matrix[task.id - 1].keys()):  # 遍历处理器集
                    # print(list(computation_time_matrix[task.id-1].keys()))
                    p = processor_set[p1 - 1]
                    earliest_start_time_of_this_processor = SchedulerUtils.IPPS_calculate_earliest_start_time(task, p,
                                                                                                              chromosome_dex)  # 当前遍历处理器上最早可用启动时间
                    earliest_finish_time_of_this_processor = earliest_start_time_of_this_processor + \
                                                             task.processor__computation_time[p]  # 当前遍历处理器上最早可用结束时间
                    # print('!!!')
                    if earliest_finish_time > earliest_finish_time_of_this_processor:  # 如果全局最早启动时间大于当前遍历处理器上最早可用启动时间, 则
                        earliest_start_time = earliest_start_time_of_this_processor  # 设置全局最早启动时间为当前遍历处理器上最早可用启动时间
                        earliest_finish_time = earliest_finish_time_of_this_processor  # 设置全局最早结束时间为当前遍历处理器上最早可用结束时间
                        processor = p  # 设置全局处理器为当前遍历处理器
                prossor_sequence.append(processor)
                running_span = RunningSpan(earliest_start_time,
                                           earliest_finish_time)  # 上述 for 循环结束后, 最合适的处理器已被找出, 此时可以记录下任务的运行时间段
                assignment = Assignment(processor, running_span)  # 同时记录下任务的运行时环境
                temp_list.append(assignment)

                if processor.id not in processor_load.keys():
                    processor_load[processor.id] = running_span.span
                else:
                    processor_load[processor.id] += running_span.span

                task.assignment = assignment  # 设置任务的运行时环境

                task.is_assigned = True  # 标记任务已被分配

                processor.resident_tasks.append(task)  # 将任务添加至处理器的驻留任务集中
                processor.resident_tasks.sort(
                    key=lambda tsk: tsk.assignment.running_span.start_time)  # 对处理器的驻留任务进行排序, 依据任务启动时间升序排列
                scheduling_list.list[task] = assignment  # 将任务与对应运行时环境置于原始调度列表
            makespan = calculate_makespan(scheduling_list)

            # print(processor_load)
            k = k + 1
            min_sigma_load = 0
            _load = np.mean(list(processor_load.values()))
            for pro in processor_load.values():
                min_sigma_load += abs(pro - _load)

            s = Sequence(chromosome1, tsk_sequence, prossor_sequence, makespan)
            s.temp_list = temp_list
            s.min_sigma_load = min_sigma_load
            s.processor_load = processor_load
            population.append(s)

        return population

    def reset_tasks(self, tasks,chromosome_dex):
        for task in tasks:
            if task.id in chromosome_dex:
                task.is_decoded = False
            else:
                task.is_decoded = True


    def reset_tasks_init(self, tasks):
        for task in tasks:
            task.is_decoded = False

    def is_ready(self, task):
        for predecessor in task.predecessors:
            if not predecessor.is_decoded:
                return False
        return True

    def get_index(self, chromosome, chromosome_dex, candidate_tasks, size, ctm):#gene个体的基因，scale=1/size
        m=INF
        task_dex=0
        p_dex=0
        for i in range(size):
            if chromosome[chromosome_dex.index(candidate_tasks[i].id)]<m:
                m=chromosome[chromosome_dex.index(candidate_tasks[i].id)]
                task_dex=i
                p_dex=chromosome[chromosome_dex.index(candidate_tasks[i].id)]%ctm[candidate_tasks[i].id-1]
        return task_dex,p_dex


    def calculate_response_time_and_cost(self, app, chromosome_dex, counter, task_sequence, processor_sequence):
        ComputingSystem.reset(app)  # 重置计算系统. !!!VERY IMPORTANT!!!

        scheduling_list = self.scheduling_lists.setdefault(counter)#判断字典scheduling_lists里是否有键counter，没有自动添加

        if not scheduling_list:
            scheduling_list = SchedulingList("Scheduling_List_%d" % counter)
            self.scheduling_lists[counter] = scheduling_list

        temp_list=[]
        sp=INF
        processor_load={}
        for i in range(0, len(task_sequence)):  # 遍历当前消息分组内的所有消息
            task = task_sequence[i]  # 取任务 task
            processor = processor_sequence[i]  # 取任务 task 对应的运行处理器 processor

            start_time = SchedulerUtils.IPPS_calculate_earliest_start_time(task, processor, chromosome_dex)  # 当前遍历处理器上最早可用启动时间
            finish_time = start_time + task.processor__computation_time[processor]  # 当前遍历处理器上最早可用结束时间

            running_span = RunningSpan(start_time, finish_time)  # 上述 for 循环结束后, 最合适的处理器已被找出, 此时可以记录下任务的运行时间段
            assignment = Assignment(processor, running_span)  # 同时记录下任务的运行时环境
            temp_list.append(assignment)

            if processor.id not in processor_load.keys():
                processor_load[processor.id]=running_span.span
            else:
                processor_load[processor.id]+=running_span.span

            task.assignment = assignment  # 设置任务的运行时环境

            task.is_assigned = True  # 标记任务已被分配

            processor.resident_tasks.append(task)  # 将任务添加至处理器的驻留任务集中
            processor.resident_tasks.sort(key=lambda tsk: tsk.assignment.running_span.start_time)  # 对处理器的驻留任务进行排序, 依据任务启动时间升序排列

            self.scheduling_lists[counter].list[task] = assignment  # 将任务与对应运行时环境置于原始调度列表

        makespan = calculate_makespan(self.scheduling_lists[counter])
        # cost = calculate_cost(self.scheduling_lists[counter])
        self.scheduling_lists[counter].makespan = makespan  # 计算原始调度列表的完工时间
        # self.scheduling_lists[counter].cost = cost

        # print("The scheduler = %s, list_name = %s, makespan = %.2f" % (self.scheduler_name,
        #                                                                self.scheduling_lists[counter].list_name,
        #                                                                self.scheduling_lists[counter].makespan))

        # if SHOW_ORIGINAL_SCHEDULING_LIST:  # 如果打印原始调度列表, 则:
        #     print_scheduling_list(self.scheduling_lists[counter])  # 打印原始调度列表

        # if makespan < 100.0:
        #     for task in self.scheduling_lists[counter].list.keys():
        #         info = "%s\t%s\t%s" % (task.name, task.assignment.assigned_processor.name, task.assignment.running_span)
        #         print(info)
        #     print("-" * 100)
        #     print("The scheduler = %s, list_name = %s, makespan = %.2f<br/>" % (self.scheduler_name, self.scheduling_lists[counter].list_name, self.scheduling_lists[counter].makespan))
        #     print("#" * 100)
        # else:
        #     print("The scheduler = %s, list_name = %s, makespan = %.2f<br/>" % (self.scheduler_name, self.scheduling_lists[counter].list_name, self.scheduling_lists[counter].makespan))

        self.scheduling_lists.clear()
        return makespan,temp_list,processor_load

    def judge_OR(self, x, OR_in, b):
        if x not in list(OR_in.keys()):
            return True
        if b[OR_in[x][0] - 1].id == OR_in[x][1]:
            if OR_in[x][0] in list(OR_in.keys()):
                ans = self.judge_OR(OR_in[x][0], OR_in, b)
            else:
                return True
        else:
            return False
        return ans



    def OptimizationSSA(self, app, population):
        # print(population)
        ORnum = len(app.OR)
        # OR = app.OR  #多工艺线路
        OR = list(app.OR.items())    #两条线路
        # print(OR)
        not_in_or=app.not_in_OR
        neighborhood_list = []
        for p in population:
            sign1 = 0
            sign2 = 0
            pos1 = random.randint(0, len(p) - 1)
            # while pos1<ORnum and len(p[pos1].lis)==0:
            #     pos1 = random.randint(0, len(p) - 1)
            pos2 = random.randint(0, len(p) - 1)
            # print(pos1)
            # while (pos2>=ORnum and pos1>=ORnum and pos1==pos2) or (pos2<=ORnum and pos1<=ORnum and pos1==pos2 and len(p[pos1].lis[p[pos1].id])==1):   #多条工艺线路
            while (pos2 >= ORnum and pos1 >= ORnum and pos1 == pos2) or (
                    pos2 <= ORnum and pos1 <= ORnum and pos1 == pos2 and len(p[pos1].lis) == 1):  # 两条工艺线路
                pos2 = random.randint(0, len(p) - 1)
            # print(ORnum)

            if pos1 < ORnum:
                sign1 = 1
                # pos1_1=random.randint(0, len(p[pos1].lis[p[pos1].id])-1)   #多条工艺线路
                pos1_1 = random.randint(0, len(p[pos1].lis) - 1)  # 两条工艺线路
            if pos2 < ORnum:
                sign2 = 1
                # pos2_2=random.randint(0, len(p[pos2].lis[p[pos2].id])-1)   #多条工艺线路
                pos2_2 = random.randint(0, len(p[pos2].lis) - 1)  # 两条工艺线路
                while pos2 == pos1 and pos2_2 == pos1_1:
                    # print(f'{pos1} {pos2}....')
                    # print(f'{pos1_1} {pos2_2}')
                    # pos2_2 = random.randint(0, len(p[pos2].lis[p[pos2].id]) - 1)   #多条工艺线路
                    pos2_2 = random.randint(0, len(p[pos2].lis) - 1)  # 两条工艺线路
            # print(f'{sign1}   {sign2}')

            if sign1 == 0 and sign2 == 0:
                flag=[not_in_or[pos1-ORnum],not_in_or[pos2-ORnum]]
                p[pos1], p[pos2] = p[pos2], p[pos1]
            elif sign1 == 1 and sign2 == 0:
                # print(pos1,p[pos1].id,pos1_1,pos2)
                # print(OR[pos1])
                flag=[OR[pos1][p[pos1].id][pos1_1],not_in_or[pos2-ORnum]]
                # p[pos1].lis[p[pos1].id][pos1_1], p[pos2] = p[pos2], p[pos1].lis[p[pos1].id][pos1_1]   #多条工艺线路
                p[pos1].lis[pos1_1], p[pos2] = p[pos2], p[pos1].lis[pos1_1]  # 两条工艺线路
            elif sign1 == 0 and sign2 == 1:
                # print(pos1,pos2,p[pos2].id,pos2_2)
                # print(OR[pos2])
                flag=[not_in_or[pos1-ORnum],OR[pos2][p[pos2].id][pos2_2]]
                # p[pos1], p[pos2].lis[p[pos2].id][pos2_2] = p[pos2].lis[p[pos2].id][pos2_2], p[pos1]   #多条工艺线路
                p[pos1], p[pos2].lis[pos2_2] = p[pos2].lis[pos2_2], p[pos1]  # 多条工艺线路
            else:
                flag=[OR[pos1][p[pos1].id][pos1_1],OR[pos2][p[pos2].id][pos2_2]]
                # p[pos1].lis[p[pos1].id][pos1_1], p[pos2].lis[p[pos2].id][pos2_2] = p[pos2].lis[p[pos2].id][pos2_2], p[pos1].lis[p[pos1].id][pos1_1]   #多条工艺线路
                p[pos1].lis[pos1_1], p[pos2].lis[pos2_2] = p[pos2].lis[pos2_2], p[pos1].lis[pos1_1]  # 多条工艺线路

            """or变异"""
            a = random.randint(0, ORnum - 1)

            # 多条工艺线路线路
            # p[a].id=random.randint(0,len(OR[a])-1)

            # 两条工艺线路线路
            p[a].id = not p[a].id
            p[a].lis, p[a]._lis = p[a]._lis, p[a].lis

            population = self.create_population(app, [p])

        # print(neighborhood_list)


        return population



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


class GWOIPPSScheduler(Scheduler):

    # sys.stdout = Logger('./result/result_%d.html' % (random.randint(1000, 9999)))
    sys.stdout = Logger('E:/pycpp/GABUDGET/result/result_task.html')

    def schedule(self, app):
        ComputingSystem.reset(app)  # 重置计算系统. !!!VERY IMPORTANT!!!

        w1 = 1/3
        w2 = 1/3
        w3 = 1/3
        pop_size = 50
        population = self.init_population(app, pop_size)
        k = 0
        alpha = population[0] # 全局最优值、当前最优值
        beta = population[1]
        detla = population[2]
        best_ans=[]
        # print(len(population))
        while k < 400:
            # print(k)
            half_population = []
            # half_population.extend(self.select(population))#每次选择种群里适应度值靠前的一半
            crossover_chromosomes = self.crossover(app,population, alpha, beta, detla, w1, w2, w3)#前后交叉
            # mutation_chromosomes = self.mutate(app,crossover_chromosomes)#个体变异
            # population = population[:len(population)//2]
            population=self.create_population(app, crossover_chromosomes)
            population.sort(key=lambda seq: seq.makespan)
            alpha = population[0]
            beta = population[1]
            detla = population[2]  # 更新当前最优值、最优解
            # if pBest <= gBest:
            #     gBest, gLine = population[0].makespan, population[0]  # 更新全局最优值、最优解
            # population.sort(key=lambda seq: seq.min_sigma_load)
            # print("<br/>generation = %d, makespan = %.2f, cost = %.2f, time = %s" % (k, population[0].makespan, population[0].cost, datetime.now().strftime('%Y-%m-%d %H:%M:%S %f')))
            k = k + 1
            best_ans.append(population[0])

        # print("-" * 100)
        # print("<br/>pop_size = %d<br/>" % pop_size)
        elite_sequence = population[0]
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


    def init_population(self, app, pop_size):
        processors = ComputingSystem.processors
        chromosomes = []
        OR=list(app.OR.items())
        computation_time_matrix=app.computation_time_matrix
        for i in range(0, pop_size):  #种群数量
            ComputingSystem.reset(app)  # 重置计算系统. !!!VERY IMPORTANT!!!
            self.reset_tasks_init(app.tasks)
            candidate_tasks=app.IPPS_entry_task.copy()
            chromosome = []
            for j in range(len(OR)):
                a=random.randint(1,2)
                chromosome.append(a)
            for j in range(len(app.tasks)):
                a=random.randint(0,len(candidate_tasks)-1)
                chromosome.append(candidate_tasks[a])
                candidate_tasks[a].is_decoded = True
                for successor in candidate_tasks[a].successors:
                    if self.is_ready(successor):
                        candidate_tasks.append(successor)
                candidate_tasks.remove(candidate_tasks[a])
            for j in range(len(app.tasks)):
                a=random.randint(0,len(computation_time_matrix[j])-1)
                chromosome.append(processors[list(computation_time_matrix[j].keys())[a]-1])
            chromosomes.append(chromosome)

        # print(chromosomes)
        population = self.create_population(app, chromosomes)#1000*len(tasks)
        population.sort(key=lambda seq: seq.makespan)
        return population

    def create_population(self, app, chromosomes):
        k = 0
        not_in_OR=app.not_in_OR
        OR = list(app.OR.items())
        population = []
        candidate_tasks = []
        processor_set = ComputingSystem.processors#处理器
        while len(chromosomes) > 0:
            ComputingSystem.reset(app)  # 重置计算系统. !!!VERY IMPORTANT!!!

            candidate_tasks.clear()
            tsk_sequence = []
            prossor_sequence = []
            candidate_tasks+=app.IPPS_entry_task#添加入口任务

            ctm=app.ctm
            computation_time_matrix=app.computation_time_matrix
            chromosome1 = chromosomes.pop(0)#取出chromosomes种群中第一个个体
            or_chromosome=chromosome1[:len(OR)]
            op_chromosome=chromosome1[len(OR):len(OR)+len(app.tasks)]
            ma_chromosome=chromosome1[len(OR)+len(app.tasks):]
            continue_op=[]
            chromosome_dex=list(range(1,len(app.tasks)+1))
            for i in range(len(or_chromosome)):
                if not self.judge_OR(i+1,app.OR_in,or_chromosome):
                    continue_op += OR[i][0]
                    continue_op += OR[i][1]
                    continue
                if or_chromosome[i]==1:
                    continue_op+=OR[i][1]
                else:
                    continue_op += OR[i][0]

            for i in op_chromosome:
                if i.id in continue_op:
                    continue
                else:
                    tsk_sequence.append(i)
                    prossor_sequence.append(ma_chromosome[i.id-1])
            for i in continue_op:
                chromosome_dex.remove(i)
            # self.reset_tasks(app.tasks,not_in_OR)
            # self.reset_tasks(app.tasks, chromosome_dex)
            # for i in not_in_OR:
            #     app.tasks[i-1].is_decoded=False

            # print(or_chromosome)
            # print(f'{len(tsk_sequence)}   {tsk_sequence}')
            # print(f'{len(prossor_sequence)}   {prossor_sequence}')
            makespan, temp_list, processor_load = self.calculate_response_time_and_cost(app, chromosome_dex, k, tsk_sequence, prossor_sequence)



            # print(processor_load)
            k = k + 1
            min_sigma_load=0
            _load=np.mean(list(processor_load.values()))
            for pro in processor_load.values():
                min_sigma_load+=abs(pro-_load)

            s = Sequence( chromosome1, tsk_sequence, prossor_sequence, makespan)
            s.temp_list=temp_list
            s.min_sigma_load=min_sigma_load
            s.processor_load=processor_load
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

    def judge_OR(self,x,OR_in,b):
        if x not in list(OR_in.keys()):
            return True
        if b[OR_in[x][0]-1]==OR_in[x][1]+1:
            if OR_in[x][0] in list(OR_in.keys()):
                ans=self.judge_OR(OR_in[x][0],OR_in,b)
            else:
                return True
        else:
            return False
        return ans



    def crossover(self, app, population, alpha, beta, detla, w1, w2, w3):
        offspring_population = []
        ORnum=len(app.OR)
        for i in range(0, len(population)):

            parent1 = population[i]  # 选择parent1
            # 选择parent2（轮盘赌操作）
            randNum = random.uniform(0, sum([w1, w2, w3]))
            if randNum <= w1:
                parent2 = population[0]
            elif randNum <= w1 + w2:
                parent2 = population[1]
            else:
                parent2 = population[2]

            or_chromosome1 = []
            or_chromosome2 = []
            prev_chromosome = []
            next_chromosome = []
            ma_chromosome1 = []
            ma_chromosome2 = []
            or_chromosome1.extend(parent1.chromosome[:ORnum])
            or_chromosome2.extend(parent2.chromosome[:ORnum])
            prev_chromosome.extend(parent1.chromosome[ORnum:ORnum + len(app.tasks)])
            next_chromosome.extend(parent2.chromosome[ORnum:ORnum + len(app.tasks)])
            ma_chromosome1.extend(parent1.chromosome[ORnum + len(app.tasks):])
            ma_chromosome2.extend(parent2.chromosome[ORnum + len(app.tasks):])

            """OR交叉"""
            a=random.randint(0,len(or_chromosome1)-1)
            for j in range(a):
                or_chromosome1[j],or_chromosome2[j] = or_chromosome2[j],or_chromosome1[j]

            """操作交叉"""
            O1=[]
            O2=[]
            next_chromosome_to_O1=[]
            prev_chromosome_to_O2=[]
            crossover_point1 = random.randint(1, len(app.tasks)-1)
            crossover_point2 = random.randint(1, len(app.tasks)-1)
            while crossover_point1==crossover_point2:
                crossover_point2 = random.randint(1, len(app.tasks) - 1)
            if crossover_point2<crossover_point1:
                crossover_point2 , crossover_point1=crossover_point1,crossover_point2

            P1=[]
            P1 += prev_chromosome[:crossover_point1]
            P1 += prev_chromosome[crossover_point2:]

            P2 = []
            P2 += next_chromosome[:crossover_point1]
            P2 += next_chromosome[crossover_point2:]
            for j in range(len(prev_chromosome)):
                if next_chromosome[j] not in P1:
                    next_chromosome_to_O1.append(next_chromosome[j])
                if prev_chromosome[j] not in P2:
                    prev_chromosome_to_O2.append(prev_chromosome[j])

            O1+=prev_chromosome[:crossover_point1]
            O1+=next_chromosome_to_O1
            O1+=prev_chromosome[crossover_point2:]

            O2 += next_chromosome[:crossover_point1]
            O2 += prev_chromosome_to_O2
            O2 += next_chromosome[crossover_point2:]

            """机器交叉"""
            crossover_point1 = random.randint(1, len(app.tasks) - 1)
            crossover_point2 = random.randint(1, len(app.tasks) - 1)
            while crossover_point1 == crossover_point2:
                crossover_point2 = random.randint(1, len(app.tasks) - 1)
            if crossover_point2 < crossover_point1:
                crossover_point2, crossover_point1 = crossover_point1, crossover_point2
            for j in range(crossover_point1,crossover_point2):
                ma_chromosome1[j], ma_chromosome2[j] = ma_chromosome2[j], ma_chromosome1[j]

            merge_chromosome1=[]
            merge_chromosome1+=or_chromosome1
            merge_chromosome1 += O1
            merge_chromosome1 += ma_chromosome1

            # merge_chromosome2 = []
            # merge_chromosome2 += or_chromosome2
            # merge_chromosome2 += O2
            # merge_chromosome2 += ma_chromosome2
            offspring_population.append(merge_chromosome1)
            # offspring_population.append(merge_chromosome2)



        return offspring_population



#!/usr/bin/env python3
# *-* coding:utf8 *-*

import sys
import os

# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
from time import *
import tracemalloc
import plotly as py
import plotly.figure_factory as ff

from CONFIG.config import *
from UTIL.logger import Logger
from UTIL.genericutils import *
from COMPONENT.application import Application
from service.appIPPS import ApplicationService
from system.computingsystem import ComputingSystem
from UTIL.genericutils import print_list
from SCHEDULER.task.heftscheduler import HeftScheduler
from SCHEDULER.task.cpopscheduler import CpopScheduler
from SCHEDULER.task.IPPSScheduler import GeneticScheduler
from SCHEDULER.task.MGA import MGAScheduler
from SCHEDULER.task.PSOIPPS import PSOIPPSScheduler
from SCHEDULER.task.PSOIPPS_new import newPSOIPPSScheduler
from SCHEDULER.task.GWOIPPS import GWOIPPSScheduler
from SCHEDULER.task.GWOIPPS_NEW import newGWOIPPSScheduler
from SCHEDULER.task.TabuScheduler import tabuIPPSScheduler
from SCHEDULER.task.SAScheduler import SAIPPSScheduler
from SCHEDULER.task.Evolutionscheduler import EvolutionScheduler
from SCHEDULER.task.wolfscheduler import wolfScheduler
from SCHEDULER.task.ASDE_GWO import ASDE_GWOcheduler
from SCHEDULER.task.poshscheduler import POSHScheduler
from datetime import datetime


def main():
    # 处理器数
    processor_number = 15
    # 计算系统初始化
    ComputingSystem.init(processor_number)
    # 生成应用
    appA = Application("A")
    appB = Application("B")
    appC = Application("C")
    appD = Application("D")


    """job1"""
    computation_time_matrix1 = [
        {9: 13, 14: 10}, {11: 24, 15: 18}, {15: 43}, {12: 43}, {13: 30},
        {4: 32, 12: 25}, {1: 40, 5: 49, 11: 39}, {8: 47}
    ]
    communication_time_matrix1 = [
        [INF, 0, INF, INF, INF, INF, INF, INF],
        [INF, INF, 0, INF, 0, INF, INF, INF],
        [INF, INF, INF, 0, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, 0, INF],
        [INF, INF, INF, INF, INF, 0, INF, INF],
        [INF, INF, INF, INF, INF, INF, 0, INF],
        [INF, INF, INF, INF, INF, INF, INF, 0],
        [INF, INF, INF, INF, INF, INF, INF, INF]
    ]
    not_in_OR1 = [1, 2, 3, 4, 5, 6, 7, 8]
    OR1 = {}
    OR_in1 = {}

    """job2"""
    computation_time_matrix2 = [
        {5: 10, 8: 16, 14:13}, {8: 6, 9:8, 15: 7}, {4: 40}, {6:14, 9:10, 7:20, 12:13}, {1:33, 7:40, 11: 43},
        {1: 42, 5: 38}, {6: 25, 11: 33, 15: 30}, {10: 41, 15:44}, {2:10, 13:12}, {11:34,14:24,15:30},
        {6:38,11:42}, {4:25,8:26,12:30}, {7:39}, {10:37,12:40}
    ]
    communication_time_matrix2 = [
        [INF, 0, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, 0, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF]
    ]
    not_in_OR2 = [1, 2, 3, 4, 5, 6, 7, 8, 14]
    OR2 = {(9,10,11):(12,13)}
    OR_in2 = {}

    """job3"""
    computation_time_matrix3 = [
        {4: 29, 7: 36, 11: 34}, {2:35, 8: 29, 11: 27, 12: 30, 15: 33}, {1:11, 6: 9, 7: 8, 10: 19, 14: 12},
        {2: 18, 3: 20, 6: 27, 7: 13}, {2:19, 3: 24, 4: 22, 5: 31, 7: 37}, {1:13, 7: 8, 10: 12, 12: 9, 13: 5},
        {1: 50, 5: 39, 6: 44, 13: 48}, {1: 6, 12: 9}, {3: 44, 4: 36, 5: 30, 13: 39, 14:33},
        {3: 39, 4: 45, 7: 41, 12: 50, 13:40}, {2: 29, 4: 36, 9: 33}, {1: 19, 4: 20, 9: 17, 12: 16, 14:21},
        {9: 40, 14: 33, 15:35}, {2: 11, 4: 12, 6: 14, 7:15}, {1: 10, 8: 19, 12: 20,13: 17, 14: 16}, {10: 49, 15: 44},
        {10: 20, 12: 33, 13: 39}, {6: 30, 7: 29, 9: 40, 10: 39, 13:33}, {3: 20, 8: 29, 10: 40, 14: 34, 15: 31}
    ]
    communication_time_matrix3 = [
        [INF,   0, INF, INF,   0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF,   0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF,   0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF,   0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF,   0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF,   0, INF, INF, INF, INF,   0, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF,   0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF,   0, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   0, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   0, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   0, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   0, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   0, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   0, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   0, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   0],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF]
    ]
    not_in_OR3 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    OR3 = {}
    OR_in3 = {}

    """job4"""
    computation_time_matrix4 = [
        {2: 18}, {8: 38, 12: 30}, {9: 20}, {5: 8, 11: 9}, {3: 29}, {2: 36, 4: 33, 5: 39},
        {1: 23, 10: 20}, {6: 45}, {3: 5, 12: 9}, {5: 39, 9: 33}, {7: 36, 11: 41}, {14: 31, 15: 29},
        {1: 28, 2: 22, 6: 21}, {10: 18, 14: 28}, {12: 24}, {7: 23, 9: 25}
    ]
    communication_time_matrix4 = [
        [INF, 0, INF, INF,   0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF,   0, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF,   0, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0,   0, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF]
    ]
    not_in_OR4 = [1, 7, 8, 9, 10, 11, 12, 13, 16]
    OR4 = {(2, 3, 4): (5, 6), (14,): (15,)}
    OR_in4 = {}

    """job5"""
    computation_time_matrix5 = [
        {3: 35, 4: 29, 5: 36, 11: 31}, {1: 40, 3: 34, 5: 44, 10: 41, 15: 39}, {11: 15, 15: 13}, {2: 31, 6: 33, 7: 29, 10: 27, 14: 25},
        {6: 13, 12: 9, 13: 8, 14: 14}, {11: 28, 12: 29}, {2: 31, 4: 24, 10: 28, 12: 26, 14: 32}, {6: 34, 10: 33, 11: 30, 13: 29},
        {3: 41, 4: 37, 13: 40}, {1: 38, 4: 29, 5: 35, 8: 30, 9: 31}, {6: 48, 10: 50, 11: 44, 14: 41, 15: 47},
        {6: 26, 7: 32, 9: 38, 12: 29, 13: 30}, {6: 23, 10: 20, 13: 25, 14: 18, 15: 22}, {2: 14, 8: 11, 10: 17, 12:13},
        {3: 27, 5: 24, 13: 26}, {4: 20, 8: 19, 13: 14}, {2: 27, 3: 21, 4: 28, 7: 30, 14: 29}, {3: 39, 7: 34, 10: 40, 14: 35}
    ]
    communication_time_matrix5 = [
        [INF,   0, INF, INF, INF, INF, INF, INF, INF,   0, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF,   0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF,   0, INF, INF,   0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF,   0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF,   0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF,   0, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF,   0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF,   0, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   0],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   0, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   0, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   0, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   0, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   0, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   0, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   0, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   0],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF]
    ]
    not_in_OR5 = [1, 14, 15, 16, 17, 18]
    OR5 = {(2, 3, 9): (10, 11, 12, 13), (4,5,6):(7,8)}
    OR_in5 = {2:(1,0)}  #表示第2个OR在第1个OR的"0"(左侧)线路

    """job6"""
    computation_time_matrix6 = [
        {3: 38, 4: 33, 11: 36}, {2: 22, 3: 21, 6: 19}, {5: 14}, {5: 17, 8: 20}, {3: 36, 6: 33, 11: 39},
        {2: 24, 3: 20, 10: 18}, {3: 21, 9: 17, 15: 24}, {8: 38}, {4: 19, 11: 15}, {8: 14, 10: 19, 15: 17},
        {1: 25, 4: 21, 9: 19, 13: 28}, {7: 42, 12: 43}, {1: 48, 2: 42, 6: 46}, {7: 10, 10: 14},
        {2: 14, 13: 16, 14: 13}, {1: 36, 4: 33, 5: 31, 8: 34}, {1: 47, 12: 44}, {9: 30, 10: 26, 13: 29},
        {1: 18, 5: 19, 12: 15}, {9:24, 11:25}
    ]
    communication_time_matrix6 = [
        [INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, 0, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, 0, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF]
    ]
    not_in_OR6 = [1, 2, 3, 4, 5, 14, 15, 16, 17, 18, 19, 20]
    OR6 = {(6,7,9):(11,12,13),(8,):(10,)}
    OR_in6 = {2:(1,0)}

    """job7"""
    computation_time_matrix7 = [
        {7: 12, 8: 17}, {2: 7, 5: 6, 8: 8, 12: 11}, {11: 30, 15: 27}, {7: 27}, {2: 10, 3: 11, 9: 16},
        {4: 46, 11: 50, 13: 49}, {3: 22}, {10: 10, 5: 9}, {1: 27, 2: 28, 6: 24}, {6: 8, 12: 5, 13: 11},
        {2: 47, 9: 48}, {1: 27, 8: 30}, {9: 18, 13: 19, 14: 20}, {7: 22, 11: 20},
        {7: 13, 8: 11, 13: 14}, {6: 15, 14: 10}, {4: 21, 5: 26, 7: 20}, {4: 29, 5: 30, 13: 26},
        {1: 35, 2: 31}, {5: 22, 6: 18, 11: 23}, {1: 32, 7: 33, 10: 28}
    ]
    communication_time_matrix7 = [
        [INF, 0, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF],
        [INF, INF, 0, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, 0, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, 0, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF]
    ]
    not_in_OR7 = [1, 10, 11, 18, 19, 20, 21]
    OR7 = {(2, 6): (7, 8, 9), (3,): (4,5), (12, 13):(14,17),(15,):(16,)}
    OR_in7 = {2: (1, 0),4: (3,1)}

    """job8"""
    computation_time_matrix8 = [
        {4: 50}, {8: 23, 15: 21}, {12: 35}, {4: 11, 8: 16, 7: 13}, {2: 18, 11: 20},
        {4: 36, 13: 33}, {1: 38, 3: 35}, {6: 16, 15: 17}, {9: 24}, {5: 23, 14: 26},
        {1: 15, 5: 16, 8: 17}, {3: 43, 10: 49}, {10: 44}, {13: 32, 14: 31},
        {10: 36, 13: 38}, {14: 28}, {2: 39, 10: 34}, {5: 18, 7: 15}, {11: 16}, {3: 45, 12: 48}
    ]
    communication_time_matrix8 = [
        [INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, 0, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, 0, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, 0, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, 0, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF]
    ]
    not_in_OR8 = [1, 17, 20]
    OR8 = {(2, 3, 7, 8, 11): (12, 13, 14, 15, 16), (4, 5): (6,), (9,):(10,), (18,):(19,)}
    OR_in8 = {2: (1, 0), 3: (1, 0)}

    """job9"""
    computation_time_matrix9 = [
        {5: 31, 9:27, 12:21, 13:28, 14:23}, {7: 21}, {7: 21, 9:22, 10:25, 13:20}, {3: 13, 15: 15},
        {3: 6, 6:5, 9:7, 12:10, 13:9}, {1: 37, 2:33, 5:39, 6:29, 10:32}, {4: 7, 10:8, 11:9, 13:5, 15:6},
        {1: 42, 4:41, 8:39, 9:45, 12:44}, {4: 10, 7:9, 14:14}, {4: 19, 5:14, 7:15, 8:12, 14:10},
        {3: 28, 9:24, 10:27, 11:22, 15:20}, {4: 45, 6: 41}, {2: 44}, {4: 47, 5:43, 14:44, 15:42},
        {2: 17, 5:14, 6:20, 9:21, 14:18}, {2: 45, 4:42, 13:46}, {1: 27, 3:25, 5:23, 6:28, 15:20},
        {2: 10, 7:12, 9:17, 10:16, 11:11}, {2: 9,13:10}, {4: 18, 6:23, 8:21, 12:25}
    ]
    communication_time_matrix9 = [
        [INF, 0, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, 0, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, 0, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, 0, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF]
    ]
    not_in_OR9 = [1, 2, 6, 7, 11, 12, 13, 14, 15, 16, 20]
    OR9 = {(3,): (4, 5), (8, 9): (10,), (17,): (18, 19)}
    OR_in9 = {}

    """job10"""
    computation_time_matrix10 = [
        {1: 34, 2: 39, 3: 40, 4:33}, {6: 27, 15: 20}, {1: 22, 13:24}, {10: 22, 13: 20}, {4: 37, 7: 35},
        {5: 10, 9: 12}, {8: 39, 12: 32, 14: 36}, {12: 44}, {2: 23, 3: 24, 6:21, 9:19}, {3: 48, 12: 45},
        {14: 17}
    ]
    communication_time_matrix10 = [
        [INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, 0, INF, INF, 0, INF, INF, INF, INF],
        [INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF]
    ]
    not_in_OR10 = [1, 2, 3, 9, 10, 11]
    OR10 = {(4,5,6):(7,8)}
    OR_in10 = {}

    """job11"""
    computation_time_matrix11 = [
        {1: 38, 6: 30}, {5: 39, 8: 40, 14:36, 15:44}, {3:11,5:13,11:9,12:12,13:8}, {5:21,6:23,8:29,13:27,14:25},
        {3:33,4:31,6: 29}, {2: 28, 10: 27}, {1: 40, 14: 42, 15: 46}, {2:6,7:8,9:10,11:11,14:7}, {5: 40, 9:39, 13:36}
    ]
    communication_time_matrix11 = [
        [INF, 0, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, 0, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, 0, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, 0, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, 0, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, 0],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF]
    ]
    not_in_OR11 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    OR11 = {}
    OR_in11 = {}

    """job12"""
    computation_time_matrix12 = [
        {1: 31, 11: 29}, {8: 46, 15: 44}, {5: 5, 11:11}, {12: 41}, {15: 24}, {2: 42, 13: 45},
        {8: 19, 11: 15}, {3: 18, 12: 20}, {6: 5, 14: 7}, {4: 18}, {7: 39}, {6: 13, 10: 7},
        {2: 26, 3:22}, {1: 5, 8: 8, 13:9}, {9:39}, {7: 10, 10: 13}, {4: 41, 12:38},{5:21,9:22,13:19}
    ]
    communication_time_matrix12 = [
        [INF, 0, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, 0, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF]
    ]
    not_in_OR12 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 18]
    OR12 = {(14, 15): (16, 17)}
    OR_in12 = {}

    """job13"""
    computation_time_matrix13 = [
        {3: 46, 4: 47, 10: 44, 11: 41, 13: 50}, {5: 8, 7: 5, 10: 10, 12: 11, 15: 9}, {2: 16, 7: 12},
        {1: 7, 3: 5, 4: 13, 6: 12, 15: 8}, {6: 26, 8: 24, 10: 28}, {1: 5, 9:4, 12: 7, 15: 9},
        {2: 27, 8: 30, 10: 33, 12: 29}, {2: 40}, {3: 23, 6: 24, 7: 29, 9: 21}, {1: 12, 2: 14, 5: 19, 9: 18, 10: 17},
        {10: 47, 11: 49, 12: 50}, {4: 44, 6: 38, 13: 41}, {1: 22, 9: 21, 10: 16, 12: 18},
        {1: 15, 4: 18, 6: 13, 13: 14, 14: 19}, {3: 6, 4: 4, 6: 5, 13: 9}, {5: 15, 11: 18, 14: 13},
        {8: 15, 9: 16, 11: 19}, {2: 44, 15: 50}
    ]
    communication_time_matrix13 = [
        [INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF],
        [INF, INF, 0, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, 0, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF]
    ]
    not_in_OR13 = [1, 17, 18]
    OR13 = {(2, 6, 7, 11): (12, 13, 14, 15, 16), (3, 4): (5,), (8,):(9,10)}
    OR_in13 = {2: (1, 0), 3: (1, 0)}  # 表示第2个OR在第1个OR的"0"(左侧)线路

    """job14"""
    computation_time_matrix14 = [
        {3: 46, 9: 43}, {1: 10, 2: 17, 7: 11, 12: 13}, {4: 8, 7: 9, 8: 10}, {3: 18, 6: 25}, {4: 9},
        {3: 29, 4: 27, 13: 33}, {5: 30, 9: 29}, {2: 9, 3: 8}, {5: 18, 12: 10, 15: 19}, {9: 28, 15: 25},
        {4: 42, 14: 43, 15: 47}, {9: 35, 10: 31, 14: 29}, {6: 9, 10: 7}
    ]
    communication_time_matrix14 = [
        [INF, 0, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, 0, 0, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, 0, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF]
    ]
    not_in_OR14 = [1, 5, 8, 9, 10, 11, 12, 13]
    OR14 = {(2, 3): (4,), (6,):(7,)}
    OR_in14 = {}

    """job15"""
    computation_time_matrix15 = [
        {1: 20, 11: 18}, {12: 41, 13: 43}, {6: 17}, {7: 8}, {2: 12, 15: 15}, {4: 48, 5: 43},
        {9: 47}, {8: 28, 12: 30}, {2: 18, 8: 22}, {14: 50}, {8: 6, 13: 7}, {5: 48, 7: 45},
        {1: 9, 5: 10, 6: 11}, {3: 22, 6: 24, 9: 21}, {1: 42, 12: 47}
    ]
    communication_time_matrix15 = [
        [INF, 0, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, 0, INF, 0, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, 0, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF]
    ]
    not_in_OR15 = [1, 6, 7, 11, 12, 13, 14, 15]
    OR15 = {(2,): (3, 4, 5), (8, 9):(10,)}
    OR_in15 = {}

    """job16"""
    computation_time_matrix16 = [
        {1: 43, 2: 45, 11: 41}, {7: 32}, {3: 33, 6: 39, 13: 35}, {4: 40,5:43,7:41,9:44,15:49},
        {1: 25, 2: 30, 6: 31}, {6: 5, 12: 10}, {3:7, 5:5, 6:9, 11:8, 14:10}, {5: 16, 7: 19, 9: 18},
        {1: 12, 4:18, 9:14, 10:9, 13:15}, {6: 19, 11:11, 12:16, 13:20, 14:21},
        {4: 31, 7:39, 8:30, 10:33, 15:40}, {5: 28, 8: 27, 14: 24}, {1: 50, 4:44, 7:47, 8:49, 15:48},
        {1: 17, 2:19, 8:20, 9:21}, {1: 8, 2:7, 4:6, 5:9, 7:10}, {9: 5, 12: 6},
        {1: 21, 3:20, 8:24, 13:19, 15:23}, {4: 26, 5: 27, 10: 24, 11: 18}, {3: 19, 11: 20},
        {7: 19, 8: 20, 14: 15, 15: 23}, {2: 27, 12: 29, 14: 14}
    ]
    communication_time_matrix16 = [
        [INF, 0, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, 0, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF]
    ]
    not_in_OR16 = [1, 18, 21]
    OR16 = {(2, 3, 4, 5): (6, 15, 16, 17), (7,8,9,10,11): (12, 13,14), (19,): (20,)}
    OR_in16 = {2: (1, 1)}

    """job17"""
    computation_time_matrix17 = [
        {10: 46, 11: 44}, {5: 16, 10: 13}, {4: 11}, {12: 13, 13: 14}, {7: 11, 12: 17},
        {13: 46}, {2: 23, 11: 19}, {1: 20, 5: 18, 9: 17}, {2: 29, 12: 30}, {13: 16, 15: 13},
        {1: 24}, {4: 21, 8: 17}, {6: 33}, {1: 17, 3: 12, 6: 14}, {5: 8, 8: 7}, {1: 5, 8: 8},
        {2: 42}, {2: 15, 10: 18}, {1: 6, 6: 7}, {9: 5, 10: 10, 11: 9}, {3: 15, 12: 18},
        {6: 19, 11: 17, 14: 20}
    ]
    communication_time_matrix17 = [
        [INF, 0, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, 0, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, 0, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, 0, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, 0, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF]
    ]
    not_in_OR17 = [1, 12, 13, 17, 18, 22]
    OR17 = {(2, 7): (8, 9, 10, 11), (3, 4, 5): (6,), (14,): (15,16), (19, 20): (21,)}
    OR_in17 = {2: (1, 0)}

    """job18"""
    computation_time_matrix18 = [
        {3: 8, 8: 13}, {5: 16, 6: 12, 8: 13}, {2: 21}, {1: 13, 5: 16, 10: 18}, {9: 17},
        {5: 46, 8: 47}, {3: 44, 7: 48, 13: 49}, {5: 17, 6: 14, 13:10}, {8: 16, 15: 13},
        {3: 28, 11: 27, 15: 30}, {10: 48, 13: 50}, {5: 31, 13: 32, 15: 36},
        {3: 30, 6: 28, 9: 26}, {2: 11}, {1: 16, 14: 18}, {4: 18, 15: 19}, {3: 36, 10: 32, 14: 35}
    ]
    communication_time_matrix18 = [
        [INF, 0, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, 0, 0, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, 0, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF]
    ]
    not_in_OR18 = [1, 6, 7, 10, 11, 12, 16, 17]
    OR18 = {(2, 3): (4, 5), (8,): (9,), (13,):(14,15)}
    OR_in18 = {}

    '''HEFT'''
    # 任务执行时间矩阵
    computation_time_matrix = [
        {3: 8, 8: 13}, {5: 16,6:12,8:13}, {2: 21}, {1: 13,5:16,10:18}, {5: 46, 8: 47},
        {3: 44, 7: 48,13:49}, {5: 17,6:14,13:10}, {5: 16,15:13}, {3: 28,11:27,15:30},
        {10: 48, 13: 50}
    ]
    # computation_time_matrix = [
    #     {1: 5, 3: 3}, {1: 6, 2: 4, 3: 7}, {2: 3,4:6}, {1: 3, 2: 5, 3: 4}, {1: 6, 3: 7},
    #     {1: 8, 2: 4, 3: 7}, {1: 7, 2: 4, 3: 3}, {2: 6, 3: 7}, {1: 8, 2: 7, 3: 3},
    #     {1: 8, 2: 5}
    # ]
    colors = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2, 7: 3, 8: 3, 9: 4, 10: 4}
    # 任务通信时间矩阵
    communication_time_matrix = [
        [INF, 0, INF, 0, INF, INF, INF, INF, INF, INF],
        [INF, INF, 0, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, 0, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, 0, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, 0, 0, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, 0, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, 0, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, 0],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF]
    ]
    not_in_OR=[1,5,6,9,10]
    OR={(2,3):(4,),(7,):(8,)}
    OR_in = {}



    # computation_time_matrix = [
    #     {1: 40, 2: 36}, {5: 78, 7: 72}, {2: 19,4:22}, {1: 36, 2: 30, 5: 37}, {9: 49, 10: 47},
    #     {2: 25, 3: 20, 4: 31}, {7: 56, 8: 50}, {5: 62, 6: 69}, {8: 44, 9: 49},{2: 50, 3: 45,4:55},
    #     {9: 45, 10: 52}, {3: 40, 4: 35}, {5: 20, 6: 23}, {7: 17, 9: 21},{7: 42, 8: 38,9:48},
    #     {2: 43, 4: 37}, {5: 82, 6: 75}, {6: 28, 8: 30}, {7: 11, 8: 15},{3: 69, 5: 72},
    #     {3: 28, 6: 25, 7: 20}, {1: 46, 2: 43, 3:39}, {4: 46, 5: 47, 6:48}, {9: 61, 10: 72}
    # ]
    # colors = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0,11:1,12:1,13:1,14:1,15:1,16:2,17:2,18:2,19:2,20:2,21:2,22:2,23:2,24:2}
    # # 任务通信时间矩阵
    # communication_time_matrix = [
    #     [INF,   0, INF, INF,   0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
    #     [INF, INF,   0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
    #     [INF, INF, INF,   0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
    #     [INF, INF, INF, INF, INF, INF, INF, INF, INF,   0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
    #     [INF, INF, INF, INF, INF,   0, INF,   0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
    #     [INF, INF, INF, INF, INF, INF,   0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
    #     [INF, INF, INF, INF, INF, INF, INF, INF,   0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
    #     [INF, INF, INF, INF, INF, INF, INF, INF,   0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
    #     [INF, INF, INF, INF, INF, INF, INF, INF, INF,   0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
    #     [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
    #     [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   0,   0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
    #     [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   0, INF, INF, INF, INF, INF, INF, INF, INF, INF],
    #     [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
    #     [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   0, INF, INF, INF, INF, INF, INF, INF, INF, INF],
    #     [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
    #     [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   0, INF,   0, INF, INF, INF, INF, INF],
    #     [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   0, INF, INF, INF, INF, INF, INF],
    #     [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   0, INF, INF, INF],
    #     [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   0, INF, INF, INF, INF],
    #     [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   0, INF, INF, INF],
    #     [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
    #     [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   0, INF],
    #     [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   0],
    #     [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF]
    # ]
    # not_in_OR = [1, 10, 11, 15, 16,17,18,19,20,21,22,23,24]
    # OR = {(2, 3,4): (5,9), (6,7): (8,), (12,):(13,14)}
    # # OR = {(2, 3, 4): (), (5,6, 7): (8, 9), (12,): (13, 14)}
    # OR_in={2:(1,1)}  #表示第2个OR在第1个OR的"1"(右侧)线路



    # computation_time_matrix = [
    #     {1:5,2:3}, {2:5}, {3:6,4:5}, {5:4}, {2:4,1:5}, {3:2,4:3}, {2:5}, {3:4}, {4:5},
    #     {1:4,2:3}, {1:2,2:4}, {3:5}, {3:4,5:3}
    # ]
    # ctm=[2,1,2,1,2,2,1,1,1,2,2,1,2]
    # colors={1:0,2:0,3:1,4:1,5:2,6:2,7:2,8:3,9:3,10:4,11:4,12:4,13:4}
    # not_in_OR = [1, 2, 3, 4, 5,6,7,8,9,10,11,12,13]
    # communication_time_matrix = [
    #     [INF, 0.00, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
    #     [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
    #     [INF, INF, INF, 0.00, INF, INF, INF, INF, INF, INF, INF, INF, INF],
    #     [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
    #     [INF, INF, INF, INF, INF, 0.00, INF, INF, INF, INF, INF, INF, INF],
    #     [INF, INF, INF, INF, INF, INF, 0.00, INF, INF, INF, INF, INF, INF],
    #     [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
    #     [INF, INF, INF, INF, INF, INF, INF, INF, 0.00, INF, INF, INF, INF],
    #     [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
    #     [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0.00, INF, INF],
    #     [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0.00],
    #     [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0.00],
    #     [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF]
    # ]
    # communication_time_matrix = [
    #     [INF, 0, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
    #     [INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
    #     [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
    #     [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
    #     [INF, INF, INF, INF, INF, 0, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
    #     [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
    #     [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
    #     [INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF],
    #     [INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF],
    #     [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
    #     [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF],
    #     [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF],
    #     [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
    #     [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF],
    #     [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF],
    #     [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
    #     [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
    #     [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0],
    #     [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
    # ]
    # OR = {}


    """多条规划线路的DAG图"""
    computation_time_matrix = [
        {1: 31, 11: 29}, {8: 46, 15: 44}, {5: 5, 11: 11}, {12: 41}, {15: 24}, {2: 42, 13: 45},
        {8: 19, 11: 15}, {3: 18, 12: 20}, {6: 5, 14: 7}, {4: 18}, {7: 39}, {6: 13, 10: 7},
        {2: 26, 3: 22}, {1: 5, 8: 8, 13: 9}, {9: 39}, {7: 10, 10: 13}, {4: 41, 12: 38}, {5: 21, 9: 22, 13: 19}
    ]
    communication_time_matrix = [
        [INF, 0, INF, 0, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, 0, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, 0, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF]
    ]
    not_in_OR = [1, 7, 8, 12, 13, 18]
    OR = [((2, 3), (4,), (5,6)),((9,10),(11,)),((14,15),(16,17))]
    OR_in = {}

    computation_time_matrix19 = [
        {1: 39.7, 2: 28.8}, {1: 40.6, 2: 53.8, 3: 12.3}, {2: 61.8, 4: 30.7},
        {1: 18.3, 3: 17.1, 4: 66.4}, {1: 40.1, 3: 79.5},
        {1: 20.7, 2: 14.5, 3: 47.6, 4: 43.7}, {2: 62.4, 3: 14.9, 4: 54.1},
        {1: 35.0, 2: 52.7, 4: 54.7}, {1: 19.2, 2: 26.8, 3: 32.7, 4: 69.8},
        {3: 77.4, 4: 31.5}
    ]
            # computation_time_matrix19 = [
    #     {1: 5, 3: 3}, {1: 6, 2: 4, 3: 7}, {2: 3,4:6}, {1: 3, 2: 5, 3: 4}, {1: 6, 3: 7},
    #     {1: 8, 2: 4, 3: 7}, {1: 7, 2: 4, 3: 3}, {2: 6, 3: 7}, {1: 8, 2: 7, 3: 3},
    #     {1: 8, 2: 5}
    # ]
    # 任务通信时间矩阵
    communication_time_matrix19 = [
        [INF, 0, INF, 0, INF, INF, INF, INF, INF, INF],
        [INF, INF, 0, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, 0, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, 0, 0, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, 0],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, 0],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF]
    ]
    not_in_OR19 = [1, 5, 6, 7, 10]
    OR19 = {(2, 3): (4,), (8,): (9,)}
    OR_in19 = {}


    print("start")

    problem = [#[19],
        [1, 2, 3, 5, 6, 10, 11, 12, 15], [1, 4, 5, 7, 8, 10, 13, 14, 16], [3, 5, 6, 9, 10, 11, 13, 14, 16],
        [1, 2, 4, 7, 8, 12, 15, 17, 18], [2, 3, 6, 9, 11, 12, 15, 17, 18], [4, 7, 8, 9, 13, 14, 16, 17, 18],
        [1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15], [1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17],
        [1, 2, 4, 6, 7, 8, 10, 12, 14, 15, 17, 18], [2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 16, 18],
        [2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18], [4, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 18],
        [2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18], [1, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    ]
    # 1: 380   [1, 2, 3, 5, 6, 10, 11, 12, 15]的问题一目标makespan
    file_ms = {1: 8, 2: 370, 3: 305, 4: 404,
               5: 366, 6: 350, 7: 430, 8: 406, 9: 460,
               10:423, 11:417, 12:413, 13:533, 14:514, 15:631}
    file_et = {1: 0.002}

    ga=[]
    gwo=[]
    pso=[]
    tabu=[]
    sa=[]
    for pro in range(0, len(problem)):
        print(f'pro{pro + 1}', problem[pro])
        # targetms = file_ms[pro+1]
        # targetet = file_et[pro+1]


        targetms = 0   #不设定目标值的情况
        targetet = INF

        a = []
        b = []
        c = []
        d = []
        e = []
        for job in problem[pro]:
            a.append(eval(f'computation_time_matrix{job}'))
            b.append(eval(f'communication_time_matrix{job}'))
            c.append(eval(f'not_in_OR{job}'))
            d.append(eval(f'OR{job}'))
            e.append(eval(f'OR_in{job}'))

        job_label = problem[pro]
        computation_time_matrix, communication_time_matrix, not_in_OR, OR, OR_in = final_matrix(a, b, c, d, e)

    # b.append(communication_time_matrix)  #MGA论文里的样例，此外需要注释
    # b.append(communication_time_matrix)  #多条规划线路的DAG图的样例，此外需要注释

    # """problem1"""
    # job_label=[1,2,3,10,11,12]
    # a.append(computation_time_matrix1)
    # a.append(computation_time_matrix2)
    # a.append(computation_time_matrix3)
    # a.append(computation_time_matrix10)
    # a.append(computation_time_matrix11)
    # a.append(computation_time_matrix12)
    #
    # b.append(communication_time_matrix1)
    # b.append(communication_time_matrix2)
    # b.append(communication_time_matrix3)
    # b.append(communication_time_matrix10)
    # b.append(communication_time_matrix11)
    # b.append(communication_time_matrix12)
    #
    # c.append(not_in_OR1)
    # c.append(not_in_OR2)
    # c.append(not_in_OR3)
    # c.append(not_in_OR10)
    # c.append(not_in_OR11)
    # c.append(not_in_OR12)
    #
    # d.append(OR1)
    # d.append(OR2)
    # d.append(OR3)
    # d.append(OR10)
    # d.append(OR11)
    # d.append(OR12)
    #
    # e.append(OR_in1)
    # e.append(OR_in2)
    # e.append(OR_in3)
    # e.append(OR_in10)
    # e.append(OR_in11)
    # e.append(OR_in12)

    # """problem2"""
    # job_label=[4,5,6,13,14,15]
    # a.append(computation_time_matrix4)
    # a.append(computation_time_matrix5)
    # a.append(computation_time_matrix6)
    # a.append(computation_time_matrix13)
    # a.append(computation_time_matrix14)
    # a.append(computation_time_matrix15)
    #
    # b.append(communication_time_matrix4)
    # b.append(communication_time_matrix5)
    # b.append(communication_time_matrix6)
    # b.append(communication_time_matrix13)
    # b.append(communication_time_matrix14)
    # b.append(communication_time_matrix15)
    #
    # c.append(not_in_OR4)
    # c.append(not_in_OR5)
    # c.append(not_in_OR6)
    # c.append(not_in_OR13)
    # c.append(not_in_OR14)
    # c.append(not_in_OR15)
    #
    # d.append(OR4)
    # d.append(OR5)
    # d.append(OR6)
    # d.append(OR13)
    # d.append(OR14)
    # d.append(OR15)
    #
    # e.append(OR_in4)
    # e.append(OR_in5)
    # e.append(OR_in6)
    # e.append(OR_in13)
    # e.append(OR_in14)
    # e.append(OR_in15)

    # """problem3"""
    # job_label = [7, 8, 9, 16, 17, 18]
    # a.append(computation_time_matrix7)
    # a.append(computation_time_matrix8)
    # a.append(computation_time_matrix9)
    # a.append(computation_time_matrix16)
    # a.append(computation_time_matrix17)
    # a.append(computation_time_matrix18)
    #
    # b.append(communication_time_matrix7)
    # b.append(communication_time_matrix8)
    # b.append(communication_time_matrix9)
    # b.append(communication_time_matrix16)
    # b.append(communication_time_matrix17)
    # b.append(communication_time_matrix18)
    #
    # c.append(not_in_OR7)
    # c.append(not_in_OR8)
    # c.append(not_in_OR9)
    # c.append(not_in_OR16)
    # c.append(not_in_OR17)
    # c.append(not_in_OR18)
    #
    # d.append(OR7)
    # d.append(OR8)
    # d.append(OR9)
    # d.append(OR16)
    # d.append(OR17)
    # d.append(OR18)
    #
    # e.append(OR_in7)
    # e.append(OR_in8)
    # e.append(OR_in9)
    # e.append(OR_in16)
    # e.append(OR_in17)
    # e.append(OR_in18)

    # """problem4"""
    # job_label = [1, 4, 7, 10, 13, 16]
    # a.append(computation_time_matrix1)
    # a.append(computation_time_matrix4)
    # a.append(computation_time_matrix7)
    # a.append(computation_time_matrix10)
    # a.append(computation_time_matrix13)
    # a.append(computation_time_matrix16)
    #
    # b.append(communication_time_matrix1)
    # b.append(communication_time_matrix4)
    # b.append(communication_time_matrix7)
    # b.append(communication_time_matrix10)
    # b.append(communication_time_matrix13)
    # b.append(communication_time_matrix16)
    #
    # c.append(not_in_OR1)
    # c.append(not_in_OR4)
    # c.append(not_in_OR7)
    # c.append(not_in_OR10)
    # c.append(not_in_OR13)
    # c.append(not_in_OR16)
    #
    # d.append(OR1)
    # d.append(OR4)
    # d.append(OR7)
    # d.append(OR10)
    # d.append(OR13)
    # d.append(OR16)
    #
    # e.append(OR_in1)
    # e.append(OR_in4)
    # e.append(OR_in7)
    # e.append(OR_in10)
    # e.append(OR_in13)
    # e.append(OR_in16)

    # """problem5"""
    # job_label = [2, 5, 8, 11, 14, 17]
    # a.append(computation_time_matrix2)
    # a.append(computation_time_matrix5)
    # a.append(computation_time_matrix8)
    # a.append(computation_time_matrix11)
    # a.append(computation_time_matrix14)
    # a.append(computation_time_matrix17)
    #
    # b.append(communication_time_matrix2)
    # b.append(communication_time_matrix5)
    # b.append(communication_time_matrix8)
    # b.append(communication_time_matrix11)
    # b.append(communication_time_matrix14)
    # b.append(communication_time_matrix17)
    #
    # c.append(not_in_OR2)
    # c.append(not_in_OR5)
    # c.append(not_in_OR8)
    # c.append(not_in_OR11)
    # c.append(not_in_OR14)
    # c.append(not_in_OR17)
    #
    # d.append(OR2)
    # d.append(OR5)
    # d.append(OR8)
    # d.append(OR11)
    # d.append(OR14)
    # d.append(OR17)
    #
    # e.append(OR_in2)
    # e.append(OR_in5)
    # e.append(OR_in8)
    # e.append(OR_in11)
    # e.append(OR_in14)
    # e.append(OR_in17)

    # """problem6"""
    # job_label = [3, 6, 9, 12, 15, 18]
    # a.append(computation_time_matrix3)
    # a.append(computation_time_matrix6)
    # a.append(computation_time_matrix9)
    # a.append(computation_time_matrix12)
    # a.append(computation_time_matrix15)
    # a.append(computation_time_matrix18)
    #
    # b.append(communication_time_matrix3)
    # b.append(communication_time_matrix6)
    # b.append(communication_time_matrix9)
    # b.append(communication_time_matrix12)
    # b.append(communication_time_matrix15)
    # b.append(communication_time_matrix18)
    #
    # c.append(not_in_OR3)
    # c.append(not_in_OR6)
    # c.append(not_in_OR9)
    # c.append(not_in_OR12)
    # c.append(not_in_OR15)
    # c.append(not_in_OR18)
    #
    # d.append(OR3)
    # d.append(OR6)
    # d.append(OR9)
    # d.append(OR12)
    # d.append(OR15)
    # d.append(OR18)
    #
    # e.append(OR_in3)
    # e.append(OR_in6)
    # e.append(OR_in9)
    # e.append(OR_in12)
    # e.append(OR_in15)
    # e.append(OR_in18)

    # """problem7"""
    # job_label = [1, 4, 8, 12, 15, 17]
    # a.append(computation_time_matrix1)
    # a.append(computation_time_matrix4)
    # a.append(computation_time_matrix8)
    # a.append(computation_time_matrix12)
    # a.append(computation_time_matrix15)
    # a.append(computation_time_matrix17)
    #
    # b.append(communication_time_matrix1)
    # b.append(communication_time_matrix4)
    # b.append(communication_time_matrix8)
    # b.append(communication_time_matrix12)
    # b.append(communication_time_matrix15)
    # b.append(communication_time_matrix17)
    #
    # c.append(not_in_OR1)
    # c.append(not_in_OR4)
    # c.append(not_in_OR8)
    # c.append(not_in_OR12)
    # c.append(not_in_OR15)
    # c.append(not_in_OR17)
    #
    # d.append(OR1)
    # d.append(OR4)
    # d.append(OR8)
    # d.append(OR12)
    # d.append(OR15)
    # d.append(OR17)
    #
    # e.append(OR_in1)
    # e.append(OR_in4)
    # e.append(OR_in8)
    # e.append(OR_in12)
    # e.append(OR_in15)
    # e.append(OR_in17)

    # """problem8"""
    # job_label = [2, 6, 7, 10, 14, 18]
    # a.append(computation_time_matrix2)
    # a.append(computation_time_matrix6)
    # a.append(computation_time_matrix7)
    # a.append(computation_time_matrix10)
    # a.append(computation_time_matrix14)
    # a.append(computation_time_matrix18)
    #
    # b.append(communication_time_matrix2)
    # b.append(communication_time_matrix6)
    # b.append(communication_time_matrix7)
    # b.append(communication_time_matrix10)
    # b.append(communication_time_matrix14)
    # b.append(communication_time_matrix18)
    #
    # c.append(not_in_OR2)
    # c.append(not_in_OR6)
    # c.append(not_in_OR7)
    # c.append(not_in_OR10)
    # c.append(not_in_OR14)
    # c.append(not_in_OR18)
    #
    # d.append(OR2)
    # d.append(OR6)
    # d.append(OR7)
    # d.append(OR10)
    # d.append(OR14)
    # d.append(OR18)
    #
    # e.append(OR_in2)
    # e.append(OR_in6)
    # e.append(OR_in7)
    # e.append(OR_in10)
    # e.append(OR_in14)
    # e.append(OR_in18)
    # #

    # """problem9"""
    # job_label = [3, 5, 9, 11, 13, 16]
    # a.append(computation_time_matrix3)
    # a.append(computation_time_matrix5)
    # a.append(computation_time_matrix9)
    # a.append(computation_time_matrix11)
    # a.append(computation_time_matrix13)
    # a.append(computation_time_matrix16)
    #
    # b.append(communication_time_matrix3)
    # b.append(communication_time_matrix5)
    # b.append(communication_time_matrix9)
    # b.append(communication_time_matrix11)
    # b.append(communication_time_matrix13)
    # b.append(communication_time_matrix16)
    #
    # c.append(not_in_OR3)
    # c.append(not_in_OR5)
    # c.append(not_in_OR9)
    # c.append(not_in_OR11)
    # c.append(not_in_OR13)
    # c.append(not_in_OR16)
    #
    # d.append(OR3)
    # d.append(OR5)
    # d.append(OR9)
    # d.append(OR11)
    # d.append(OR13)
    # d.append(OR16)
    #
    # e.append(OR_in3)
    # e.append(OR_in5)
    # e.append(OR_in9)
    # e.append(OR_in11)
    # e.append(OR_in13)
    # e.append(OR_in16)

    # """problem10"""
    # job_label = [1, 2, 3, 5, 6, 10, 11, 12, 15]
    # a.append(computation_time_matrix1)
    # a.append(computation_time_matrix2)
    # a.append(computation_time_matrix3)
    # a.append(computation_time_matrix5)
    # a.append(computation_time_matrix6)
    # a.append(computation_time_matrix10)
    # a.append(computation_time_matrix11)
    # a.append(computation_time_matrix12)
    # a.append(computation_time_matrix15)
    #
    # b.append(communication_time_matrix1)
    # b.append(communication_time_matrix2)
    # b.append(communication_time_matrix3)
    # b.append(communication_time_matrix5)
    # b.append(communication_time_matrix6)
    # b.append(communication_time_matrix10)
    # b.append(communication_time_matrix11)
    # b.append(communication_time_matrix12)
    # b.append(communication_time_matrix15)
    #
    # c.append(not_in_OR1)
    # c.append(not_in_OR2)
    # c.append(not_in_OR3)
    # c.append(not_in_OR5)
    # c.append(not_in_OR6)
    # c.append(not_in_OR10)
    # c.append(not_in_OR11)
    # c.append(not_in_OR12)
    # c.append(not_in_OR15)
    #
    # d.append(OR1)
    # d.append(OR2)
    # d.append(OR3)
    # d.append(OR5)
    # d.append(OR6)
    # d.append(OR10)
    # d.append(OR11)
    # d.append(OR12)
    # d.append(OR15)
    #
    # e.append(OR_in1)
    # e.append(OR_in2)
    # e.append(OR_in3)
    # e.append(OR_in5)
    # e.append(OR_in6)
    # e.append(OR_in10)
    # e.append(OR_in11)
    # e.append(OR_in12)
    # e.append(OR_in15)

    # """problem11"""
    # job_label = [4, 7, 8, 9, 13, 14, 16, 17, 18]
    # a.append(computation_time_matrix4)
    # a.append(computation_time_matrix7)
    # a.append(computation_time_matrix8)
    # a.append(computation_time_matrix9)
    # a.append(computation_time_matrix13)
    # a.append(computation_time_matrix14)
    # a.append(computation_time_matrix16)
    # a.append(computation_time_matrix17)
    # a.append(computation_time_matrix18)
    #
    # b.append(communication_time_matrix4)
    # b.append(communication_time_matrix7)
    # b.append(communication_time_matrix8)
    # b.append(communication_time_matrix9)
    # b.append(communication_time_matrix13)
    # b.append(communication_time_matrix14)
    # b.append(communication_time_matrix16)
    # b.append(communication_time_matrix17)
    # b.append(communication_time_matrix18)
    #
    # c.append(not_in_OR4)
    # c.append(not_in_OR7)
    # c.append(not_in_OR8)
    # c.append(not_in_OR9)
    # c.append(not_in_OR13)
    # c.append(not_in_OR14)
    # c.append(not_in_OR16)
    # c.append(not_in_OR17)
    # c.append(not_in_OR18)
    #
    # d.append(OR4)
    # d.append(OR7)
    # d.append(OR8)
    # d.append(OR9)
    # d.append(OR13)
    # d.append(OR14)
    # d.append(OR16)
    # d.append(OR17)
    # d.append(OR18)
    #
    # e.append(OR_in4)
    # e.append(OR_in7)
    # e.append(OR_in8)
    # e.append(OR_in9)
    # e.append(OR_in13)
    # e.append(OR_in14)
    # e.append(OR_in16)
    # e.append(OR_in17)
    # e.append(OR_in18)

    # """problem12"""
    # job_label = [1, 4, 5, 7, 8, 10, 13, 14, 16]
    # a.append(computation_time_matrix1)
    # a.append(computation_time_matrix4)
    # a.append(computation_time_matrix5)
    # a.append(computation_time_matrix7)
    # a.append(computation_time_matrix8)
    # a.append(computation_time_matrix10)
    # a.append(computation_time_matrix13)
    # a.append(computation_time_matrix14)
    # a.append(computation_time_matrix16)
    #
    # b.append(communication_time_matrix1)
    # b.append(communication_time_matrix4)
    # b.append(communication_time_matrix5)
    # b.append(communication_time_matrix7)
    # b.append(communication_time_matrix8)
    # b.append(communication_time_matrix10)
    # b.append(communication_time_matrix13)
    # b.append(communication_time_matrix14)
    # b.append(communication_time_matrix16)
    #
    # c.append(not_in_OR1)
    # c.append(not_in_OR4)
    # c.append(not_in_OR5)
    # c.append(not_in_OR7)
    # c.append(not_in_OR8)
    # c.append(not_in_OR10)
    # c.append(not_in_OR13)
    # c.append(not_in_OR14)
    # c.append(not_in_OR16)
    #
    # d.append(OR1)
    # d.append(OR4)
    # d.append(OR5)
    # d.append(OR7)
    # d.append(OR8)
    # d.append(OR10)
    # d.append(OR13)
    # d.append(OR14)
    # d.append(OR16)
    #
    # e.append(OR_in1)
    # e.append(OR_in4)
    # e.append(OR_in5)
    # e.append(OR_in7)
    # e.append(OR_in8)
    # e.append(OR_in10)
    # e.append(OR_in13)
    # e.append(OR_in14)
    # e.append(OR_in16)

    # """problem13"""
    # job_label = [ 2, 3, 6, 9, 11, 12, 15, 17, 18]
    # a.append(computation_time_matrix2)
    # a.append(computation_time_matrix3)
    # a.append(computation_time_matrix6)
    # a.append(computation_time_matrix9)
    # a.append(computation_time_matrix11)
    # a.append(computation_time_matrix12)
    # a.append(computation_time_matrix15)
    # a.append(computation_time_matrix17)
    # a.append(computation_time_matrix18)
    #
    # b.append(communication_time_matrix2)
    # b.append(communication_time_matrix3)
    # b.append(communication_time_matrix6)
    # b.append(communication_time_matrix9)
    # b.append(communication_time_matrix11)
    # b.append(communication_time_matrix12)
    # b.append(communication_time_matrix15)
    # b.append(communication_time_matrix17)
    # b.append(communication_time_matrix18)
    #
    # c.append(not_in_OR2)
    # c.append(not_in_OR3)
    # c.append(not_in_OR6)
    # c.append(not_in_OR9)
    # c.append(not_in_OR11)
    # c.append(not_in_OR12)
    # c.append(not_in_OR15)
    # c.append(not_in_OR17)
    # c.append(not_in_OR18)
    #
    # d.append(OR2)
    # d.append(OR3)
    # d.append(OR6)
    # d.append(OR9)
    # d.append(OR11)
    # d.append(OR12)
    # d.append(OR15)
    # d.append(OR17)
    # d.append(OR18)
    #
    # e.append(OR_in2)
    # e.append(OR_in3)
    # e.append(OR_in6)
    # e.append(OR_in9)
    # e.append(OR_in11)
    # e.append(OR_in12)
    # e.append(OR_in15)
    # e.append(OR_in17)
    # e.append(OR_in18)

    # """problem14"""
    # job_label = [1, 2, 4, 7, 8, 12, 15, 17, 18]
    # a.append(computation_time_matrix1)
    # a.append(computation_time_matrix2)
    # a.append(computation_time_matrix4)
    # a.append(computation_time_matrix7)
    # a.append(computation_time_matrix8)
    # a.append(computation_time_matrix12)
    # a.append(computation_time_matrix15)
    # a.append(computation_time_matrix17)
    # a.append(computation_time_matrix18)
    #
    # b.append(communication_time_matrix1)
    # b.append(communication_time_matrix2)
    # b.append(communication_time_matrix4)
    # b.append(communication_time_matrix7)
    # b.append(communication_time_matrix8)
    # b.append(communication_time_matrix12)
    # b.append(communication_time_matrix15)
    # b.append(communication_time_matrix17)
    # b.append(communication_time_matrix18)
    #
    # c.append(not_in_OR1)
    # c.append(not_in_OR2)
    # c.append(not_in_OR4)
    # c.append(not_in_OR7)
    # c.append(not_in_OR8)
    # c.append(not_in_OR12)
    # c.append(not_in_OR15)
    # c.append(not_in_OR17)
    # c.append(not_in_OR18)
    #
    # d.append(OR1)
    # d.append(OR2)
    # d.append(OR4)
    # d.append(OR7)
    # d.append(OR8)
    # d.append(OR12)
    # d.append(OR15)
    # d.append(OR17)
    # d.append(OR18)
    #
    # e.append(OR_in1)
    # e.append(OR_in2)
    # e.append(OR_in4)
    # e.append(OR_in7)
    # e.append(OR_in8)
    # e.append(OR_in12)
    # e.append(OR_in15)
    # e.append(OR_in17)
    # e.append(OR_in18)

    # """problem15"""
    # job_label = [3, 5, 6, 9, 10, 11, 13, 14, 16]
    # a.append(computation_time_matrix3)
    # a.append(computation_time_matrix5)
    # a.append(computation_time_matrix6)
    # a.append(computation_time_matrix9)
    # a.append(computation_time_matrix10)
    # a.append(computation_time_matrix11)
    # a.append(computation_time_matrix13)
    # a.append(computation_time_matrix14)
    # a.append(computation_time_matrix16)
    #
    # b.append(communication_time_matrix3)
    # b.append(communication_time_matrix5)
    # b.append(communication_time_matrix6)
    # b.append(communication_time_matrix9)
    # b.append(communication_time_matrix10)
    # b.append(communication_time_matrix11)
    # b.append(communication_time_matrix13)
    # b.append(communication_time_matrix14)
    # b.append(communication_time_matrix16)
    #
    # c.append(not_in_OR3)
    # c.append(not_in_OR5)
    # c.append(not_in_OR6)
    # c.append(not_in_OR9)
    # c.append(not_in_OR10)
    # c.append(not_in_OR11)
    # c.append(not_in_OR13)
    # c.append(not_in_OR14)
    # c.append(not_in_OR16)
    #
    # d.append(OR3)
    # d.append(OR5)
    # d.append(OR6)
    # d.append(OR9)
    # d.append(OR10)
    # d.append(OR11)
    # d.append(OR13)
    # d.append(OR14)
    # d.append(OR16)
    #
    # e.append(OR_in3)
    # e.append(OR_in5)
    # e.append(OR_in6)
    # e.append(OR_in9)
    # e.append(OR_in10)
    # e.append(OR_in11)
    # e.append(OR_in13)
    # e.append(OR_in14)
    # e.append(OR_in16)

    # """problem16"""
    # job_label = [ 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15]
    # a.append(computation_time_matrix1)
    # a.append(computation_time_matrix2)
    # a.append(computation_time_matrix3)
    # a.append(computation_time_matrix4)
    # a.append(computation_time_matrix5)
    # a.append(computation_time_matrix6)
    # a.append(computation_time_matrix10)
    # a.append(computation_time_matrix11)
    # a.append(computation_time_matrix12)
    # a.append(computation_time_matrix13)
    # a.append(computation_time_matrix14)
    # a.append(computation_time_matrix15)
    #
    # b.append(communication_time_matrix1)
    # b.append(communication_time_matrix2)
    # b.append(communication_time_matrix3)
    # b.append(communication_time_matrix4)
    # b.append(communication_time_matrix5)
    # b.append(communication_time_matrix6)
    # b.append(communication_time_matrix10)
    # b.append(communication_time_matrix11)
    # b.append(communication_time_matrix12)
    # b.append(communication_time_matrix13)
    # b.append(communication_time_matrix14)
    # b.append(communication_time_matrix15)
    #
    # c.append(not_in_OR1)
    # c.append(not_in_OR2)
    # c.append(not_in_OR3)
    # c.append(not_in_OR4)
    # c.append(not_in_OR5)
    # c.append(not_in_OR6)
    # c.append(not_in_OR10)
    # c.append(not_in_OR11)
    # c.append(not_in_OR12)
    # c.append(not_in_OR13)
    # c.append(not_in_OR14)
    # c.append(not_in_OR15)
    #
    # d.append(OR1)
    # d.append(OR2)
    # d.append(OR3)
    # d.append(OR4)
    # d.append(OR5)
    # d.append(OR6)
    # d.append(OR10)
    # d.append(OR11)
    # d.append(OR12)
    # d.append(OR13)
    # d.append(OR14)
    # d.append(OR15)
    #
    # e.append(OR_in1)
    # e.append(OR_in2)
    # e.append(OR_in3)
    # e.append(OR_in4)
    # e.append(OR_in5)
    # e.append(OR_in6)
    # e.append(OR_in10)
    # e.append(OR_in11)
    # e.append(OR_in12)
    # e.append(OR_in13)
    # e.append(OR_in14)
    # e.append(OR_in15)

    # """problem17"""
    # job_label = [ 4, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 18]
    # a.append(computation_time_matrix4)
    # a.append(computation_time_matrix5)
    # a.append(computation_time_matrix6)
    # a.append(computation_time_matrix7)
    # a.append(computation_time_matrix8)
    # a.append(computation_time_matrix9)
    # a.append(computation_time_matrix13)
    # a.append(computation_time_matrix14)
    # a.append(computation_time_matrix15)
    # a.append(computation_time_matrix16)
    # a.append(computation_time_matrix17)
    # a.append(computation_time_matrix18)
    #
    # b.append(communication_time_matrix4)
    # b.append(communication_time_matrix5)
    # b.append(communication_time_matrix6)
    # b.append(communication_time_matrix7)
    # b.append(communication_time_matrix8)
    # b.append(communication_time_matrix9)
    # b.append(communication_time_matrix13)
    # b.append(communication_time_matrix14)
    # b.append(communication_time_matrix15)
    # b.append(communication_time_matrix16)
    # b.append(communication_time_matrix17)
    # b.append(communication_time_matrix18)
    #
    # c.append(not_in_OR4)
    # c.append(not_in_OR5)
    # c.append(not_in_OR6)
    # c.append(not_in_OR7)
    # c.append(not_in_OR8)
    # c.append(not_in_OR9)
    # c.append(not_in_OR13)
    # c.append(not_in_OR14)
    # c.append(not_in_OR15)
    # c.append(not_in_OR16)
    # c.append(not_in_OR17)
    # c.append(not_in_OR18)
    #
    # d.append(OR4)
    # d.append(OR5)
    # d.append(OR6)
    # d.append(OR7)
    # d.append(OR8)
    # d.append(OR9)
    # d.append(OR13)
    # d.append(OR14)
    # d.append(OR15)
    # d.append(OR16)
    # d.append(OR17)
    # d.append(OR18)
    #
    # e.append(OR_in4)
    # e.append(OR_in5)
    # e.append(OR_in6)
    # e.append(OR_in7)
    # e.append(OR_in8)
    # e.append(OR_in9)
    # e.append(OR_in13)
    # e.append(OR_in14)
    # e.append(OR_in15)
    # e.append(OR_in16)
    # e.append(OR_in17)
    # e.append(OR_in18)

    # """problem18"""
    # job_label = [ 1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17]
    # a.append(computation_time_matrix1)
    # a.append(computation_time_matrix2)
    # a.append(computation_time_matrix4)
    # a.append(computation_time_matrix5)
    # a.append(computation_time_matrix7)
    # a.append(computation_time_matrix8)
    # a.append(computation_time_matrix10)
    # a.append(computation_time_matrix11)
    # a.append(computation_time_matrix13)
    # a.append(computation_time_matrix14)
    # a.append(computation_time_matrix16)
    # a.append(computation_time_matrix17)
    #
    # b.append(communication_time_matrix1)
    # b.append(communication_time_matrix2)
    # b.append(communication_time_matrix4)
    # b.append(communication_time_matrix5)
    # b.append(communication_time_matrix7)
    # b.append(communication_time_matrix8)
    # b.append(communication_time_matrix10)
    # b.append(communication_time_matrix11)
    # b.append(communication_time_matrix13)
    # b.append(communication_time_matrix14)
    # b.append(communication_time_matrix16)
    # b.append(communication_time_matrix17)
    #
    # c.append(not_in_OR1)
    # c.append(not_in_OR2)
    # c.append(not_in_OR4)
    # c.append(not_in_OR5)
    # c.append(not_in_OR7)
    # c.append(not_in_OR8)
    # c.append(not_in_OR10)
    # c.append(not_in_OR11)
    # c.append(not_in_OR13)
    # c.append(not_in_OR14)
    # c.append(not_in_OR16)
    # c.append(not_in_OR17)
    #
    # d.append(OR1)
    # d.append(OR2)
    # d.append(OR4)
    # d.append(OR5)
    # d.append(OR7)
    # d.append(OR8)
    # d.append(OR10)
    # d.append(OR11)
    # d.append(OR13)
    # d.append(OR14)
    # d.append(OR16)
    # d.append(OR17)
    #
    # e.append(OR_in1)
    # e.append(OR_in2)
    # e.append(OR_in4)
    # e.append(OR_in5)
    # e.append(OR_in7)
    # e.append(OR_in8)
    # e.append(OR_in10)
    # e.append(OR_in11)
    # e.append(OR_in13)
    # e.append(OR_in14)
    # e.append(OR_in16)
    # e.append(OR_in17)

    # """problem19"""
    # job_label = [ 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18]
    # a.append(computation_time_matrix2)
    # a.append(computation_time_matrix3)
    # a.append(computation_time_matrix5)
    # a.append(computation_time_matrix6)
    # a.append(computation_time_matrix8)
    # a.append(computation_time_matrix9)
    # a.append(computation_time_matrix11)
    # a.append(computation_time_matrix12)
    # a.append(computation_time_matrix14)
    # a.append(computation_time_matrix15)
    # a.append(computation_time_matrix17)
    # a.append(computation_time_matrix18)
    #
    # b.append(communication_time_matrix2)
    # b.append(communication_time_matrix3)
    # b.append(communication_time_matrix5)
    # b.append(communication_time_matrix6)
    # b.append(communication_time_matrix8)
    # b.append(communication_time_matrix9)
    # b.append(communication_time_matrix11)
    # b.append(communication_time_matrix12)
    # b.append(communication_time_matrix14)
    # b.append(communication_time_matrix15)
    # b.append(communication_time_matrix17)
    # b.append(communication_time_matrix18)
    #
    # c.append(not_in_OR2)
    # c.append(not_in_OR3)
    # c.append(not_in_OR5)
    # c.append(not_in_OR6)
    # c.append(not_in_OR8)
    # c.append(not_in_OR9)
    # c.append(not_in_OR11)
    # c.append(not_in_OR12)
    # c.append(not_in_OR14)
    # c.append(not_in_OR15)
    # c.append(not_in_OR17)
    # c.append(not_in_OR18)
    #
    # d.append(OR2)
    # d.append(OR3)
    # d.append(OR5)
    # d.append(OR6)
    # d.append(OR8)
    # d.append(OR9)
    # d.append(OR11)
    # d.append(OR12)
    # d.append(OR14)
    # d.append(OR15)
    # d.append(OR17)
    # d.append(OR18)
    #
    # e.append(OR_in2)
    # e.append(OR_in3)
    # e.append(OR_in5)
    # e.append(OR_in6)
    # e.append(OR_in8)
    # e.append(OR_in9)
    # e.append(OR_in11)
    # e.append(OR_in12)
    # e.append(OR_in14)
    # e.append(OR_in15)
    # e.append(OR_in17)
    # e.append(OR_in18)

    # """problem20"""
    # job_label = [1, 2, 4, 6, 7, 8, 10, 12, 14, 15, 17, 18]
    # a.append(computation_time_matrix1)
    # a.append(computation_time_matrix2)
    # a.append(computation_time_matrix4)
    # a.append(computation_time_matrix6)
    # a.append(computation_time_matrix7)
    # a.append(computation_time_matrix8)
    # a.append(computation_time_matrix10)
    # a.append(computation_time_matrix12)
    # a.append(computation_time_matrix14)
    # a.append(computation_time_matrix15)
    # a.append(computation_time_matrix17)
    # a.append(computation_time_matrix18)
    #
    # b.append(communication_time_matrix1)
    # b.append(communication_time_matrix2)
    # b.append(communication_time_matrix4)
    # b.append(communication_time_matrix6)
    # b.append(communication_time_matrix7)
    # b.append(communication_time_matrix8)
    # b.append(communication_time_matrix10)
    # b.append(communication_time_matrix12)
    # b.append(communication_time_matrix14)
    # b.append(communication_time_matrix15)
    # b.append(communication_time_matrix17)
    # b.append(communication_time_matrix18)
    #
    # c.append(not_in_OR1)
    # c.append(not_in_OR2)
    # c.append(not_in_OR4)
    # c.append(not_in_OR6)
    # c.append(not_in_OR7)
    # c.append(not_in_OR8)
    # c.append(not_in_OR10)
    # c.append(not_in_OR12)
    # c.append(not_in_OR14)
    # c.append(not_in_OR15)
    # c.append(not_in_OR17)
    # c.append(not_in_OR18)
    #
    # d.append(OR1)
    # d.append(OR2)
    # d.append(OR4)
    # d.append(OR6)
    # d.append(OR7)
    # d.append(OR8)
    # d.append(OR10)
    # d.append(OR12)
    # d.append(OR14)
    # d.append(OR15)
    # d.append(OR17)
    # d.append(OR18)
    #
    # e.append(OR_in1)
    # e.append(OR_in2)
    # e.append(OR_in4)
    # e.append(OR_in6)
    # e.append(OR_in7)
    # e.append(OR_in8)
    # e.append(OR_in10)
    # e.append(OR_in12)
    # e.append(OR_in14)
    # e.append(OR_in15)
    # e.append(OR_in17)
    # e.append(OR_in18)

    # """problem21"""
    # job_label = [2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 16, 18]
    # a.append(computation_time_matrix2)
    # a.append(computation_time_matrix3)
    # a.append(computation_time_matrix5)
    # a.append(computation_time_matrix6)
    # a.append(computation_time_matrix7)
    # a.append(computation_time_matrix9)
    # a.append(computation_time_matrix10)
    # a.append(computation_time_matrix11)
    # a.append(computation_time_matrix13)
    # a.append(computation_time_matrix14)
    # a.append(computation_time_matrix16)
    # a.append(computation_time_matrix18)
    #
    # b.append(communication_time_matrix2)
    # b.append(communication_time_matrix3)
    # b.append(communication_time_matrix5)
    # b.append(communication_time_matrix6)
    # b.append(communication_time_matrix7)
    # b.append(communication_time_matrix9)
    # b.append(communication_time_matrix10)
    # b.append(communication_time_matrix11)
    # b.append(communication_time_matrix13)
    # b.append(communication_time_matrix14)
    # b.append(communication_time_matrix16)
    # b.append(communication_time_matrix18)
    #
    # c.append(not_in_OR2)
    # c.append(not_in_OR3)
    # c.append(not_in_OR5)
    # c.append(not_in_OR6)
    # c.append(not_in_OR7)
    # c.append(not_in_OR9)
    # c.append(not_in_OR10)
    # c.append(not_in_OR11)
    # c.append(not_in_OR13)
    # c.append(not_in_OR14)
    # c.append(not_in_OR16)
    # c.append(not_in_OR18)
    #
    # d.append(OR2)
    # d.append(OR3)
    # d.append(OR5)
    # d.append(OR6)
    # d.append(OR7)
    # d.append(OR9)
    # d.append(OR10)
    # d.append(OR11)
    # d.append(OR13)
    # d.append(OR14)
    # d.append(OR16)
    # d.append(OR18)
    #
    # e.append(OR_in2)
    # e.append(OR_in3)
    # e.append(OR_in5)
    # e.append(OR_in6)
    # e.append(OR_in7)
    # e.append(OR_in9)
    # e.append(OR_in10)
    # e.append(OR_in11)
    # e.append(OR_in13)
    # e.append(OR_in14)
    # e.append(OR_in16)
    # e.append(OR_in18)

    # """problem22"""
    # job_label = [ 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18]
    # a.append(computation_time_matrix2)
    # a.append(computation_time_matrix3)
    # a.append(computation_time_matrix4)
    # a.append(computation_time_matrix5)
    # a.append(computation_time_matrix6)
    # a.append(computation_time_matrix8)
    # a.append(computation_time_matrix9)
    # a.append(computation_time_matrix10)
    # a.append(computation_time_matrix11)
    # a.append(computation_time_matrix12)
    # a.append(computation_time_matrix13)
    # a.append(computation_time_matrix14)
    # a.append(computation_time_matrix16)
    # a.append(computation_time_matrix17)
    # a.append(computation_time_matrix18)
    #
    # b.append(communication_time_matrix2)
    # b.append(communication_time_matrix3)
    # b.append(communication_time_matrix4)
    # b.append(communication_time_matrix5)
    # b.append(communication_time_matrix6)
    # b.append(communication_time_matrix8)
    # b.append(communication_time_matrix9)
    # b.append(communication_time_matrix10)
    # b.append(communication_time_matrix11)
    # b.append(communication_time_matrix12)
    # b.append(communication_time_matrix13)
    # b.append(communication_time_matrix14)
    # b.append(communication_time_matrix16)
    # b.append(communication_time_matrix17)
    # b.append(communication_time_matrix18)
    #
    # c.append(not_in_OR2)
    # c.append(not_in_OR3)
    # c.append(not_in_OR4)
    # c.append(not_in_OR5)
    # c.append(not_in_OR6)
    # c.append(not_in_OR8)
    # c.append(not_in_OR9)
    # c.append(not_in_OR10)
    # c.append(not_in_OR11)
    # c.append(not_in_OR12)
    # c.append(not_in_OR13)
    # c.append(not_in_OR14)
    # c.append(not_in_OR16)
    # c.append(not_in_OR17)
    # c.append(not_in_OR18)
    #
    # d.append(OR2)
    # d.append(OR3)
    # d.append(OR4)
    # d.append(OR5)
    # d.append(OR6)
    # d.append(OR8)
    # d.append(OR9)
    # d.append(OR10)
    # d.append(OR11)
    # d.append(OR12)
    # d.append(OR13)
    # d.append(OR14)
    # d.append(OR16)
    # d.append(OR17)
    # d.append(OR18)
    #
    # e.append(OR_in2)
    # e.append(OR_in3)
    # e.append(OR_in4)
    # e.append(OR_in5)
    # e.append(OR_in6)
    # e.append(OR_in8)
    # e.append(OR_in9)
    # e.append(OR_in10)
    # e.append(OR_in11)
    # e.append(OR_in12)
    # e.append(OR_in13)
    # e.append(OR_in14)
    # e.append(OR_in16)
    # e.append(OR_in17)
    # e.append(OR_in18)

    # """problem23"""
    # job_label = [ 1, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18]
    # a.append(computation_time_matrix1)
    # a.append(computation_time_matrix4)
    # a.append(computation_time_matrix5)
    # a.append(computation_time_matrix6)
    # a.append(computation_time_matrix7)
    # a.append(computation_time_matrix8)
    # a.append(computation_time_matrix9)
    # a.append(computation_time_matrix11)
    # a.append(computation_time_matrix12)
    # a.append(computation_time_matrix13)
    # a.append(computation_time_matrix14)
    # a.append(computation_time_matrix15)
    # a.append(computation_time_matrix16)
    # a.append(computation_time_matrix17)
    # a.append(computation_time_matrix18)
    #
    # b.append(communication_time_matrix1)
    # b.append(communication_time_matrix4)
    # b.append(communication_time_matrix5)
    # b.append(communication_time_matrix6)
    # b.append(communication_time_matrix7)
    # b.append(communication_time_matrix8)
    # b.append(communication_time_matrix9)
    # b.append(communication_time_matrix11)
    # b.append(communication_time_matrix12)
    # b.append(communication_time_matrix13)
    # b.append(communication_time_matrix14)
    # b.append(communication_time_matrix15)
    # b.append(communication_time_matrix16)
    # b.append(communication_time_matrix17)
    # b.append(communication_time_matrix18)
    #
    # c.append(not_in_OR1)
    # c.append(not_in_OR4)
    # c.append(not_in_OR5)
    # c.append(not_in_OR6)
    # c.append(not_in_OR7)
    # c.append(not_in_OR8)
    # c.append(not_in_OR9)
    # c.append(not_in_OR11)
    # c.append(not_in_OR12)
    # c.append(not_in_OR13)
    # c.append(not_in_OR14)
    # c.append(not_in_OR15)
    # c.append(not_in_OR16)
    # c.append(not_in_OR17)
    # c.append(not_in_OR18)
    #
    # d.append(OR1)
    # d.append(OR4)
    # d.append(OR5)
    # d.append(OR6)
    # d.append(OR7)
    # d.append(OR8)
    # d.append(OR9)
    # d.append(OR11)
    # d.append(OR12)
    # d.append(OR13)
    # d.append(OR14)
    # d.append(OR15)
    # d.append(OR16)
    # d.append(OR17)
    # d.append(OR18)
    #
    # e.append(OR_in1)
    # e.append(OR_in4)
    # e.append(OR_in5)
    # e.append(OR_in6)
    # e.append(OR_in7)
    # e.append(OR_in8)
    # e.append(OR_in9)
    # e.append(OR_in11)
    # e.append(OR_in12)
    # e.append(OR_in13)
    # e.append(OR_in14)
    # e.append(OR_in15)
    # e.append(OR_in16)
    # e.append(OR_in17)
    # e.append(OR_in18)

    # """problem24"""
    # job_label = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    # a.append(computation_time_matrix1)
    # a.append(computation_time_matrix2)
    # a.append(computation_time_matrix3)
    # a.append(computation_time_matrix4)
    # a.append(computation_time_matrix5)
    # a.append(computation_time_matrix6)
    # a.append(computation_time_matrix7)
    # a.append(computation_time_matrix8)
    # a.append(computation_time_matrix9)
    # a.append(computation_time_matrix10)
    # a.append(computation_time_matrix11)
    # a.append(computation_time_matrix12)
    # a.append(computation_time_matrix13)
    # a.append(computation_time_matrix14)
    # a.append(computation_time_matrix15)
    # a.append(computation_time_matrix16)
    # a.append(computation_time_matrix17)
    # a.append(computation_time_matrix18)
    #
    # b.append(communication_time_matrix1)
    # b.append(communication_time_matrix2)
    # b.append(communication_time_matrix3)
    # b.append(communication_time_matrix4)
    # b.append(communication_time_matrix5)
    # b.append(communication_time_matrix6)
    # b.append(communication_time_matrix7)
    # b.append(communication_time_matrix8)
    # b.append(communication_time_matrix9)
    # b.append(communication_time_matrix10)
    # b.append(communication_time_matrix11)
    # b.append(communication_time_matrix12)
    # b.append(communication_time_matrix13)
    # b.append(communication_time_matrix14)
    # b.append(communication_time_matrix15)
    # b.append(communication_time_matrix16)
    # b.append(communication_time_matrix17)
    # b.append(communication_time_matrix18)
    #
    # c.append(not_in_OR1)
    # c.append(not_in_OR2)
    # c.append(not_in_OR3)
    # c.append(not_in_OR4)
    # c.append(not_in_OR5)
    # c.append(not_in_OR6)
    # c.append(not_in_OR7)
    # c.append(not_in_OR8)
    # c.append(not_in_OR9)
    # c.append(not_in_OR10)
    # c.append(not_in_OR11)
    # c.append(not_in_OR12)
    # c.append(not_in_OR13)
    # c.append(not_in_OR14)
    # c.append(not_in_OR15)
    # c.append(not_in_OR16)
    # c.append(not_in_OR17)
    # c.append(not_in_OR18)
    #
    # d.append(OR1)
    # d.append(OR2)
    # d.append(OR3)
    # d.append(OR4)
    # d.append(OR5)
    # d.append(OR6)
    # d.append(OR7)
    # d.append(OR8)
    # d.append(OR9)
    # d.append(OR10)
    # d.append(OR11)
    # d.append(OR12)
    # d.append(OR13)
    # d.append(OR14)
    # d.append(OR15)
    # d.append(OR16)
    # d.append(OR17)
    # d.append(OR18)
    #
    # e.append(OR_in1)
    # e.append(OR_in2)
    # e.append(OR_in3)
    # e.append(OR_in4)
    # e.append(OR_in5)
    # e.append(OR_in6)
    # e.append(OR_in7)
    # e.append(OR_in8)
    # e.append(OR_in9)
    # e.append(OR_in10)
    # e.append(OR_in11)
    # e.append(OR_in12)
    # e.append(OR_in13)
    # e.append(OR_in14)
    # e.append(OR_in15)
    # e.append(OR_in16)
    # e.append(OR_in17)
    # e.append(OR_in18)
    #computation_time_matrix, communication_time_matrix, not_in_OR, OR, OR_in= final_matrix(a, b, c, d, e)



    # print(computation_time_matrix)
    # print(len(communication_time_matrix[0]))
    # for i in communication_time_matrix:
    #     print(i)
    # print(not_in_OR)
    # print(OR)
    # print(OR_in)

        colors = {0:0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 0, 6: 1, 7: 2, 8: 3, 9: 4, 10: 0, 11: 1, 12: 2, 13: 3, 14: 4, 15: 0, 16: 1,
              17: 2, 18: 3, 19: 4, 20: 0, 21: 1, 22: 2, 23: 3, 24: 4}



        # 任务数
        task_number = len(computation_time_matrix)
        # OR内任务节点数
        ORnum=task_number-len(not_in_OR)
        # 每个操作的可选机器数
        ctm = []
        for i in computation_time_matrix:
            ctm.append(len(i))

        jo = []  #每个job的操作数
        for i in b:
            jo.append(len(i))

        # 初始化应用

        appA = Application("A")
        # ApplicationService.init_application(appA, OR_in1, not_in_OR1, OR1, ORnum, task_number, computation_time_matrix1, ctm,communication_time_matrix1)
        ApplicationService.init_application(appA, OR_in, not_in_OR, OR, ORnum, task_number, computation_time_matrix, ctm, communication_time_matrix, jo, job_label)
        # ApplicationService.init_application(appB, task_number, computation_time_matrix, computation_cost_matrix, communication_time_matrix)
        # ApplicationService.init_application(appC, task_number, computation_time_matrix, computation_cost_matrix, communication_time_matrix)
        # ApplicationService.init_application(appD, task_number, computation_time_matrix, computation_cost_matrix, communication_time_matrix)
        # print([qq.name for qq in appA.tasks[96].successors])

        tn=1
        # for i in range(tn):
        #     begin_time = time()
        #     IPPS_RPGA = GeneticScheduler("Genetic")  # 遗传调度
        #     ga_makespan, complete_time = IPPS_RPGA.schedule(appA,targetms)
        #     end_time = time()
        #     run_time = end_time - begin_time
        #     # output_path = 'result of Calculating st^la.txt'
        #     # with open(output_path, 'a', encoding='utf-8') as file1:
        #     #     print(ga_makespan, end=" ", file=file1)
        #     #     print(run_time, file=file1)
        #     print(run_time)

        #     ga.append(run_time)
        #
        # output_path = 'result of Calculating st^la.txt'
        # with open(output_path, 'a', encoding='utf-8') as file1:
        #     print(file=file1)
        print()

        # for i in range(tn):
        #     begin_time = time()
        #     IPPS_GWO = newGWOIPPSScheduler("GWO")  # 灰狼调度
        #     ga_makespan, complete_time = IPPS_GWO.schedule(appA,targetms)
        #     end_time = time()
        #     run_time = end_time - begin_time
        #     output_path = 'result of Calculating st^la.txt'
        #     with open(output_path, 'a', encoding='utf-8') as file1:
        #         print(ga_makespan, end=" ", file=file1)
        #         print(run_time, file=file1)
        #     print(run_time)
        #
        #     gwo.append(run_time)
        #
        # output_path = 'result of Calculating st^la.txt'
        # with open(output_path, 'a', encoding='utf-8') as file1:
        #     print(file=file1)
        # print()
        #
        # for i in range(tn):
        #     begin_time = time()
        #     IPPS_PSO = newPSOIPPSScheduler("PSO")  # 粒子群调度
        #     ga_makespan, complete_time = IPPS_PSO.schedule(appA,targetms)
        #     end_time = time()
        #     run_time = end_time - begin_time
        #     output_path = 'result of Calculating st^la.txt'
        #     with open(output_path, 'a', encoding='utf-8') as file1:
        #         print(ga_makespan, end=" ", file=file1)
        #         print(run_time, file=file1)
        #     print(run_time)
        #
        #     pso.append(run_time)
        #
        # output_path = 'result of Calculating st^la.txt'
        # with open(output_path, 'a', encoding='utf-8') as file1:
        #     print(file=file1)
        # print()
        # #
        # for i in range(tn):
        #     begin_time = time()
        #     Tabu = tabuIPPSScheduler("Tabu")  # 禁忌搜索
        #     ga_makespan, complete_time = Tabu.schedule(appA,targetms)
        #     end_time = time()
        #     run_time = end_time - begin_time
        #     output_path = 'result of Calculating st^la.txt'
        #     with open(output_path, 'a', encoding='utf-8') as file1:
        #         print(ga_makespan, end=" ", file=file1)
        #         print(run_time, file=file1)
        #     print(run_time)
        #
        #     tabu.append(run_time)
        #
        # output_path = 'result of Calculating st^la.txt'
        # with open(output_path, 'a', encoding='utf-8') as file1:
        #     print(file=file1)
        # print()
        #
        # for i in range(tn):
        #     begin_time = time()
        #     SA = SAIPPSScheduler("SA")  # 模拟退火
        #     ga_makespan, complete_time = SA.schedule(appA,targetms)
        #     end_time = time()
        #     run_time = end_time - begin_time
        #     output_path = 'result of Calculating st^la.txt'
        #     with open(output_path, 'a', encoding='utf-8') as file1:
        #         print(ga_makespan, end=" ", file=file1)
        #         print(run_time, file=file1)
        #     print(run_time)
        #
        #     sa.append(run_time)
        #
        # output_path = 'result of Calculating st^la.txt'
        # with open(output_path, 'a', encoding='utf-8') as file1:
        #     print(file=file1)
        # print()

        begin_time = time()
        tracemalloc.start()
        # for i in range(10):
            # start_time = datetime.now()
            # print("-" * 100)
            # cpop = CpopScheduler("CPOP")
            # cpop_tradeoff = cpop.schedule(appA)
            # heft = HeftScheduler("HEFT")
            # heft_tradeoff = heft.schedule(appC)  # 调度器执行调度
            # genetic = GeneticScheduler("Genetic")  #遗传调度
            # ga_makespan, ga_cost = genetic.schedule(appA)

        IPPS_MGA = MGAScheduler("MGA")  # 遗传调度
        ga_makespan, complete_time = IPPS_MGA.schedule(appA,targetms, targetet, begin_time)
            # ex1:7 10 3 11 8 1 2 9 4 12 13 5 6
                # 2 1 4 1 3 2 2 4 5 3 5 1 3
            # evolution=EvolutionScheduler("Evolution")  #差分进化调度
            # de_makespan, de_cost = evolution.schedule(appC)
            # wolf = wolfScheduler("wolf")  # 灰狼调度
            # GWO_makespan, GWO_cost = wolf.schedule(appC)
            # ASDE_GWO = ASDE_GWOcheduler("ASDE_GWO")  # 退火差分灰狼调度
            # ASDE_GWO_makespan, ASDE_GWO_cost = ASDE_GWO.schedule(appC)
            # posh = POSHScheduler("POSH")
            # posh_makespan, posh_cost = posh.schedule(appD)
            # end_time = datetime.now()
            # print("-" * 100)
            # print("<br/>%s ---- %s<br/>" % (start_time.strftime('%Y-%m-%d %H:%M:%S %f'), end_time.strftime('%Y-%m-%d %H:%M:%S %f')))
            # print("%s seconds" % (end_time - start_time).seconds)

        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
        tracemalloc.stop()
        end_time = time()
        run_time = end_time - begin_time
        print(run_time)

        processor_list = ['8', '16', '32', '64', '128', '256', '512']
        slr_list_ga = [6.32, 6.02, 5.21, 4.46, 3.36, 2.66, 2.03]
        slr_list_posh = [6.54, 6.34, 5.73, 5.11, 4.30, 3.27, 2.64]
        mcr_list_ga = [1.56, 2.37, 3.46, 4.08, 4.82, 5.38, 5.61]
        mcr_list_posh = [1.66, 2.98, 4.17, 4.84, 5.91, 6.20, 6.66]

        '''
            颜色  color:修改颜色，可以简写成c
            样式  linestyle='--' 修改线条的样式 可以简写成 ls
            标注  marker : 标注
            线宽  linewidth: 设置线宽 可以简写成 lw   （lw=2）
    
        '''
        # plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内, 必须放置在 title、xlabel、ylabel 之前才起作用
        # plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内, 必须放置在 title、xlabel、ylabel 之前才起作用
        # plt.xlabel("Alpha")
        # plt.ylabel("TradeOff")
        #
        # plt.plot(processor_list, mcr_list_ga, c='#000000', linestyle='--', marker='o')
        # plt.plot(processor_list, mcr_list_posh, c='#000000', linestyle='-.', marker='>')
        # plt.legend(['NEGA', 'POSH'])
        # plt.savefig("./result/GABUDGET_exp_result_mcr_processor.svg", dpi=600)
        # plt.show()
        plt.rcParams['font.family'] = 'Times New Roman ,SimSun '  # 设置字体族，中文为SimSun，英文为Times New Roman
        e=255
        color = [(244/e,241/e,222/e), (223/e,122/e,94/e), (60/e,64/e,91/e),(130/e,178/e,154/e),(242/e,204/e,142/e)]
        # print("@")
        for k, v in complete_time.items():
            # print("%")
            plt.barh(y=k[1], width=v[2], left=v[0], edgecolor="black", color=color[colors[k[2]%25]])
            plt.text(v[0], k[1]-0.1, k[0], fontsize=7, verticalalignment="center")
            # plt.text(v[0] + 0.2, 2 * k[2] - 0.2, str(v[0]) + " " + str(v[1]), fontdict=fontdict_time)

        my_y_ticks = np.arange(1, processor_number+1, 1)  # 原始数据有13个点，故此处为设置从0开始，间隔为1

        plt.yticks(my_y_ticks)
        plt.xlim(0, ga_makespan+5)
        plt.title("Gantt chart")
        plt.xlabel("makespan")
        plt.ylabel("Machine")
        plt.vlines(ga_makespan,0,processor_number, colors='black',label="makespan="+str(int(ga_makespan)))
        plt.legend(bbox_to_anchor=(0.68, 1.01), loc=3, borderaxespad=0)   #右上角的标记


        ##任务过大时需要取消注释
        plt.gca().margins(x=0)
        plt.gcf().canvas.draw()
        # set size
        maxsize = 6
        m = 0.2
        N = len(complete_time)
        s = maxsize / plt.gcf().dpi * N + 2 * m
        margin = m / plt.gcf().get_size_inches()[0]
        plt.gcf().subplots_adjust(left=margin, right=1. - margin)
        plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])



        plt.savefig("Gantt chart.svg",format='svg')
        plt.show()

    print(ga)
    print(gwo)
    print(pso)
    print(tabu)
    print(sa)


if __name__ == '__main__':
    main()

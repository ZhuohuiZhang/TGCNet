# -*- coding: utf-8 -*-
# @Time    : 2023/7/27 15:13
# @Author  : Zhuohui Zhang
# @File    : __init__.py
# @Software: PyCharm
# @mail    : zhangzh.grey@gmail.com
from .tgc_learner import TGCLearner

REGISTRY = {"tgc_learner": TGCLearner,}


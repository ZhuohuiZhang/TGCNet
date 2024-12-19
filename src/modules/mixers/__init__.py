# -*- coding: utf-8 -*-
# @Time    : 2023/7/27 22:12
# @Author  : Zhuohui Zhang
# @File    : __init__.py
# @Software: PyCharm
# @mail    : zhangzh.grey@gmail.com
from .graph_coarsening import GraphCoarsening

REGISTRY = {"coarsen": GraphCoarsening,}

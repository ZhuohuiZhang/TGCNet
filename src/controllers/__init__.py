# -*- coding: utf-8 -*-
# @Time    : 2023/7/27 15:14
# @Author  : Zhuohui Zhang
# @File    : __init__.py
# @Software: PyCharm
# @mail    : zhangzh.grey@gmail.com
from .tgcnet_controller import TGCNetMAC


REGISTRY = {"tgcnet_mac": TGCNetMAC,}

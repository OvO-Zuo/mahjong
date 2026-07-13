Botzone 推理代码
================

Botzone 设置
------------

游戏:     Chinese-Standard-Mahjong
交互方式: JSON 交互
运行方式: 正常
编译器:   Python 3.6.5

提交步骤
--------

1. 将 model.npz 上传到 Botzone 用户存储
2. 将以下文件打包为 ZIP 提交:
   - __main__.py
   - feature.py
   - model.py
   - agent.py

模型
----

架构: CNNModel (3层Conv2d + FC, 6通道输入 -> 235动作输出)
推理: 确定性 argmax
来源: v15 联赛训练 (s0, 峰值胜率 0.650 @ep2400, 回滚+EMA+BC校准)

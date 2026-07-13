目录结构：
submission/
  研究性报告.pdf       
  README.txt                  本文件
  botzone/                    (1) Botzone 推理代码
    README.txt
    __main__.py               JSON 交互主入口
    feature.py                FeatureAgent (特征提取 + 动作映射)
    model.py                  CNNModel
    agent.py                  MahjongGBAgent 基类
    model.npz                 模型权重 (v15 s0, 峰值胜率 0.650)
  SL/                         (2a) 监督学习预训练
    train_sl.py               SL 训练主脚本
    supervised.py             SL 训练原始入口
    __main__.py               SL 推理入口
    model.py                  CNNModel (策略网络)
    feature.py                FeatureAgent
    agent.py                  MahjongGBAgent 基类
    dataset.py                MahjongGBDataset
    preprocess.py             数据预处理 (data.txt -> .npz)
    model_20.pt               SL 预训练模型 (验证集 84% Top-1 准确率)
  RL/                         (2b) 强化学习精调
    train_final.py            最终增强基线: SL锚定PPO
    __main__.py               多进程 Actor-Learner 入口
    model_var.py              可变通道 CNN (SL兼容)
    model.py                  CNNModel + Value Head
    feature.py                FeatureAgent (235维动作空间)
    env.py                    MahjongGBEnv (4人自弈环境)
    actor.py                  Actor 进程
    learner.py                Learner 进程
    model_pool.py             模型池 (权重共享)
    replay_buffer.py          经验回放缓冲
    agent.py                  Agent 基类
  league/                     (3) 联赛训练系统
    cloud_run.py              v15 — 当前最优 (回滚+EMA+BC校准)
    train_v17.py              v17 — 探索中 (BC权重0.8+三层缓冲)
    league_train_v2.py        v8 — 联赛竞争动态最优 (Elo跨度187)
    league_train_v10.py       v10 — 单调性约束
    league_train_v11.py       v11 — 双头模型
    league_train_v12.py       v12 — PPO+KL+BC+Exploiter
    league_train.py           早期联赛训练
    league_eval.py            联赛评估
    league_model_pool.py      联赛模型池
    league_manager.py         联赛管理器
    elo.py                    Elo 评分系统
    dynamic_sampler.py        动态对手采样
    exploit_detector.py       对抗检测
    drift_detector.py         策略漂移检测
    meta_policy.py            元策略
    dual_head_model.py        双头模型架构
    cloud_refine.py           云端精调 (v16)
    distill.py                知识蒸馏 (在线)
    distill_offline.py        知识蒸馏 (离线)
  experiments/                (4) 附加/消融实验
    run_ablation.py           消融实验主脚本
    run_all_features.py       全特征对比实验
    run_ting_experiment.py    听牌奖励实验
    feature_ablation_final.py 特征消融 (最终版)
    feature_agent_ext.py      扩展特征 Agent
    train_enhanced.py         增强训练 (特征扩展)
    train_optimize.py         超参数优化
    train_suphx.py            Suphx 技术实验
    train_adaptive_entropy.py 自适应熵实验
    cloud_train.py            云端训练 v1
    cloud_train_v2.py         云端训练 v2
    cloud_enhanced.py         云端增强训练
    train.py                  早期 PPO 训练
    __main__.py               RL 多进程入口
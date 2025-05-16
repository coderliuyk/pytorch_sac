#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import time
import pickle as pkl
from video import VideoRecorder
from logger import Logger
from replay_buffer import ReplayBuffer
import utils
import dmc2gym
import hydra
def make_env(cfg):
    """创建dm_control环境的辅助函数"""
    # 根据配置选择环境的领域和任务
    if cfg.env == 'ball_in_cup_catch':
        domain_name = 'ball_in_cup'
        task_name = 'catch'
    else:
        domain_name = cfg.env.split('_')[0]  # 从环境名称中提取领域
        task_name = '_'.join(cfg.env.split('_')[1:])  # 提取任务名称
    
    # 创建环境并设置随机种子
    env = dmc2gym.make(domain_name=domain_name,
                       task_name=task_name,
                       seed=cfg.seed,
                       visualize_reward=True)
    env.seed(cfg.seed)  # 设置环境的随机种子
    assert env.action_space.low.min() >= -1  # 确保动作空间的下限
    assert env.action_space.high.max() <= 1  # 确保动作空间的上限
    return env  # 返回创建的环境
class Workspace(object):
    """工作空间类，负责管理训练过程中的各种组件和状态"""
    def __init__(self, cfg):
        self.work_dir = os.getcwd()  # 获取当前工作目录
        print(f'workspace: {self.work_dir}')  # 打印工作空间
        self.cfg = cfg  # 保存配置
        # 初始化日志记录器
        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             agent=cfg.agent.name)
        utils.set_seed_everywhere(cfg.seed)  # 设置随机种子
        self.device = torch.device(cfg.device)  # 选择设备（CPU或GPU）
        self.env = utils.make_env(cfg)  # 创建环境
        
        # 配置代理的观察和动作空间
        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        
        # 实例化代理
        self.agent = hydra.utils.instantiate(cfg.agent)
        # 创建重放缓冲区
        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          int(cfg.replay_buffer_capacity),
                                          self.device)
        # 创建视频记录器
        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)
        self.step = 0  # 初始化步骤计数
    def evaluate(self):
        """评估代理性能"""
        average_episode_reward = 0  # 初始化平均奖励
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()  # 重置环境
            self.agent.reset()  # 重置代理
            self.video_recorder.init(enabled=(episode == 0))  # 初始化视频记录器
            done = False  # 初始化完成标志
            episode_reward = 0  # 初始化当前回合奖励
            while not done:
                with utils.eval_mode(self.agent):  # 设置代理为评估模式
                    action = self.agent.act(obs, sample=False)  # 选择动作
                obs, reward, done, _ = self.env.step(action)  # 执行动作
                self.video_recorder.record(self.env)  # 记录视频
                episode_reward += reward  # 累加奖励
            
            average_episode_reward += episode_reward  # 累加回合奖励
            self.video_recorder.save(f'{self.step}.mp4')  # 保存视频
            
        average_episode_reward /= self.cfg.num_eval_episodes  # 计算平均奖励
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)  # 记录评估结果
        self.logger.dump(self.step)  # 保存日志
    def run(self):
        """运行训练循环"""
        episode, episode_reward, done = 0, 0, True  # 初始化回合、奖励和完成标志
        start_time = time.time()  # 记录开始时间
        while self.step < self.cfg.num_train_steps:  # 当步骤小于训练步骤时循环
            if done:  # 如果回合结束
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)  # 记录训练持续时间
                    start_time = time.time()  # 重置开始时间
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))  # 保存日志
                # 定期评估代理
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)  # 记录评估回合
                    self.evaluate()  # 评估代理
                self.logger.log('train/episode_reward', episode_reward,
                                self.step)  # 记录训练回合奖励
                obs = self.env.reset()  # 重置环境
                self.agent.reset()  # 重置代理
                done = False  # 重置完成标志
                episode_reward = 0  # 重置当前回合奖励
                episode_step = 0  # 初始化回合步骤
                episode += 1  # 增加回合计数
                self.logger.log('train/episode', episode, self.step)  # 记录训练回合
            # 从环境中随机采样动作以进行数据收集
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()  # 随机选择动作
            else:
                with utils.eval_mode(self.agent):  # 设置代理为评估模式
                    action = self.agent.act(obs, sample=True)  # 选择动作
            
            # 执行训练更新
            if self.step >= self.cfg.num_seed_steps:
                self.agent.update(self.replay_buffer, self.logger, self.step)  # 更新代理
            
            next_obs, reward, done, _ = self.env.step(action)  # 执行动作
            # 允许无限引导
            done = float(done)  # 将完成标志转换为浮点数
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done  # 处理最大步骤
            episode_reward += reward  # 累加奖励
            self.replay_buffer.add(obs, action, reward, next_obs, done,
                                   done_no_max)  # 将经验添加到重放缓冲区
            obs = next_obs  # 更新观察值
            episode_step += 1  # 增加回合步骤
            self.step += 1  # 增加步骤计数
@hydra.main(config_path='config/train.yaml', strict=True)
def main(cfg):
    """主函数，初始化工作空间并运行训练"""
    workspace = Workspace(cfg)  # 创建工作空间
    workspace.run()  # 运行训练
if __name__ == '__main__':
    main()  # 运行主函数

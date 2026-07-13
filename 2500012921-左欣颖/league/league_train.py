"""
League-based Self-Play Training Orchestrator.

Replaces starter_code/RL/train.py with league-aware training:
- Dynamic opponent sampling from league model pool
- Drift detection + KL-constrained rollback
- Periodic league evaluation (every 200ep)
- Exploit detection & exploiter model training
- Meta-policy strategy selection
- Elo-based strength tracking
- Historical model snapshotting

Usage:
    python -m starter_code.league.league_train
"""

import os
import sys
import time
import json
import signal
import torch
import numpy as np

# Ensure RL modules are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'RL'))

from model import CNNModel
from feature import FeatureAgent
from env import MahjongGBEnv
from replay_buffer import ReplayBuffer
from actor import Actor
from learner import Learner

from elo import EloRanker
from league_model_pool import LeagueModelPool
from league_manager import LeagueManager
from league_eval import LeagueEvaluator


# ====================================================================== #
#  Default Configuration
# ====================================================================== #
DEFAULT_CONFIG = {
    # --- PPO ---
    'replay_buffer_size': 50000,
    'replay_buffer_episode': 400,
    'num_actors': 24,
    'episodes_per_actor': 1000,
    'gamma': 0.98,
    'lambda': 0.95,
    'min_sample': 200,
    'batch_size': 256,
    'epochs': 5,
    'clip': 0.1,
    'lr': 5e-5,
    'value_coeff': 1.0,
    'entropy_coeff': 0.01,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    # --- Model Pool ---
    'model_pool_size': 20,
    'model_pool_name': 'league-model-pool',
    'max_historical': 10,
    'max_exploiters': 5,
    'eval_interval': 200,
    'eval_games_per_matchup': 30,
    'historical_snapshot_interval': 100,

    # --- Drift Detection ---
    'kl_threshold': 0.5,
    'kl_window': 10,
    'rollback_cooldown': 50,

    # --- Exploit Detection ---
    'exploit_window': 200,
    'exploit_train_episodes': 500,

    # --- Elo ---
    'initial_elo': 1500.0,
    'elo_k_factor': 32.0,

    # --- Checkpointing ---
    'ckpt_save_interval': 300,
    'ckpt_save_path': os.path.join(os.path.dirname(__file__), 'league_checkpoints'),
    'league_state_path': os.path.join(os.path.dirname(__file__), 'league_state.json'),

    # --- SL Anchor ---
    'sl_model_path': os.path.join(os.path.dirname(__file__), '..', 'SL', 'model', 'checkpoint', 'model_20.pt'),
}

# ====================================================================== #
#  Main Training Loop
# ====================================================================== #
def train_league(config=None):
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    os.makedirs(cfg['ckpt_save_path'], exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Initialize League Components
    # ------------------------------------------------------------------ #
    print('[League] Initializing...')

    # Shared replay buffer
    replay_buffer = ReplayBuffer(
        cfg['replay_buffer_size'],
        cfg['replay_buffer_episode']
    )

    # League model pool
    league_pool = LeagueModelPool(
        max_historical=cfg['max_historical'],
        max_exploiters=cfg['max_exploiters'],
        save_dir=os.path.join(cfg['ckpt_save_path'], 'league_models'),
    )

    # Elo ranker
    elo_ranker = EloRanker(
        initial_elo=cfg['initial_elo'],
        k_factor=cfg['elo_k_factor'],
    )

    # Load SL model for anchor
    sl_path = cfg['sl_model_path']
    if not os.path.exists(sl_path):
        print(f'[League] Warning: SL model not found at {sl_path}')
        sl_path = None

    # League manager
    manager = LeagueManager(cfg, league_pool, elo_ranker, sl_model_path=sl_path)

    # Create initial RL model
    current_model = CNNModel()
    if cfg['device'] == 'cuda':
        current_model = current_model.cuda()

    # Initialize league with current model
    rl_id = manager.initialize(current_model.state_dict())
    print(f'[League] Initialized RL active model: {rl_id}')

    # League evaluator
    evaluator = LeagueEvaluator(manager, cfg)

    # ------------------------------------------------------------------ #
    #  Spawn Actors and Learner
    # ------------------------------------------------------------------ #
    print(f'[League] Spawning {cfg["num_actors"]} actors + 1 learner...')

    actors = []
    for i in range(cfg['num_actors']):
        cfg['name'] = f'Actor-{i}'
        actor = Actor(cfg, replay_buffer)
        actors.append(actor)

    learner = Learner(cfg, replay_buffer)

    # Signal handling
    stop_requested = {'value': False}

    def _request_stop(signum, frame):
        if not stop_requested['value']:
            print(f'[League] Received signal {signum}, shutting down...')
        stop_requested['value'] = True

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    # ------------------------------------------------------------------ #
    #  Training Loop
    # ------------------------------------------------------------------ #
    started_actors = []
    learner_started = False

    try:
        for actor in actors:
            actor.start()
            started_actors.append(actor)
        learner.start()
        learner_started = True
        print('[League] All processes started. Training...')

        last_ckpt_time = time.time()
        last_league_eval_time = time.time()
        eval_interval_seconds = cfg.get('eval_interval_seconds', 120)

        while True:
            # Check actor liveness
            alive_actors = [a for a in started_actors if a.is_alive()]
            if not alive_actors:
                print('[League] All actors finished.')
                break

            for actor in alive_actors:
                actor.join(timeout=0.5)

            if stop_requested['value']:
                raise KeyboardInterrupt

            current_ep = manager.current_ep + 1
            manager.current_ep = current_ep

            # ---- Periodic League Evaluation (time-based) ----
            if time.time() - last_league_eval_time >= eval_interval_seconds:
                print(f'\n[League] === Evaluation at ep {current_ep} ===')
                eval_results = evaluator.evaluate_all(current_model, current_ep)
                print(f'[League] Eval: avg_hu={eval_results["summary"]["avg_hu_rate"]:.3f}')

                # Update best model
                manager.update_best(
                    current_model.state_dict(),
                    eval_results['summary']['avg_hu_rate'],
                    current_ep
                )

                # Update champion
                manager.update_champion()

                # Snapshot historical
                manager.pool.add_historical(current_model.state_dict(), {
                    'ep': current_ep,
                    'hu_rate': eval_results['summary']['avg_hu_rate'],
                })

                # Drift check
                should_rollback, drift_info = manager.check_drift(current_model, current_ep)
                if should_rollback:
                    print(f'[League] DRIFT DETECTED (KL={drift_info["mean_kl"]:.3f}), rolling back!')
                    manager.maybe_rollback(current_model)

                # Exploit check
                exploit_report = manager.exploit.get_weakness_report()
                if exploit_report['significant_weakness']:
                    print(f'[League] Weakness detected (hu={exploit_report["recent_hu_rate"]:.2f}), '
                          f'considering exploiter training...')

                # Log
                manager.league_log.append({
                    'ep': current_ep,
                    'eval': eval_results['summary'],
                    'drift': drift_info,
                    'elo': elo_ranker.get_elo(rl_id),
                })

                last_league_eval_time = time.time()

            # ---- Periodic Checkpoint ----
            if time.time() - last_ckpt_time >= cfg['ckpt_save_interval']:
                ckpt_path = os.path.join(
                    cfg['ckpt_save_path'], f'league_ep{current_ep}.pt'
                )
                torch.save({
                    'model_state_dict': current_model.state_dict(),
                    'ep': current_ep,
                    'elo': elo_ranker.get_elo(rl_id),
                    'league_stats': manager.get_league_stats(),
                }, ckpt_path)
                manager.save_state(cfg['league_state_path'])
                print(f'[League] Checkpoint saved: {ckpt_path}')
                last_ckpt_time = time.time()

            # Check if maximum episodes reached
            max_ep = cfg.get('max_total_episodes', cfg['num_actors'] * cfg['episodes_per_actor'])
            if current_ep >= max_ep:
                print(f'[League] Reached max episodes ({max_ep}). Finishing.')
                break

        # ---- Final: stop learner ----
        learner.stop()
        while learner.is_alive():
            learner.join(timeout=1)
            if stop_requested['value']:
                break

    except KeyboardInterrupt:
        print('[League] Interrupted. Shutting down...')
    finally:
        # ---- Graceful shutdown ----
        for actor in started_actors:
            actor.stop()

        # Wait for actors
        deadline = time.time() + 5
        for actor in started_actors:
            remaining = deadline - time.time()
            if remaining > 0:
                actor.join(timeout=remaining)
            if actor.is_alive():
                actor.terminate()

        # Wait for learner
        if learner_started and learner.is_alive():
            learner.stop()
            learner.join(timeout=30)
            if learner.is_alive():
                learner.terminate()

        # Cleanup
        for actor in started_actors:
            try:
                actor.close()
            except Exception:
                pass
        try:
            learner.close()
        except Exception:
            pass
        replay_buffer.close()

        # Final save
        print('[League] Saving final state...')
        manager.save_state(cfg['league_state_path'])
        torch.save({
            'model_state_dict': current_model.state_dict(),
            'ep': manager.current_ep,
            'league_stats': manager.get_league_stats(),
        }, os.path.join(cfg['ckpt_save_path'], 'league_final.pt'))

        print('[League] Training complete.')
        print(json.dumps(manager.get_league_stats(), indent=2, default=str))


# ====================================================================== #
#  Entry Point
# ====================================================================== #
if __name__ == '__main__':
    train_league()

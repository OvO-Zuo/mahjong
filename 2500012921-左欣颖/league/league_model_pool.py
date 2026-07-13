"""
League Model Pool — enhanced version with model categories.

Categories:
  - sl:        Supervised-learning base model (frozen anchor)
  - rl_active: Currently training RL model
  - best:      Best-performing model so far
  - historical:Snapshot of past models for opponent diversity
  - exploiter: Models trained to exploit specific opponents
  - champion:  Current strongest model (highest Elo or eval score)
"""

import os
import torch
import pickle
import time


class LeagueModelPool:
    def __init__(self, max_historical=10, max_exploiters=5,
                 save_dir=None):
        self.max_historical = max_historical
        self.max_exploiters = max_exploiters
        self.save_dir = save_dir or os.path.join(os.path.dirname(__file__), 'league_models')

        # Storage: category -> list of (model_id, state_dict, metadata) tuples
        self._models = {
            'sl':          [],
            'rl_active':   [],
            'best':        [],
            'historical':  [],
            'exploiter':   [],
            'champion':    [],
        }

        # Model ID counter
        self._counter = 0

        os.makedirs(self.save_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def add_sl(self, state_dict, metadata=None):
        """Register the SL anchor model."""
        self._models['sl'] = [self._make_entry('sl', 0, state_dict, metadata)]

    def get_sl(self):
        """Return state_dict of SL model, or None."""
        entries = self._models['sl']
        return entries[0]['state_dict'] if entries else None

    def add_rl_active(self, state_dict, metadata=None):
        """Update the currently-training RL model (only one at a time)."""
        model_id = self._next_id()
        self._models['rl_active'] = [self._make_entry('rl_active', model_id, state_dict, metadata)]
        return model_id

    def get_rl_active(self):
        entries = self._models['rl_active']
        return entries[0] if entries else None

    def add_best(self, state_dict, metadata=None):
        """Replace best model (only keep one)."""
        model_id = self._next_id()
        # Move old best to historical
        old = self._models['best']
        if old:
            self._add_historical(old[0]['state_dict'], old[0]['metadata'])
        self._models['best'] = [self._make_entry('best', model_id, state_dict, metadata)]
        return model_id

    def get_best(self):
        entries = self._models['best']
        return entries[0] if entries else None

    def add_historical(self, state_dict, metadata=None):
        """Snapshot current policy into historical pool."""
        self._add_historical(state_dict, metadata)

    def get_historical(self):
        return self._models['historical']

    def add_exploiter(self, target_id, state_dict, metadata=None):
        """Add an exploiter model targeting a specific opponent."""
        model_id = self._next_id()
        meta = metadata or {}
        meta['target_id'] = target_id
        entry = self._make_entry('exploiter', model_id, state_dict, meta)
        self._models['exploiter'].append(entry)
        # Trim excess
        if len(self._models['exploiter']) > self.max_exploiters:
            removed = self._models['exploiter'].pop(0)
            self._remove_disk(removed)
        return model_id

    def get_exploiters(self):
        return self._models['exploiter']

    def set_champion(self, state_dict, metadata=None):
        model_id = self._next_id()
        self._models['champion'] = [self._make_entry('champion', model_id, state_dict, metadata)]
        return model_id

    def get_champion(self):
        entries = self._models['champion']
        return entries[0] if entries else None

    def get_all_opponents(self):
        """Return all models usable as opponents for self-play."""
        opponents = []
        for cat in ['historical', 'exploiter', 'best', 'sl']:
            for entry in self._models[cat]:
                opponents.append(entry)
        return opponents

    def get_opponent_ids(self):
        return [e['model_id'] for e in self.get_all_opponents()]

    def save_to_disk(self):
        """Persist all models to disk."""
        for cat, entries in self._models.items():
            for entry in entries:
                path = self._model_path(entry['model_id'])
                torch.save({
                    'state_dict': entry['state_dict'],
                    'metadata': entry['metadata'],
                    'category': cat,
                }, path)

    def load_from_disk(self):
        """Restore models from disk."""
        if not os.path.isdir(self.save_dir):
            return
        for fname in sorted(os.listdir(self.save_dir)):
            if not fname.endswith('.pt'):
                continue
            path = os.path.join(self.save_dir, fname)
            data = torch.load(path, map_location='cpu', weights_only=False)
            cat = data.get('category', 'historical')
            sd = data['state_dict']
            meta = data.get('metadata', {})
            model_id = meta.get('model_id', 0)
            self._models.setdefault(cat, []).append(
                self._make_entry(cat, model_id, sd, meta)
            )
            self._counter = max(self._counter, model_id + 1)

    # ------------------------------------------------------------------ #
    #  Internal
    # ------------------------------------------------------------------ #
    def _next_id(self):
        self._counter += 1
        return self._counter

    def _make_entry(self, category, model_id, state_dict, metadata):
        return {
            'category': category,
            'model_id': model_id,
            'state_dict': {k: v.detach().cpu().clone() for k, v in state_dict.items()},
            'metadata': metadata or {},
            'created_at': time.time(),
        }

    def _add_historical(self, state_dict, metadata=None):
        model_id = self._next_id()
        self._models['historical'].append(
            self._make_entry('historical', model_id, state_dict, metadata)
        )
        # Trim excess
        while len(self._models['historical']) > self.max_historical:
            removed = self._models['historical'].pop(0)
            self._remove_disk(removed)

    def _remove_disk(self, entry):
        path = self._model_path(entry['model_id'])
        if os.path.exists(path):
            os.remove(path)

    def _model_path(self, model_id):
        return os.path.join(self.save_dir, f'model_{model_id}.pt')

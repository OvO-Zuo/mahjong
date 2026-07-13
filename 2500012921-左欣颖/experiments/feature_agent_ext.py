"""Extended FeatureAgent with configurable observation channels."""
from agent import MahjongGBAgent
from collections import defaultdict
import numpy as np

try:
    from MahjongGB import MahjongFanCalculator
except ImportError:
    print('MahjongGB library required!')
    raise

TILE_LIST = [
    *('W%d' % (i+1) for i in range(9)),
    *('T%d' % (i+1) for i in range(9)),
    *('B%d' % (i+1) for i in range(9)),
    *('F%d' % (i+1) for i in range(4)),
    *('J%d' % (i+1) for i in range(3)),
]
OFFSET_TILE = {c: i for i, c in enumerate(TILE_LIST)}

# --- Feature set definitions ---
# Each feature set defines additional channels beyond the base 6
# Format: dict of channel_name -> {offset, description}

BASE_OBS_CHANNELS = 6  # seat_wind, prevalent_wind, hand×4

FEATURE_SETS = {
    'baseline': {
        'obs_size': 6,
        'extra_channels': {},
    },
    'key': {
        'obs_size': 15,
        'extra_channels': {
            'DISCARD_0': 6,   # player 0 discard pool (1 channel)
            'DISCARD_1': 7,   # player 1 discard pool
            'DISCARD_2': 8,   # player 2 discard pool
            'DISCARD_3': 9,   # player 3 discard pool
            'MELD_0': 10,     # player 0 visible melds
            'MELD_1': 11,     # player 1 visible melds
            'MELD_2': 12,     # player 2 visible melds
            'MELD_3': 13,     # player 3 visible melds
            'SHOWN_TILES': 14,  # global shown tile count (normalized)
        },
    },
    'full': {
        'obs_size': 21,
        'extra_channels': {
            'DISCARD_0': 6, 'DISCARD_1': 7, 'DISCARD_2': 8, 'DISCARD_3': 9,
            'MELD_0': 10, 'MELD_1': 11, 'MELD_2': 12, 'MELD_3': 13,
            'SHOWN_TILES': 14,
            'OPP_HAND_1': 15,  # training-only: opponent 1 hand
            'OPP_HAND_2': 16,  # training-only: opponent 2 hand
            'OPP_HAND_3': 17,  # training-only: opponent 3 hand
            'WALL_REMAIN': 18, # remaining tiles in wall
            'PHASE': 19,       # game phase indicator
            'TING_INDICATOR': 20,  # ting indicator (training-only)
        },
    },
}

# --- Action space ---
ACT_SIZE = 235
OFFSET_ACT = {
    'Pass': 0, 'Hu': 1, 'Play': 2, 'Chi': 36,
    'Peng': 99, 'Gang': 133, 'AnGang': 167, 'BuGang': 201,
}


def make_agent_cls(obs_size):
    """Factory: create an agent class with fixed obs_size for env compatibility."""
    class _Agent(ExtendedFeatureAgent):
        observation_space = None
        action_space = None
        def __init__(self, seatWind):
            super().__init__(seatWind, obs_size)
    return _Agent


class ExtendedFeatureAgent(MahjongGBAgent):
    """Feature agent with configurable observation size.
    Use make_agent_cls(obs_size) to get an env-compatible class."""

    def __init__(self, seatWind, obs_size=6):
        self.seatWind = seatWind
        self.obs_size = obs_size
        self._inner_init()

    def _inner_init(self):
        self.packs = [[] for _ in range(4)]
        self.history = [[] for _ in range(4)]
        self.tileWall = [21] * 4
        self.shownTiles = defaultdict(int)
        self.wallLast = False
        self.isAboutKong = False
        self.obs = np.zeros((self.obs_size, 36))

        # Base features
        self.obs[0][OFFSET_TILE['F%d' % (self.seatWind + 1)]] = 1  # seat wind

        # Cache for training-only opponent hands (set externally by env)
        self._opponent_hands = {}

    def set_opponent_hands(self, hands_dict):
        """Set opponent hands for training-only complete information."""
        self._opponent_hands = hands_dict

    def request2obs(self, request):
        t = request.split()
        if t[0] == 'Wind':
            self.prevalentWind = int(t[1])
            self.obs[1][OFFSET_TILE['F%d' % (self.prevalentWind + 1)]] = 1
            return
        if t[0] == 'Deal':
            self.hand = t[1:]
            self._hand_embedding_update()
            return
        if t[0] == 'Huang':
            self.valid = []
            return self._obs()
        if t[0] == 'Draw':
            self.tileWall[0] -= 1
            self.wallLast = self.tileWall[1] == 0
            tile = t[1]
            self.valid = []
            if self._check_mahjong(tile, isSelfDrawn=True, isAboutKong=self.isAboutKong):
                self.valid.append(OFFSET_ACT['Hu'])
            self.isAboutKong = False
            self.hand.append(tile)
            self._hand_embedding_update()
            for tile_ in set(self.hand):
                self.valid.append(OFFSET_ACT['Play'] + OFFSET_TILE[tile_])
                if self.hand.count(tile_) == 4 and not self.wallLast and self.tileWall[0] > 0:
                    self.valid.append(OFFSET_ACT['AnGang'] + OFFSET_TILE[tile_])
            if not self.wallLast and self.tileWall[0] > 0:
                for packType, tile_, offer in self.packs[0]:
                    if packType == 'PENG' and tile_ in self.hand:
                        self.valid.append(OFFSET_ACT['BuGang'] + OFFSET_TILE[tile_])
            return self._obs()
        # Player N action notification
        p = (int(t[1]) + 4 - self.seatWind) % 4
        if t[2] == 'Draw':
            self.tileWall[p] -= 1
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            return
        if t[2] in ('Invalid', 'Hu'):
            self.valid = []
            return self._obs()
        if t[2] == 'Play':
            self.tileFrom = p
            self.curTile = t[3]
            self.shownTiles[self.curTile] += 1
            self.history[p].append(self.curTile)
            if p == 0:
                self.hand.remove(self.curTile)
                self._hand_embedding_update()
                return
            else:
                self.valid = []
                if self._check_mahjong(self.curTile):
                    self.valid.append(OFFSET_ACT['Hu'])
                if not self.wallLast:
                    if self.hand.count(self.curTile) >= 2:
                        self.valid.append(OFFSET_ACT['Peng'] + OFFSET_TILE[self.curTile])
                        if self.hand.count(self.curTile) == 3 and self.tileWall[0]:
                            self.valid.append(OFFSET_ACT['Gang'] + OFFSET_TILE[self.curTile])
                    color = self.curTile[0]
                    if p == 3 and color in 'WTB':
                        num = int(self.curTile[1])
                        tmp = [color + str(num + i) for i in range(-2, 3)]
                        if tmp[0] in self.hand and tmp[1] in self.hand:
                            self.valid.append(OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 3) * 3 + 2)
                        if tmp[1] in self.hand and tmp[3] in self.hand:
                            self.valid.append(OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 2) * 3 + 1)
                        if tmp[3] in self.hand and tmp[4] in self.hand:
                            self.valid.append(OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 1) * 3)
                self.valid.append(OFFSET_ACT['Pass'])
                return self._obs()
        if t[2] == 'Chi':
            tile = t[3]
            color, num = tile[0], int(tile[1])
            self.packs[p].append(('CHI', tile, int(self.curTile[1]) - num + 2))
            self.shownTiles[self.curTile] -= 1
            for i in range(-1, 2):
                self.shownTiles[color + str(num + i)] += 1
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            if p == 0:
                self.valid = []
                self.hand.append(self.curTile)
                for i in range(-1, 2):
                    self.hand.remove(color + str(num + i))
                self._hand_embedding_update()
                for tile_ in set(self.hand):
                    self.valid.append(OFFSET_ACT['Play'] + OFFSET_TILE[tile_])
                return self._obs()
            return
        if t[2] in ('UnChi', 'UnPeng'):
            # Undo operations — handle minimally
            return
        if t[2] == 'Peng':
            self.packs[p].append(('PENG', self.curTile, (4 + p - self.tileFrom) % 4))
            self.shownTiles[self.curTile] += 2
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            if p == 0:
                self.valid = []
                for _ in range(2):
                    self.hand.remove(self.curTile)
                self._hand_embedding_update()
                for tile_ in set(self.hand):
                    self.valid.append(OFFSET_ACT['Play'] + OFFSET_TILE[tile_])
                return self._obs()
            return
        if t[2] == 'Gang':
            self.packs[p].append(('GANG', self.curTile, (4 + p - self.tileFrom) % 4))
            self.shownTiles[self.curTile] += 3
            if p == 0:
                for _ in range(3):
                    self.hand.remove(self.curTile)
                self._hand_embedding_update()
                self.isAboutKong = True
            return
        if t[2] == 'AnGang':
            tile = 'CONCEALED' if p else t[3]
            self.packs[p].append(('GANG', tile, 0))
            if p == 0:
                self.isAboutKong = True
                for _ in range(4):
                    self.hand.remove(tile)
            else:
                self.isAboutKong = False
            return
        if t[2] == 'BuGang':
            tile = t[3]
            for i in range(len(self.packs[p])):
                if tile == self.packs[p][i][1]:
                    self.packs[p][i] = ('GANG', tile, self.packs[p][i][2])
                    break
            self.shownTiles[tile] += 1
            if p == 0:
                self.hand.remove(tile)
                self._hand_embedding_update()
                self.isAboutKong = True
                return
            else:
                self.valid = []
                if self._check_mahjong(tile, isSelfDrawn=False, isAboutKong=True):
                    self.valid.append(OFFSET_ACT['Hu'])
                self.valid.append(OFFSET_ACT['Pass'])
                return self._obs()
        raise NotImplementedError('Unknown request %s!' % request)

    def action2response(self, action):
        if action < OFFSET_ACT['Hu']: return 'Pass'
        if action < OFFSET_ACT['Play']: return 'Hu'
        if action < OFFSET_ACT['Chi']:
            return 'Play ' + TILE_LIST[action - OFFSET_ACT['Play']]
        if action < OFFSET_ACT['Peng']:
            t = (action - OFFSET_ACT['Chi']) // 3
            return 'Chi ' + 'WTB'[t // 7] + str(t % 7 + 2)
        if action < OFFSET_ACT['Gang']: return 'Peng'
        if action < OFFSET_ACT['AnGang']: return 'Gang'
        if action < OFFSET_ACT['BuGang']:
            return 'Gang ' + TILE_LIST[action - OFFSET_ACT['AnGang']]
        return 'BuGang ' + TILE_LIST[action - OFFSET_ACT['BuGang']]

    def response2action(self, response):
        t = response.split()
        if t[0] == 'Pass': return OFFSET_ACT['Pass']
        if t[0] == 'Hu': return OFFSET_ACT['Hu']
        if t[0] == 'Play': return OFFSET_ACT['Play'] + OFFSET_TILE[t[1]]
        if t[0] == 'Chi':
            return (OFFSET_ACT['Chi'] + 'WTB'.index(t[1][0]) * 7 * 3 +
                    (int(t[2][1]) - 2) * 3 + int(t[1][1]) - int(t[2][1]) + 1)
        if t[0] == 'Peng': return OFFSET_ACT['Peng'] + OFFSET_TILE[t[1]]
        if t[0] == 'Gang': return OFFSET_ACT['Gang'] + OFFSET_TILE[t[1]]
        if t[0] == 'AnGang': return OFFSET_ACT['AnGang'] + OFFSET_TILE[t[1]]
        if t[0] == 'BuGang': return OFFSET_ACT['BuGang'] + OFFSET_TILE[t[1]]
        return OFFSET_ACT['Pass']

    def _obs(self):
        # Fill extra feature channels
        self._fill_extra_features()
        mask = np.zeros(ACT_SIZE)
        for a in self.valid:
            mask[a] = 1
        return {
            'observation': self.obs.reshape((self.obs_size, 4, 9)).copy(),
            'action_mask': mask,
        }

    def _hand_embedding_update(self):
        self.obs[2:] = 0
        d = defaultdict(int)
        for tile in self.hand:
            d[tile] += 1
        for tile, cnt in d.items():
            self.obs[2:2 + cnt, OFFSET_TILE[tile]] = 1

    def _fill_extra_features(self):
        """Fill extra feature channels beyond base 6."""
        if self.obs_size < 7:
            return
        # Discard history per player (channels 6-9)
        if self.obs_size >= 10:
            for pi in range(4):
                ch = 6 + pi
                self.obs[ch] = 0
                for tile in self.history[pi]:
                    self.obs[ch][OFFSET_TILE[tile]] = min(1.0, self.obs[ch][OFFSET_TILE[tile]] + 0.1)

        # Melds per player (channels 10-13)
        if self.obs_size >= 14:
            for pi in range(4):
                ch = 10 + pi
                self.obs[ch] = 0
                for packType, tile, _ in self.packs[pi]:
                    idx = OFFSET_TILE.get(tile)
                    if idx is not None:
                        self.obs[ch][idx] = 1

        # Global shown tiles (channel 14)
        if self.obs_size >= 15:
            max_count = 4
            self.obs[14] = 0
            for tile, cnt in self.shownTiles.items():
                idx = OFFSET_TILE.get(tile)
                if idx is not None:
                    self.obs[14][idx] = min(cnt / max_count, 1.0)

        # Opponent hands (channels 15-17) — training only
        if self.obs_size >= 18:
            for opp_idx, ch in [(1, 15), (2, 16), (3, 17)]:
                self.obs[ch] = 0
                if opp_idx in self._opponent_hands:
                    for tile in self._opponent_hands[opp_idx]:
                        idx = OFFSET_TILE.get(tile)
                        if idx is not None:
                            self.obs[ch][idx] = 1

        # Wall remaining (channel 18)
        if self.obs_size >= 19:
            remaining = sum(self.tileWall)
            self.obs[18].fill(min(remaining / 144.0, 1.0))

        # Phase indicator (channel 19)
        if self.obs_size >= 20:
            self.obs[19] = 0
            if self.valid and OFFSET_ACT['Hu'] in self.valid:
                self.obs[19].fill(1.0)  # can hu → late game

        # Ting indicator (channel 20) — training only
        if self.obs_size >= 21:
            self.obs[20] = 0
            # Check if any discard would lead to ting
            if len(self.hand) > 0 and self.valid:
                has_ting = any(
                    a >= OFFSET_ACT['Hu'] and a < OFFSET_ACT['Play']
                    for a in self.valid[:5]
                )
                if has_ting:
                    self.obs[20].fill(1.0)

    def _check_mahjong(self, winTile, isSelfDrawn=False, isAboutKong=False):
        try:
            fans = MahjongFanCalculator(
                pack=tuple(self.packs[0]),
                hand=tuple(self.hand),
                winTile=winTile,
                flowerCount=0,
                isSelfDrawn=isSelfDrawn,
                is4thTile=(self.shownTiles[winTile] + isSelfDrawn) == 4,
                isAboutKong=isAboutKong,
                isWallLast=self.wallLast,
                seatWind=self.seatWind,
                prevalentWind=self.prevalentWind,
                verbose=True,
            )
            fanCnt = sum(fanPoint * cnt for fanPoint, cnt, _, _ in fans)
            if fanCnt < 8:
                raise Exception('Not Enough Fans')
        except Exception:
            return False
        return True

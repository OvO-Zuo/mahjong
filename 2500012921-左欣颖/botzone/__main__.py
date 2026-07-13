"""
Botzone submission - JSON interaction mode.
Receives JSON input with full history, outputs JSON response with debug field.
"""
import sys
import os
import json
import traceback as _tb

try:
    from feature import FeatureAgent
    from model import CNNModel
    import numpy as np
    import torch
except Exception as _e:
    print(json.dumps({"response": "PASS", "debug": "IMPORT_ERROR: %s" % _e}))
    sys.exit(0)

# ========== MODEL LOADING ==========
def load_model(path):
    model = CNNModel()
    if path.endswith('.npz'):
        data = np.load(path)
        sd = {k: torch.from_numpy(data[k]) for k in data.files}
    else:
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location='cpu')
        sd = ckpt.get('model', ckpt)
        sd = {k: v for k, v in sd.items() if not k.startswith('_value_branch')}
    model.load_state_dict(sd)
    model.eval()
    return model


def obs2action(model, agent, obs):
    obs_t = torch.from_numpy(np.expand_dims(obs['observation'], 0))
    mask_t = torch.from_numpy(np.expand_dims(obs['action_mask'], 0))
    logits = model({'is_training': False, 'obs': {'observation': obs_t, 'action_mask': mask_t}})
    return int(logits.detach().numpy().flatten().argmax())


def replay_history(agent, requests, responses):
    """Replay all past requests/responses to build agent state."""
    last_draw_player = -1
    last_draw_tile = None

    for i, (req, resp) in enumerate(zip(requests, responses)):
        t = req.split()
        if t[0] == '0':
            pass  # init already handled
        elif t[0] == '1':
            hand_tiles = [x for x in t[5:18] if not x.startswith('H')]
            agent.request2obs(' '.join(['Deal', *hand_tiles]))
        elif t[0] == '2':
            # Bot draws: tile = t[1]
            last_draw_player = agent.seatWind
            last_draw_tile = t[1]
            agent.request2obs('Draw %s' % t[1])
        elif t[0] == '3':
            p = int(t[1])
            if t[2] == 'DRAW':
                last_draw_player = p
                last_draw_tile = None
                agent.request2obs('Player %d Draw' % p)
            elif t[2] == 'BUHUA':
                agent.request2obs('Player %d BUHUA' % p)
            elif t[2] == 'GANG':
                # Determine if AnGang: same player just drew
                if p == last_draw_player and last_draw_tile:
                    agent.request2obs('Player %d AnGang %s' % (p, last_draw_tile))
                elif p == last_draw_player:
                    agent.request2obs('Player %d AnGang' % p)
                else:
                    agent.request2obs('Player %d Gang' % p)
                last_draw_player = -1
            elif t[2] == 'BUGANG':
                agent.request2obs('Player %d BuGang %s' % (p, t[3]))
            elif t[2] in ('HU', 'INVALID'):
                agent.request2obs('Player %d %s' % (p, t[2]))
            else:
                # PLAY / CHI / PENG → clear draw tracking
                last_draw_player = -1
                if t[2] == 'CHI':
                    agent.request2obs('Player %d Chi %s' % (p, t[3]))
                elif t[2] == 'PENG':
                    agent.request2obs('Player %d Peng' % p)
                agent.request2obs('Player %d Play %s' % (p, t[-1]))


# ========== MAIN ==========
def main():
    debug_lines = []
    try:
        # Load model
        candidates = ['model.npz', 'model.pt', '/data/model.npz', '/data/model.pt']
        model_path = None
        for c in candidates:
            if os.path.exists(c):
                model_path = c
                break
        if model_path is None:
            raise FileNotFoundError('model not found: %s' % candidates)

        model = load_model(model_path)
        debug_lines.append('model=%s' % model_path)

        # Read JSON input
        raw = sys.stdin.read()
        input_json = json.loads(raw)
        requests = input_json['requests']
        responses = input_json['responses']
        current_request = requests[-1]
        debug_lines.append('turn=%d req=%s' % (len(responses), current_request[:100]))

        # Init agent from first request
        t0 = requests[0].split()
        seatWind = int(t0[1])
        agent = FeatureAgent(seatWind)
        agent.request2obs('Wind %s' % t0[2])
        debug_lines.append('seat=%d quan=%s' % (seatWind, t0[2]))

        # Replay history (all past requests/responses)
        replay_history(agent, requests[:-1], responses)

        # Handle current request
        t = current_request.split()
        resp = 'PASS'

        if t[0] == '0':
            seatWind = int(t[1])
            agent = FeatureAgent(seatWind)
            agent.request2obs('Wind %s' % t[2])
            debug_lines.append('new_game seat=%d' % seatWind)

        elif t[0] == '1':
            hand_tiles = [x for x in t[5:18] if not x.startswith('H')]
            agent.request2obs(' '.join(['Deal', *hand_tiles]))
            debug_lines.append('deal hand=%d' % len(hand_tiles))

        elif t[0] == '2':
            obs = agent.request2obs('Draw %s' % t[1])
            if obs is not None:
                action = obs2action(model, agent, obs)
                r = agent.action2response(action)
                rt = r.split()
                debug_lines.append('draw=%s action=%d resp=%s' % (t[1], action, r))
                if rt[0] == 'Hu': resp = 'HU'
                elif rt[0] == 'Play': resp = 'PLAY %s' % rt[1]
                elif rt[0] == 'Gang': resp = 'GANG %s' % rt[1]
                elif rt[0] == 'BuGang': resp = 'BUGANG %s' % rt[1]

        elif t[0] == '3':
            p = int(t[1])
            if t[2] == 'DRAW':
                agent.request2obs('Player %d Draw' % p)
            elif t[2] == 'BUHUA':
                agent.request2obs('Player %d BUHUA' % p)
            elif t[2] == 'GANG':
                agent.request2obs('Player %d Gang' % p)
            elif t[2] == 'BUGANG':
                obs = agent.request2obs('Player %d BuGang %s' % (p, t[3]))
                if p != seatWind and obs is not None:
                    action = obs2action(model, agent, obs)
                    r = agent.action2response(action)
                    if r == 'Hu': resp = 'HU'
                    debug_lines.append('bugang p=%d action=%d resp=%s' % (p, action, r))
            elif t[2] == 'HU' or t[2] == 'INVALID':
                agent.request2obs('Player %d %s' % (p, t[2]))
            else:
                # PLAY / CHI / PENG
                if t[2] == 'CHI':
                    agent.request2obs('Player %d Chi %s' % (p, t[3]))
                elif t[2] == 'PENG':
                    agent.request2obs('Player %d Peng' % p)
                obs = agent.request2obs('Player %d Play %s' % (p, t[-1]))
                if p != seatWind and obs is not None:
                    action = obs2action(model, agent, obs)
                    r = agent.action2response(action)
                    rt = r.split()
                    debug_lines.append('play p=%d action=%d resp=%s' % (p, action, r))
                    if rt[0] == 'Hu': resp = 'HU'
                    elif rt[0] == 'Pass': resp = 'PASS'
                    elif rt[0] == 'Gang': resp = 'GANG'
                    elif rt[0] in ('Peng', 'Chi'):
                        chi_obs = agent.request2obs('Player %d ' % seatWind + r)
                        if chi_obs is not None:
                            action2 = obs2action(model, agent, chi_obs)
                            r2 = agent.action2response(action2)
                            resp = ' '.join([rt[0].upper(), *rt[1:], r2.split()[-1]])
                        else:
                            resp = ' '.join([rt[0].upper(), *rt[1:]])
                        debug_lines.append('combined=%s' % resp)

        output = {'response': resp, 'debug': ' | '.join(debug_lines)}
        print(json.dumps(output))

    except Exception as _e:
        output = {
            'response': 'PASS',
            'debug': 'ERROR: %s\n%s' % (_e, _tb.format_exc())
        }
        print(json.dumps(output))


if __name__ == '__main__':
    main()

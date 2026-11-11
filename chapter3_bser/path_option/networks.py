"""Causal history, ordered variable-length routes, and candidate-aware policies."""
import math
import numpy as np
import torch
from torch import nn
from chapter3_bser.path_option.path_context import PATH_DIM


def tensor_batch(rows, intervention='none'):
    lengths = [len(p) for r in rows for p in r['paths']]
    size = max(lengths)
    paths = np.zeros((len(rows), 8, size, PATH_DIM), np.float32)
    valid = np.zeros((len(rows), 8, size), bool)
    for b, r in enumerate(rows):
        for j, p in enumerate(r['paths']):
            paths[b, j, :len(p)] = p
            valid[b, j, :len(p)] = True
    batch = {k: torch.as_tensor(np.stack([r[k] for r in rows])) for k in
             ('history', 'goals', 'actions', 'action_mask', 'goal_mask', 'event_relative')}
    batch.update(paths=torch.tensor(paths), path_mask=torch.tensor(valid))
    if intervention == 'no_path':
        batch['paths'].zero_()
        batch['path_mask'].zero_()
        batch['path_mask'][:, :, 0] = True
    if intervention == 'no_history':
        batch['history'] = batch['history'][:, -1:].expand_as(batch['history']).clone()
    return batch


def masked_distribution(logits, mask):
    if not bool(mask.any(-1).all()):
        raise ValueError('empty distribution mask')
    return torch.softmax(logits.masked_fill(~mask, -1e9), -1)


class Encoder(nn.Module):
    def __init__(self, o, variant):
        super().__init__()
        d = o['width']
        self.variant = variant
        self.frame = nn.Linear(90, d)
        self.goal = nn.Linear(19, d)
        self.point = nn.Sequential(nn.Linear(PATH_DIM, d), nn.ReLU(), nn.Linear(d, d))
        self.position = nn.Parameter(torch.randn(1, o['history'], d)*.01)
        if variant == 'B':
            self.temporal = nn.Sequential(nn.Linear(o['history']*90, 2*d), nn.ReLU(), nn.Linear(2*d, d))
        else:
            layer = nn.TransformerEncoderLayer(d, o['heads'], 2*d, dropout=0., batch_first=True)
            self.temporal = nn.TransformerEncoder(layer, o['layers'], enable_nested_tensor=False)
            self.cross = nn.MultiheadAttention(d, o['heads'], dropout=0., batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, d)*.01)
        self.fusion = nn.Sequential(nn.Linear(3*d, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU())

    def history_encoding(self, history):
        if self.variant == 'B':
            return self.temporal(history.flatten(1))
        x = self.frame(history)+self.position[:, :history.shape[1]]
        causal = torch.ones(x.shape[1], x.shape[1], dtype=torch.bool).triu(1)
        return self.temporal(x, mask=causal)

    def forward(self, b):
        h = self.history_encoding(b['history'])
        if h.ndim == 3:
            h = h[:, -1]
        n, g, l, _ = b['paths'].shape
        p = self.point(b['paths']).reshape(n*g, l, -1)
        mask = b['path_mask'].reshape(n*g, l)
        goal = self.goal(b['goals']).reshape(n*g, -1)
        history = h[:, None].expand(-1, g, -1).reshape(n*g, -1)
        if self.variant == 'B':
            # Every ordered vertex is supplied; arc length/order are explicit token fields.
            context = (p*mask[..., None]).sum(1)/mask.sum(1, keepdim=True)
        else:
            q = self.query.expand(n*g, -1, -1) if self.variant == 'fixed_query' else (history+goal)[:, None]
            context = self.cross(q, p, p, key_padding_mask=~mask, need_weights=False)[0][:, 0]
        return self.fusion(torch.cat((history, goal, context), -1)).reshape(n, g, -1)


class Policy(nn.Module):
    def __init__(self, o, variant):
        super().__init__()
        d = o['width']
        self.encoder = Encoder(o, variant)
        self.action = nn.Sequential(nn.Linear(3, d), nn.ReLU(), nn.Linear(d, d))
        self.goal = nn.Linear(d, 1)
        self.termination = nn.Sequential(nn.Linear(d+2, d), nn.ReLU(), nn.Linear(d, 1))
        nn.init.constant_(self.termination[-1].bias, -3.)

    def forward(self, b, events):
        z = self.encoder(b)
        a = self.action(b['actions'])
        logits = torch.einsum('bgd,bad->bga', z, a)/math.sqrt(z.shape[-1])
        low = masked_distribution(logits, b['action_mask'][:, None].expand_as(logits))
        high = self.goal(z).squeeze(-1)
        beta = torch.sigmoid(self.termination(torch.cat((z, events.detach()), -1))).squeeze(-1)
        return low, high, beta


class Critic(nn.Module):
    def __init__(self, o, variant):
        super().__init__()
        d = o['width']
        self.encoder = Encoder(o, variant)
        self.q = nn.Sequential(nn.Linear(d+3, d), nn.ReLU(), nn.Linear(d, 1))
        self.events = nn.Sequential(nn.Linear(d+6, d), nn.ReLU(), nn.Linear(d, 3*o['horizon']))
        with torch.no_grad():
            self.events[-1].bias.reshape(-1, 3)[:] = torch.tensor([-4., -4., 0.])

    def forward(self, b, actions=None, event_relative=None):
        z = self.encoder(b)
        a = b['actions'] if actions is None else actions
        if a.ndim == 3:
            a = a[:, None].expand(-1, z.shape[1], -1, -1)
        zz = z[:, :, None].expand(-1, -1, a.shape[2], -1)
        query = b['event_relative'] if event_relative is None else event_relative
        query = query[:, :, None].expand(-1, -1, a.shape[2], -1)
        q = self.q(torch.cat((zz, a), -1)).squeeze(-1)
        hazard = self.events(torch.cat((zz, a, query), -1))
        return q, hazard

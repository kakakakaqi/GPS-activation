"""Benchmark models and datasets for the paper's cells.

Extracted verbatim from run_bench2.py so the entry scripts in tests/ carry only
the hyperparameters.  Registers on import:

  * DatasetRegistry: charlm (the byte corpus), wikitext, openorca, calhousing
  * ArchitectureRegistry: charlm, charlm_glu, regmlp (rn18 alias for resnet18)

CORPUS_MAX selects the corpus tier of the text cells; the non-vision runner sets
it per job to produce the 12/50/100 MB rows of tab:accs.
"""
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from gps_lab import ActivationRegistry, ArchitectureRegistry, DatasetRegistry

# ---------------------------------------------------------------- char corpus
def build_corpus(max_bytes=2_000_000):
    """Byte corpus for the `charlm` cell: markdown + source text from the tree.

    Set GPS_CHARLM_DOCS to an extra directory of .md files to widen the corpus
    (the study additionally pooled a local documentation tree).
    """
    root = Path('.')
    docs = os.environ.get('GPS_CHARLM_DOCS')
    files = (list(Path(docs).rglob('*.md')) if docs else []) + \
            list(root.glob('report/*.md')) + list(root.glob('*.tex')) + \
            list(root.glob('*.py'))
    buf = bytearray()
    for f in files:
        try:
            buf.extend(f.read_bytes())
        except Exception:
            continue
        if len(buf) >= max_bytes:
            break
    return bytes(buf[:max_bytes])


class CharDataset(torch.utils.data.Dataset):
    def __init__(self, data, seq=256):
        self.data = torch.frombuffer(bytearray(data), dtype=torch.uint8).long()
        self.seq = seq
        self.n = max(0, (len(self.data) - 1) // seq)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        j = i * self.seq                      # non-overlapping chunks
        x = self.data[j:j + self.seq]
        y = self.data[j + 1:j + self.seq + 1]
        return x, y


CORPUS_MAX = int(os.environ.get('GPS_CORPUS_BYTES', 12_000_000))


def build_corpus_wiki(max_bytes=None):
    max_bytes = max_bytes or CORPUS_MAX
    f = Path('data/datasets/wikitext-103/wikitext-103/wiki.train.tokens')
    return f.read_bytes()[:max_bytes]


def build_corpus_orca(max_bytes=None):
    max_bytes = max_bytes or CORPUS_MAX
    import pyarrow.parquet as pq
    t = pq.read_table('data/datasets/1M-GPT4-Augmented.parquet', columns=['question', 'response'])
    buf = bytearray()
    for q, r in zip(t.column('question').to_pylist(), t.column('response').to_pylist()):
        buf.extend((q + '\n' + r + '\n').encode())
        if len(buf) >= max_bytes:
            break
    return bytes(buf[:max_bytes])


def _make_charlm_registry(builder):
    @classmethod
    def _f(cls, batch_size=64, data_dir='./data', seq=256, **kwargs):
        corpus = builder()
        cut = int(len(corpus) * 0.95)
        tr = CharDataset(corpus[:cut], seq)
        te = CharDataset(corpus[cut:], seq)

        def collate(batch):
            x = torch.stack([b[0] for b in batch])
            y = torch.stack([b[1] for b in batch]).reshape(-1)
            return x, y

        pin = torch.cuda.is_available()
        trl = torch.utils.data.DataLoader(tr, batch_size=batch_size, shuffle=True,
                                          num_workers=2, pin_memory=pin, collate_fn=collate)
        tel = torch.utils.data.DataLoader(te, batch_size=batch_size, shuffle=False,
                                          num_workers=2, pin_memory=pin, collate_fn=collate)
        return trl, tel, 1, 256
    return _f


@classmethod
def _calhousing(cls, batch_size=256, data_dir='./data', **kw):
    import numpy as np
    z = np.load('data/datasets/calhousing.npz')
    def mk(X, y, sh):
        ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=sh, num_workers=2)
    return mk(z['X_train'], z['y_train'], True), mk(z['X_test'], z['y_test'], False), 8, 1


DatasetRegistry.calhousing = _calhousing
DatasetRegistry.wikitext = _make_charlm_registry(build_corpus_wiki)
DatasetRegistry.openorca = _make_charlm_registry(build_corpus_orca)


@classmethod
def _agnews_like(cls, batch_size=64, data_dir='./data', seq=256, **kwargs):
    corpus = build_corpus()
    cut = int(len(corpus) * 0.95)
    tr = CharDataset(corpus[:cut], seq)
    te = CharDataset(corpus[cut:], seq)

    def collate(batch):
        x = torch.stack([b[0] for b in batch])          # (B, T)
        y = torch.stack([b[1] for b in batch]).reshape(-1)  # (B*T,) -> deterministic CE
        return x, y

    pin = torch.cuda.is_available()
    trl = torch.utils.data.DataLoader(tr, batch_size=batch_size, shuffle=True,
                                      num_workers=2, pin_memory=pin, collate_fn=collate)
    tel = torch.utils.data.DataLoader(te, batch_size=batch_size, shuffle=False,
                                      num_workers=2, pin_memory=pin, collate_fn=collate)
    return trl, tel, 1, 256


DatasetRegistry.charlm = _agnews_like


# ---------------------------------------------------------------- char LM model
class CharBlock(nn.Module):
    def __init__(self, dim, heads, act):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.act = act

    def forward(self, x, mask):
        h = self.n1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        h = self.n2(x)
        return x + self.fc2(self.act(self.fc1(h)))


class CharLM(nn.Module):
    def __init__(self, activation=('gelu', {}), act_kwargs=None, num_classes=256,
                 dim=192, depth=4, heads=3, seq=256, in_channels=1, **kw):
        super().__init__()
        if isinstance(activation, (tuple, list)):
            name, a_kw = activation[0], dict(activation[1] or {})
        else:
            name, a_kw = activation, {}
        a_kw.update(act_kwargs or {})
        mk = lambda: ActivationRegistry.get(name, **a_kw)   # per-block instance
        self.emb = nn.Embedding(num_classes, dim)
        self.pos = nn.Parameter(torch.zeros(1, seq, dim))
        self.blocks = nn.ModuleList([CharBlock(dim, heads, mk()) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        mask = torch.full((seq, seq), float('-inf'))
        self.register_buffer('mask', torch.triu(mask, diagonal=1))

    def forward(self, x):
        x = self.emb(x) + self.pos[:, :x.size(1)]
        for b in self.blocks:
            x = b(x, self.mask[:x.size(1), :x.size(1)])
        return self.head(self.norm(x)).reshape(-1, 256)   # (B*T, V) with flattened targets


class CharBlockGLU(nn.Module):
    """Transformer block with a gated (SwiGLU-style) FFN; the gate uses the
    tested activation: Swish = canonical SwiGLU; mixchoice/naive2p = GPS-GLU."""

    def __init__(self, dim, heads, act):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.w_gate = nn.Linear(dim, dim * 4)
        self.w_up = nn.Linear(dim, dim * 4)
        self.w_down = nn.Linear(dim * 4, dim)
        self.act = act

    def forward(self, x, mask):
        h = self.n1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        h = self.n2(x)
        return x + self.w_down(self.act(self.w_gate(h)) * self.w_up(h))


class CharLMGLU(nn.Module):
    def __init__(self, activation=('gelu', {}), act_kwargs=None, num_classes=256,
                 dim=192, depth=4, heads=3, seq=256, in_channels=1, **kw):
        super().__init__()
        if isinstance(activation, (tuple, list)):
            name, a_kw = activation[0], dict(activation[1] or {})
        else:
            name, a_kw = activation, {}
        a_kw.update(act_kwargs or {})
        mk = lambda: ActivationRegistry.get(name, **a_kw)   # per-block gate
        self.emb = nn.Embedding(num_classes, dim)
        self.pos = nn.Parameter(torch.zeros(1, seq, dim))
        self.blocks = nn.ModuleList([CharBlockGLU(dim, heads, mk()) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        mask = torch.full((seq, seq), float('-inf'))
        self.register_buffer('mask', torch.triu(mask, diagonal=1))

    def forward(self, x):
        x = self.emb(x) + self.pos[:, :x.size(1)]
        for b in self.blocks:
            x = b(x, self.mask[:x.size(1), :x.size(1)])
        return self.head(self.norm(x)).reshape(-1, 256)


class RegMLP(nn.Module):
    def __init__(self, activation=('gelu', {}), act_kwargs=None, num_classes=1,
                 in_channels=8, hidden=256, depth=3, **kw):
        super().__init__()
        if isinstance(activation, (tuple, list)):
            name, a_kw = activation[0], dict(activation[1] or {})
        else:
            name, a_kw = activation, {}
        a_kw.update(act_kwargs or {})
        layers, d = [], in_channels
        for _ in range(depth):
            layers += [nn.Linear(d, hidden),
                       ActivationRegistry.get(name, **a_kw)]   # per-layer instance
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).reshape(-1)


ArchitectureRegistry.MODELS['regmlp'] = RegMLP
ArchitectureRegistry.MODELS['charlm'] = CharLM
ArchitectureRegistry.MODELS['charlm_glu'] = CharLMGLU
ArchitectureRegistry.MODELS['rn18'] = ArchitectureRegistry.MODELS['resnet18']

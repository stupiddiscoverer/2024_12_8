#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AlphaZero for Gomoku (Five-in-a-Row) — Single File, PyTorch

Features
- 15x15 Gomoku by default (configurable)
- Self-play with MCTS + PUCT
- Human vs. AI; human games can be recorded and added to the training buffer
- Replay buffer with on-disk persistence (optional)
- Lightweight CNN policy+value network
- Symmetry augmentation (rotations + flips)

Quickstart
---------
1) Train from scratch with self-play:
   python gomoku_az.py --train --total-iters 20 --games-per-iter 20

2) Play against the current model (loads checkpoint if present):
   python gomoku_az.py --play-ai --mcts-sims 400

3) Play and also record your game for training:
   python gomoku_az.py --play-ai --record-human --train-after-human 2

4) Continue training using existing buffer/model:
   python gomoku_az.py --train --resume

Files written
- model.pt : latest model weights
- buffer.pkl : replay buffer (optional)

Notes
- Requires: Python 3.9+, PyTorch, NumPy.
- This is a compact educational implementation; not heavily optimized.
"""

import argparse
import math
import os
import pickle
import random
import sys
import time
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# -------------------------------
# Game (Gomoku)
# -------------------------------

class Gomoku:
    def __init__(self, size: int = 15, n_to_win: int = 5):
        self.size = size
        self.n_to_win = n_to_win

    def new_board(self):
        # board: 0 empty, 1 black, -1 white; current_player: 1 or -1 (black starts)
        board = np.zeros((self.size, self.size), dtype=np.int8)
        current_player = 1
        return board, current_player

    def legal_moves(self, board: np.ndarray) -> np.ndarray:
        return (board.reshape(-1) == 0).astype(np.uint8)

    def step(self, board: np.ndarray, player: int, move: int) -> Tuple[np.ndarray, int]:
        # move is flat index [0, size*size)
        x, y = divmod(move, self.size)
        assert board[x, y] == 0, "Illegal move"
        board2 = board.copy()
        board2[x, y] = player
        return board2, -player

    def check_winner(self, board: np.ndarray) -> Optional[int]:
        # returns 1 or -1 if that player has a 5-in-a-row, 0 for draw, None for ongoing
        s = self.size
        n = self.n_to_win
        b = board
        # quick check: if board has fewer than (2*n-1) stones, impossible to have winner early? (skip)

        # horizontal & vertical
        for i in range(s):
            for j in range(s - n + 1):
                row = b[i, j:j+n]
                if abs(row.sum()) == n and len(set(row)) == 1 and row[0] != 0:
                    return int(row[0])
                col = b[j:j+n, i]
                if abs(col.sum()) == n and len(set(col)) == 1 and col[0] != 0:
                    return int(col[0])
        # diagonals
        for i in range(s - n + 1):
            for j in range(s - n + 1):
                diag1 = [b[i+k, j+k] for k in range(n)]
                if abs(sum(diag1)) == n and len(set(diag1)) == 1 and diag1[0] != 0:
                    return int(diag1[0])
                diag2 = [b[i+n-1-k, j+k] for k in range(n)]
                if abs(sum(diag2)) == n and len(set(diag2)) == 1 and diag2[0] != 0:
                    return int(diag2[0])
        # draw?
        if (b == 0).sum() == 0:
            return 0
        return None

    def canonical(self, board: np.ndarray, player: int) -> np.ndarray:
        # from the perspective of current player; own stones -> 1, opp -> -1
        return board.astype(np.int8) * player

    def symmetries(self, board: np.ndarray, pi: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate 8 symmetries (4 rotations x mirror). pi is flat (size*size)."""
        s = self.size
        pi2d = pi.reshape(s, s)
        out = []
        for k in range(4):
            rb = np.rot90(board, k)
            rp = np.rot90(pi2d, k)
            out.append((rb, rp.reshape(-1)))
            fb = np.fliplr(rb)
            fp = np.fliplr(rp)
            out.append((fb, fp.reshape(-1)))
        return out

    # Rendering utilities for CLI play
    def render(self, board: np.ndarray):
        s = self.size
        # header
        sys.stdout.write("    ")
        for y in range(s):
            sys.stdout.write(f"{y:2d} ")
        sys.stdout.write("\n")
        for x in range(s):
            sys.stdout.write(f"{x:2d} ")
            for y in range(s):
                ch = "."
                if board[x, y] == 1:
                    ch = "●"
                elif board[x, y] == -1:
                    ch = "○"
                sys.stdout.write(f" {ch} ")
            sys.stdout.write("\n")
        sys.stdout.flush()

# -------------------------------
# Neural Network (Policy + Value)
# -------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + x)
        return out

class PolicyValueNet(nn.Module):
    def __init__(self, board_size=15, channels=64, n_blocks=5):
        super().__init__()
        self.s = board_size
        self.conv = nn.Conv2d(3, channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(n_blocks)])
        # policy head
        self.p_conv = nn.Conv2d(channels, 2, 1)
        self.p_bn = nn.BatchNorm2d(2)
        self.p_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)
        # value head
        self.v_conv = nn.Conv2d(channels, 1, 1)
        self.v_bn = nn.BatchNorm2d(1)
        self.v_fc1 = nn.Linear(board_size * board_size, 64)
        self.v_fc2 = nn.Linear(64, 1)

    def forward(self, planes):
        # planes: [B, 3, s, s]
        x = F.relu(self.bn(self.conv(planes)))
        for blk in self.blocks:
            x = blk(x)
        # policy
        p = F.relu(self.p_bn(self.p_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.p_fc(p)
        # value
        v = F.relu(self.v_bn(self.v_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.v_fc1(v))
        v = torch.tanh(self.v_fc2(v))
        return p, v.squeeze(-1)

    def make_input(self, board: np.ndarray, player: int) -> torch.Tensor:
        # 3 planes: [current stones, opponent stones, ones for player]
        cur = (board == player).astype(np.float32)
        opp = (board == -player).astype(np.float32)
        ones = np.ones_like(cur, dtype=np.float32)
        planes = np.stack([cur, opp, ones], axis=0)
        return torch.from_numpy(planes)

# -------------------------------
# MCTS (PUCT)
# -------------------------------

@dataclass
class MCTSNode:
    prior: float
    to_play: int
    visits: int = 0
    value_sum: float = 0.0
    children: dict = None  # move -> MCTSNode
    is_expanded: bool = False

    def q(self):
        return 0.0 if self.visits == 0 else self.value_sum / self.visits

class MCTS:
    def __init__(self, game: Gomoku, net: PolicyValueNet, device: str = "cpu", c_puct: float = 1.5, dirichlet_alpha: float = 0.3, dirichlet_frac: float = 0.25):
        self.game = game
        self.net = net
        self.device = device
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_frac = dirichlet_frac

    def search(self, board: np.ndarray, to_play: int, num_sims: int, add_noise: bool, temp: float, legal_mask: np.ndarray) -> np.ndarray:
        root = MCTSNode(prior=1.0, to_play=to_play, children=dict(), is_expanded=False)
        # Expand root
        self._expand(root, board, to_play, legal_mask)
        if add_noise:
            self._add_dirichlet_noise(root, legal_mask)
        for _ in range(num_sims):
            self._simulate(root, board.copy())
        # build policy (visit counts)
        s = self.game.size * self.game.size
        counts = np.zeros(s, dtype=np.float32)
        for m, child in root.children.items():
            counts[m] = child.visits
        if temp <= 1e-6:
            best = np.argmax(counts)
            pi = np.zeros_like(counts)
            pi[best] = 1.0
            return pi
        # softmax over counts^{1/temp}
        counts = counts ** (1.0 / temp)
        if counts.sum() == 0:
            counts += legal_mask.astype(np.float32)
        pi = counts / counts.sum()
        return pi

    def _expand(self, node: MCTSNode, board: np.ndarray, to_play: int, legal_mask: np.ndarray):
        s = self.game.size
        with torch.no_grad():
            inp = self.net.make_input(board, to_play).unsqueeze(0).to(self.device)
            logits, value = self.net(inp)
            logits = logits.squeeze(0).cpu().numpy()
            value = float(value.item())
        # mask illegal
        logits = logits - 1e9 * (1 - legal_mask.astype(np.float32))
        # softmax to get priors
        max_logit = np.max(logits)
        priors = np.exp(logits - max_logit)
        priors = priors / np.sum(priors)
        node.children = {}
        for m in np.where(legal_mask)[0].tolist():
            node.children[m] = MCTSNode(prior=float(priors[m]), to_play=-to_play, children=dict())
        node.is_expanded = True
        node.value_sum = 0.0
        node.visits = 0
        node.to_play = to_play
        node.prior = 1.0
        node._value = value  # store leaf value from current player's perspective

    def _add_dirichlet_noise(self, node: MCTSNode, legal_mask: np.ndarray):
        moves = np.where(legal_mask)[0]
        if len(moves) == 0:
            return
        alpha = self.dirichlet_alpha
        noise = np.random.dirichlet([alpha] * len(moves))
        for idx, m in enumerate(moves):
            node.children[m].prior = node.children[m].prior * (1 - self.dirichlet_frac) + noise[idx] * self.dirichlet_frac

    def _simulate(self, node: MCTSNode, board: np.ndarray) -> float:
        # selection
        path = []
        cur = node
        to_play = node.to_play
        game_over = self.game.check_winner(board)
        while cur.is_expanded and game_over is None:
            legal = self.game.legal_moves(board)
            best_score = -1e18
            best_move = None
            total_sqrt = math.sqrt(max(1, sum(ch.visits for ch in cur.children.values())))
            for m, ch in cur.children.items():
                if legal[m] == 0:
                    continue
                q = ch.q()
                u = self.c_puct * ch.prior * total_sqrt / (1 + ch.visits)
                # if it's opponent's turn in child, value is from that perspective; we handle sign at backup time
                score = q + u
                if score > best_score:
                    best_score = score
                    best_move = m
            if best_move is None:
                break
            path.append((cur, best_move))
            x, y = divmod(best_move, self.game.size)
            board[x, y] = to_play
            to_play = -to_play
            cur = cur.children[best_move]
            game_over = self.game.check_winner(board)
        # evaluation
        if game_over is None:
            legal = self.game.legal_moves(board)
            self._expand(cur, board, to_play, legal)
            value = cur._value
        else:
            if game_over == 0:
                value = 0.0
            else:
                value = 1.0 if game_over == node.to_play else -1.0
        # backup
        for parent, move in reversed(path):
            child = parent.children[move]
            child.visits += 1
            # value from parent's perspective: if child.to_play == parent's.to_play, sign flips accordingly
            # We placed the stone when moving to child, so value is for the parent.to_play at that time:
            child.value_sum += value
            value = -value
        return value

# -------------------------------
# Replay Buffer
# -------------------------------

@dataclass
class Sample:
    planes: np.ndarray  # [3, s, s]
    pi: np.ndarray      # [s*s]
    z: float            # scalar outcome from current player's perspective

class ReplayBuffer:
    def __init__(self, maxlen: int = 100000, persist_path: Optional[str] = None):
        self.buffer = deque(maxlen=maxlen)
        self.persist_path = persist_path

    def add_game(self, samples: List[Sample]):
        self.buffer.extend(samples)

    def sample_batch(self, batch_size: int) -> List[Sample]:
        batch = random.sample(self.buffer, batch_size)
        return batch

    def __len__(self):
        return len(self.buffer)

    def save(self):
        if self.persist_path:
            with open(self.persist_path, 'wb') as f:
                pickle.dump(self.buffer, f)

    def load(self):
        if self.persist_path and os.path.exists(self.persist_path):
            with open(self.persist_path, 'rb') as f:
                data = pickle.load(f)
                self.buffer = deque(data, maxlen=self.buffer.maxlen)

# -------------------------------
# Training & Self-Play
# -------------------------------

def play_self_game(game: Gomoku, mcts: MCTS, net: PolicyValueNet, device: str, mcts_sims: int, temp: float, temp_moves: int) -> List[Sample]:
    board, player = game.new_board()
    samples: List[Sample] = []
    move_count = 0
    while True:
        legal = game.legal_moves(board)
        pi = mcts.search(board.copy(), player, mcts_sims, add_noise=True, temp=(temp if move_count < temp_moves else 1e-6), legal_mask=legal)
        # record sample
        planes = net.make_input(board, player).numpy()
        samples.append(Sample(planes=planes, pi=pi, z=0.0))
        # choose move by pi
        move = np.random.choice(len(pi), p=pi)
        board, player = game.step(board, player, move)
        winner = game.check_winner(board)
        if winner is not None:
            # assign outcomes
            z = 0.0 if winner == 0 else 1.0
            cur = 1
            for i in range(len(samples)):
                samples[i].z = z * cur
                cur = -cur
            return samples
        move_count += 1


def train_net(net: PolicyValueNet, buffer: ReplayBuffer, device: str, batch_size: int, epochs: int, lr: float, weight_decay: float):
    if len(buffer) < batch_size:
        return 0.0, 0.0
    net.train()
    opt = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    total_pl, total_vl, steps = 0.0, 0.0, 0
    for _ in range(epochs):
        batch = buffer.sample_batch(batch_size)
        planes = torch.from_numpy(np.stack([s.planes for s in batch])).to(device)
        target_pi = torch.from_numpy(np.stack([s.pi for s in batch])).to(device)
        target_z = torch.from_numpy(np.array([s.z for s in batch], dtype=np.float32)).to(device)
        logits, v = net(planes)
        # policy loss (cross-entropy with target pi)
        log_probs = F.log_softmax(logits, dim=1)
        pl = -(target_pi * log_probs).sum(dim=1).mean()
        # value loss (MSE)
        vl = F.mse_loss(v, target_z)
        loss = pl + vl
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        total_pl += float(pl.item())
        total_vl += float(vl.item())
        steps += 1
    return total_pl / max(1, steps), total_vl / max(1, steps)

# -------------------------------
# Human vs AI
# -------------------------------

def human_vs_ai(game: Gomoku, mcts: MCTS, net: PolicyValueNet, device: str, mcts_sims: int, record: bool = False) -> Optional[List[Sample]]:
    board, player = game.new_board()
    samples: List[Sample] = []
    print("You are ○ (white). AI is ● (black). Black moves first.")
    while True:
        if player == 1:
            # AI turn
            legal = game.legal_moves(board)
            pi = mcts.search(board.copy(), player, mcts_sims, add_noise=False, temp=1e-6, legal_mask=legal)
            if record:
                planes = net.make_input(board, player).numpy()
                samples.append(Sample(planes=planes, pi=pi, z=0.0))
            move = int(np.argmax(pi))
            board, player = game.step(board, 1, move)
            print("AI moves to:", divmod(move, game.size))
        else:
            # Human turn
            game.render(board)
            while True:
                try:
                    raw = input("Your move as 'x y' (row col), or 'q' to quit: ").strip()
                    if raw.lower() == 'q':
                        print("Quit.")
                        return None
                    x, y = map(int, raw.split())
                    if not (0 <= x < game.size and 0 <= y < game.size):
                        print("Out of range.")
                        continue
                    if board[x, y] != 0:
                        print("Cell occupied.")
                        continue
                    move = x * game.size + y
                    break
                except Exception:
                    print("Invalid input.")
            if record:
                planes = net.make_input(board, player).numpy()
                # create a near-deterministic pi for the human move
                pi = np.zeros(game.size * game.size, dtype=np.float32)
                pi[move] = 1.0
                samples.append(Sample(planes=planes, pi=pi, z=0.0))
            board, player = game.step(board, -1, move)
        winner = game.check_winner(board)
        if winner is not None:
            game.render(board)
            if winner == 1:
                print("AI (black) wins.")
            elif winner == -1:
                print("You (white) win!")
            else:
                print("Draw.")
            if record:
                z = 0.0 if winner == 0 else 1.0
                cur = 1
                for i in range(len(samples)):
                    samples[i].z = z * cur
                    cur = -cur
                return samples
            return None

# -------------------------------
# Utils
# -------------------------------

def save_model(net: PolicyValueNet, path: str):
    torch.save(net.state_dict(), path)

def load_model(net: PolicyValueNet, path: str, device: str):
    sd = torch.load(path, map_location=device)
    net.load_state_dict(sd)

# -------------------------------
# Main
# -------------------------------

def main():
    p = argparse.ArgumentParser()
    # Game
    p.add_argument('--size', type=int, default=15)
    p.add_argument('--n-to-win', type=int, default=5)
    # Self-play / MCTS
    p.add_argument('--mcts-sims', type=int, default=200)
    p.add_argument('--c-puct', type=float, default=1.5)
    p.add_argument('--temp', type=float, default=1.0)
    p.add_argument('--temp-moves', type=int, default=20, help='use temperature for first N moves, then argmax')
    # Training
    p.add_argument('--train', action='store_true')
    p.add_argument('--total-iters', type=int, default=10)
    p.add_argument('--games-per-iter', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--buffer-size', type=int, default=50000)
    p.add_argument('--save-buffer', action='store_true')
    p.add_argument('--resume', action='store_true', help='load model/ buffer if present')
    # Play
    p.add_argument('--play-ai', action='store_true')
    p.add_argument('--record-human', action='store_true')
    p.add_argument('--train-after-human', type=int, default=0, help='epochs to train after recording a human game')
    # IO
    p.add_argument('--model', type=str, default='model.pt')
    p.add_argument('--buffer', type=str, default='buffer.pkl')
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    game = Gomoku(size=args.size, n_to_win=args.n_to_win)
    net = PolicyValueNet(board_size=args.size).to(device)

    if args.resume and os.path.exists(args.model):
        print("Loading model from", args.model)
        load_model(net, args.model, device)

    rb = ReplayBuffer(maxlen=args.buffer_size, persist_path=(args.buffer if args.save_buffer else None))
    if args.resume and args.save_buffer:
        rb.load()
        print(f"Loaded buffer with {len(rb)} samples")

    mcts = MCTS(game, net, device=device, c_puct=args.c_puct)

    if args.train:
        for it in range(1, args.total_iters + 1):
            t0 = time.time()
            games = []
            for g in range(args.games_per_iter):
                samples = play_self_game(game, mcts, net, device, args.mcts_sims, args.temp, args.temp_moves)
                # symmetry augmentation
                aug = []
                for s in samples:
                    b = s.planes[0] - s.planes[1]  # recover canonical board (1 for cur, -1 opp)
                    board = np.where(b > 0.5, 1, np.where(b < -0.5, -1, 0)).astype(np.int8)
                    pi = s.pi
                    for sb, spi in game.symmetries(board, pi):
                        planes = net.make_input(sb, 1).numpy()
                        aug.append(Sample(planes=planes, pi=spi.astype(np.float32), z=s.z))
                rb.add_game(aug)
            pl, vl = train_net(net, rb, device, args.batch_size, args.epochs, args.lr, args.weight_decay)
            save_model(net, args.model)
            if args.save_buffer:
                rb.save()
            dt = time.time() - t0
            print(f"Iter {it}/{args.total_iters} | buffer {len(rb)} | policy_loss {pl:.4f} | value_loss {vl:.4f} | {dt:.1f}s")

    if args.play_ai:
        if os.path.exists(args.model):
            load_model(net, args.model, device)
        else:
            print("No model checkpoint found; playing with randomly initialized network.")
        rec = args.record_human
        samples = human_vs_ai(game, mcts, net, device, args.mcts_sims, record=rec)
        if samples is not None and rec:
            # augment and add to buffer
            aug = []
            for s in samples:
                b = s.planes[0] - s.planes[1]
                board = np.where(b > 0.5, 1, np.where(b < -0.5, -1, 0)).astype(np.int8)
                pi = s.pi
                for sb, spi in game.symmetries(board, pi):
                    planes = net.make_input(sb, 1).numpy()
                    aug.append(Sample(planes=planes, pi=spi.astype(np.float32), z=s.z))
            rb.add_game(aug)
            print(f"Recorded human game. Buffer now has {len(rb)} samples.")
            if args.train_after_human > 0:
                pl, vl = train_net(net, rb, device, args.batch_size, args.train_after_human, args.lr, args.weight_decay)
                save_model(net, args.model)
                if args.save_buffer:
                    rb.save()
                print(f"Trained after human game: policy_loss {pl:.4f} | value_loss {vl:.4f}")

if __name__ == '__main__':
    main()

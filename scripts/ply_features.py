#!/usr/bin/env python3
from __future__ import annotations

"""
Feature encoding for a single ply (move) into 64 binary planes.

Definitions
===========

Presence (geometric, EMPTY BOARD):
    Squares a piece on a given square *could* capture on if the board
    were otherwise empty and there were an enemy piece there:

        * Pawns: forward diagonals (no pushes).
        * Knights: L-jumps.
        * Bishops: full diagonals to board edge (ignoring blockers).
        * Rooks: full ranks/files to board edge (ignoring blockers).
        * Queens: bishop + rook rays (ignoring blockers).
        * Kings: adjacent squares (no castling squares by default).

Control (any_control, CURRENT BOARD):
    Squares a piece *attacks* on the current board:

        * Implemented as chess.Board.attacks(sq) for each piece.
        * Includes squares with own pieces (defended squares).
        * Includes squares with enemy pieces (attacked squares).
        * Includes pinned pieces (they still "attack" in this sense).

Legal control (legal_control, CURRENT BOARD):
    Squares a piece effectively controls *after applying pin constraints*:

        * Start from any_control.
        * If the piece is NOT absolutely pinned → legal_control = any_control.
        * If the piece IS absolutely pinned (board.is_pinned):
            - For sliders and pawns: only keep squares along the pin
              line (mask from board.pin).
            - For knights: pinned knights lose all legal control
              (cannot move off the line).
        * Still includes own pieces as "defended" squares.
        * Does *not* try to model "side is currently in check" beyond
          the pin logic – this is a deliberate simplification.

Net control (pawn-weighted):
    For each square sq, define

        score(sq) = 1{W legal_control(sq)} - 1{B legal_control(sq)}
                    + 1{W pawn_legal_control(sq)} - 1{B pawn_legal_control(sq)}

    Then:
        * If score > 0 → white_net_control(sq) = 1, black_net_control(sq) = 0.
        * If score < 0 → black_net_control(sq) = 1, white_net_control(sq) = 0.
        * If score = 0 → both 0 (neutral).

Plane layout (64 total, all 0/1)
================================

  0-11   : board_pre  per piece-type x color
            (6 piece types × 2 colors = 12 planes)

  12-23  : presence_pre per piece-type x color
            (empty-board capture reach)

  24-35  : control_pre  per piece-type x color
            (any_control: attack/defense on current board)

  36-41  : aggregates (pre-move):
              36: white_any_control_pre
              37: black_any_control_pre
              38: white_legal_control_pre
              39: black_legal_control_pre
              40: white_net_control_pre
              41: black_net_control_pre

  42-47  : aggregates (post-move, same semantics):
              42: white_any_control_post
              43: black_any_control_post
              44: white_legal_control_post
              45: black_legal_control_post
              46: white_net_control_post
              47: black_net_control_post

  48-51  : move-square planes:
              48: from-square of the *current* move (one-hot)
              49: to-square of the *current* move (one-hot)
              50: prev_from_square  (one-hot: previous ply's from-square; all 0 if none)
              51: prev_to_square    (one-hot: previous ply's to-square; all 0 if none)

  52-57  : rule-state planes (pre-move):
              52: pre_white_can_castle_k  (all 1s if White can castle king-side, else all 0)
              53: pre_white_can_castle_q  (all 1s if White can castle queen-side)
              54: pre_black_can_castle_k
              55: pre_black_can_castle_q
              56: pre_side_to_move        (all 1s if it is White to move, else 0)
              57: pre_en_passant_target   (one-hot: 1 at ep target square if any, else all 0)

  58-63  : rule-state planes (post-move):
              58: post_white_can_castle_k
              59: post_white_can_castle_q
              60: post_black_can_castle_k
              61: post_black_can_castle_q
              62: post_side_to_move       (all 1s if it is White to move after the move)
              63: post_en_passant_target  (one-hot ep target after the move)

Returned shape: (64, 8, 8) with [plane, rank, file], rank 0 = rank 1 (a1..h1).
"""

from typing import Dict, Tuple

import numpy as np
import chess

# ---------------------------------------------------------------------
# Constants & plane layout
# ---------------------------------------------------------------------

NUM_SQUARES = 64
NUM_PLANES = 64

PIECE_TYPES = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
]
PIECE_TYPE_TO_INDEX = {pt: i for i, pt in enumerate(PIECE_TYPES)}
NUM_PIECE_TYPES = len(PIECE_TYPES)

COLORS = [chess.WHITE, chess.BLACK]

# Plane ranges (0–47 unchanged)
BOARD_PRE_BASE = 0            # 0-11
PRESENCE_PRE_BASE = 12        # 12-23
CONTROL_PRE_BASE = 24         # 24-35

AGG_PRE_BASE = 36             # 36-41
AGG_POST_BASE = 42            # 42-47

WHITE_ANY_CONTROL_PRE = AGG_PRE_BASE + 0
BLACK_ANY_CONTROL_PRE = AGG_PRE_BASE + 1
WHITE_LEGAL_CONTROL_PRE = AGG_PRE_BASE + 2
BLACK_LEGAL_CONTROL_PRE = AGG_PRE_BASE + 3
WHITE_NET_CONTROL_PRE = AGG_PRE_BASE + 4
BLACK_NET_CONTROL_PRE = AGG_PRE_BASE + 5

WHITE_ANY_CONTROL_POST = AGG_POST_BASE + 0
BLACK_ANY_CONTROL_POST = AGG_POST_BASE + 1
WHITE_LEGAL_CONTROL_POST = AGG_POST_BASE + 2
BLACK_LEGAL_CONTROL_POST = AGG_POST_BASE + 3
WHITE_NET_CONTROL_POST = AGG_POST_BASE + 4
BLACK_NET_CONTROL_POST = AGG_POST_BASE + 5

# Move-square planes
FROM_SQUARE_PLANE = 48
TO_SQUARE_PLANE = 49
PREV_FROM_PLANE = 50
PREV_TO_PLANE = 51

# Rule-state planes (pre)
PRE_WHITE_CASTLE_K = 52
PRE_WHITE_CASTLE_Q = 53
PRE_BLACK_CASTLE_K = 54
PRE_BLACK_CASTLE_Q = 55
PRE_SIDE_TO_MOVE   = 56
PRE_EP_TARGET      = 57

# Rule-state planes (post)
POST_WHITE_CASTLE_K = 58
POST_WHITE_CASTLE_Q = 59
POST_BLACK_CASTLE_K = 60
POST_BLACK_CASTLE_Q = 61
POST_SIDE_TO_MOVE   = 62
POST_EP_TARGET      = 63

# ---------------------------------------------------------------------
# Human-readable labels for planes
# ---------------------------------------------------------------------

COLOR_NAMES = {
    chess.WHITE: "white",
    chess.BLACK: "black",
}

PIECE_TYPE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}

def plane_label_for_index(idx: int) -> str:
    """
    Return a human-readable label for a given plane index 0..63,
    consistent with the 64-plane layout.
    """
    # 0–11: board_pre per (color, piece_type)
    if 0 <= idx <= 11:
        offset = idx
        color_idx = offset // NUM_PIECE_TYPES  # 0 = white, 1 = black
        pt_idx = offset % NUM_PIECE_TYPES
        color = COLORS[color_idx]
        pt = PIECE_TYPES[pt_idx]
        return f"board_pre_{COLOR_NAMES[color]}_{PIECE_TYPE_NAMES[pt]}"

    # 12–23: presence_pre per (color, piece_type)
    if 12 <= idx <= 23:
        offset = idx - 12
        color_idx = offset // NUM_PIECE_TYPES
        pt_idx = offset % NUM_PIECE_TYPES
        color = COLORS[color_idx]
        pt = PIECE_TYPES[pt_idx]
        return f"presence_pre_{COLOR_NAMES[color]}_{PIECE_TYPE_NAMES[pt]}"

    # 24–35: control_pre per (color, piece_type)
    if 24 <= idx <= 35:
        offset = idx - 24
        color_idx = offset // NUM_PIECE_TYPES
        pt_idx = offset % NUM_PIECE_TYPES
        color = COLORS[color_idx]
        pt = PIECE_TYPES[pt_idx]
        return f"control_pre_{COLOR_NAMES[color]}_{PIECE_TYPE_NAMES[pt]}"

    # 36–41: aggregates (pre)
    agg_pre_labels = {
        36: "white_any_control_pre",
        37: "black_any_control_pre",
        38: "white_legal_control_pre",
        39: "black_legal_control_pre",
        40: "white_net_control_pre",
        41: "black_net_control_pre",
    }
    if 36 <= idx <= 41:
        return agg_pre_labels[idx]

    # 42–47: aggregates (post)
    agg_post_labels = {
        42: "white_any_control_post",
        43: "black_any_control_post",
        44: "white_legal_control_post",
        45: "black_legal_control_post",
        46: "white_net_control_post",
        47: "black_net_control_post",
    }
    if 42 <= idx <= 47:
        return agg_post_labels[idx]

    # 48–51: move-square planes
    move_square_labels = {
        48: "from_square_current",
        49: "to_square_current",
        50: "from_square_prev",
        51: "to_square_prev",
    }
    if 48 <= idx <= 51:
        return move_square_labels[idx]

    # 52–57: rule-state planes (pre)
    rule_pre_labels = {
        52: "pre_white_can_castle_k",
        53: "pre_white_can_castle_q",
        54: "pre_black_can_castle_k",
        55: "pre_black_can_castle_q",
        56: "pre_side_to_move_white",
        57: "pre_en_passant_target",
    }
    if 52 <= idx <= 57:
        return rule_pre_labels[idx]

    # 58–63: rule-state planes (post)
    rule_post_labels = {
        58: "post_white_can_castle_k",
        59: "post_white_can_castle_q",
        60: "post_black_can_castle_k",
        61: "post_black_can_castle_q",
        62: "post_side_to_move_white",
        63: "post_en_passant_target",
    }
    if 58 <= idx <= 63:
        return rule_post_labels[idx]

    return f"unknown_plane_{idx}"

# ---------------------------------------------------------------------
# Geometric presence precomputation (empty-board capture reach)
# ---------------------------------------------------------------------

# GEOM_PRESENCE[color][piece_type][square] -> bitboard (int)
GEOM_PRESENCE: Dict[bool, Dict[chess.PieceType, Tuple[int, ...]]] = {}


def _in_bounds(file: int, rank: int) -> bool:
    return 0 <= file < 8 and 0 <= rank < 8


def _square(file: int, rank: int) -> int:
    """File a=0..7, rank 1=0..7 → 0..63 (a1=0, h8=63)."""
    return rank * 8 + file


def _precompute_geometric_presence() -> None:
    """
    Precompute geometric presence (empty-board capture squares) for all
    (color, piece_type, square).
    """
    global GEOM_PRESENCE

    GEOM_PRESENCE = {
        color: {pt: [0] * NUM_SQUARES for pt in PIECE_TYPES} for color in COLORS
    }

    knight_deltas = [
        (-2, -1), (-2, 1),
        (-1, -2), (-1, 2),
        (1, -2),  (1, 2),
        (2, -1),  (2, 1),
    ]
    king_deltas = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    bishop_dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    rook_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    queen_dirs = bishop_dirs + rook_dirs

    for sq in range(NUM_SQUARES):
        file = sq % 8
        rank = sq // 8

        # Knight (color-independent)
        knight_bb = 0
        for df, dr in knight_deltas:
            nf, nr = file + df, rank + dr
            if _in_bounds(nf, nr):
                knight_bb |= 1 << _square(nf, nr)

        # King (color-independent, *excluding* castling squares by default)
        king_bb = 0
        for df, dr in king_deltas:
            nf, nr = file + df, rank + dr
            if _in_bounds(nf, nr):
                king_bb |= 1 << _square(nf, nr)

        # Sliding pieces (color-independent)
        bishop_bb = 0
        for df, dr in bishop_dirs:
            nf, nr = file + df, rank + dr
            while _in_bounds(nf, nr):
                bishop_bb |= 1 << _square(nf, nr)
                nf += df
                nr += dr

        rook_bb = 0
        for df, dr in rook_dirs:
            nf, nr = file + df, rank + dr
            while _in_bounds(nf, nr):
                rook_bb |= 1 << _square(nf, nr)
                nf += df
                nr += dr

        queen_bb = bishop_bb | rook_bb

        # Pawns: color-dependent
        white_pawn_bb = 0
        for df, dr in [(-1, 1), (1, 1)]:
            nf, nr = file + df, rank + dr
            if _in_bounds(nf, nr):
                white_pawn_bb |= 1 << _square(nf, nr)

        black_pawn_bb = 0
        for df, dr in [(-1, -1), (1, -1)]:
            nf, nr = file + df, rank + dr
            if _in_bounds(nf, nr):
                black_pawn_bb |= 1 << _square(nf, nr)

        for color in COLORS:
            pawn_bb = white_pawn_bb if color == chess.WHITE else black_pawn_bb
            GEOM_PRESENCE[color][chess.PAWN][sq] = pawn_bb
            GEOM_PRESENCE[color][chess.KNIGHT][sq] = knight_bb
            GEOM_PRESENCE[color][chess.BISHOP][sq] = bishop_bb
            GEOM_PRESENCE[color][chess.ROOK][sq] = rook_bb
            GEOM_PRESENCE[color][chess.QUEEN][sq] = queen_bb
            GEOM_PRESENCE[color][chess.KING][sq] = king_bb


_precompute_geometric_presence()

# ---------------------------------------------------------------------
# Helpers: bitboard → plane
# ---------------------------------------------------------------------


def _set_bits_from_bb(feats: np.ndarray, plane_idx: int, bb: int) -> None:
    """Set feat[plane_idx, sq] = 1 for all bits set in bb."""
    while bb:
        lsb = bb & -bb
        sq = (lsb.bit_length() - 1)
        feats[plane_idx, sq] = 1
        bb ^= lsb


def _piece_plane_index(base: int, color: bool, piece_type: chess.PieceType) -> int:
    """Plane index for a given (color, piece_type) group, offset by base."""
    color_idx = 0 if color == chess.WHITE else 1
    pt_idx = PIECE_TYPE_TO_INDEX[piece_type]
    return base + color_idx * NUM_PIECE_TYPES + pt_idx


# ---------------------------------------------------------------------
# Presence / control / aggregates for a given board
# ---------------------------------------------------------------------


def compute_presence_bitboards(
    board: chess.Board,
) -> Dict[Tuple[bool, chess.PieceType], int]:
    """
    Compute presence bitboards for this board using precomputed geometric
    presence:

        presence[(color, pt)] = OR over pieces of type pt, color 'color'
            of GEOM_PRESENCE[color][pt][square_of_piece]
    """
    presence: Dict[Tuple[bool, chess.PieceType], int] = {
        (color, pt): 0 for color in COLORS for pt in PIECE_TYPES
    }

    for sq, piece in board.piece_map().items():
        color = piece.color
        pt = piece.piece_type
        bb = GEOM_PRESENCE[color][pt][sq]
        presence[(color, pt)] |= bb

    return presence


def compute_control_and_legal_bitboards(
    board: chess.Board,
) -> Tuple[
    Dict[Tuple[bool, chess.PieceType], int],
    Dict[Tuple[bool, chess.PieceType], int],
]:
    """
    Compute:

      * control_any[(color, pt)]   – "any control": squares attacked by
                                     pieces of that type and color.
      * legal_control[(color, pt)] – "legal control": same, but filtered
                                     by absolute pins.
    """
    control_any: Dict[Tuple[bool, chess.PieceType], int] = {
        (color, pt): 0 for color in COLORS for pt in PIECE_TYPES
    }
    legal_control: Dict[Tuple[bool, chess.PieceType], int] = {
        (color, pt): 0 for color in COLORS for pt in PIECE_TYPES
    }

    for sq, piece in board.piece_map().items():
        color = piece.color
        pt = piece.piece_type

        attacks_bb = int(board.attacks(sq))  # SquareSet -> int bitboard
        control_any[(color, pt)] |= attacks_bb

        # Default: legal_control = attacks_bb
        legal_bb = attacks_bb

        # Absolute pin logic
        if board.is_pinned(color, sq):
            pin_mask = int(board.pin(color, sq))  # mask of rank/file/diag of pin

            if pt in (chess.ROOK, chess.BISHOP, chess.QUEEN, chess.PAWN):
                # Sliders + pawns can only stay on the pin line.
                legal_bb &= pin_mask
            elif pt == chess.KNIGHT:
                # Knights cannot move along the line of a pin.
                legal_bb = 0
            else:
                # King is not treated as "pinned" here.
                pass

        legal_control[(color, pt)] |= legal_bb

    return control_any, legal_control


def compute_aggregate_bitboards(
    control_any: Dict[Tuple[bool, chess.PieceType], int],
    legal_control: Dict[Tuple[bool, chess.PieceType], int],
) -> Dict[str, int]:
    """
    From per-piece any_control/legal_control, compute aggregated bitboards:

        white_any_control,  black_any_control
        white_legal_control, black_legal_control
        white_net_control,  black_net_control

    Net control is pawn-weighted:

        score(sq) = 1{W legal_control(sq)} - 1{B legal_control(sq)}
                    + 1{W pawn_legal_control(sq)} - 1{B pawn_legal_control(sq)}

      * If score > 0 → white_net_control[sq] = 1, black_net_control[sq] = 0
      * If score < 0 → black_net_control[sq] = 1, white_net_control[sq] = 0
      * If score = 0 → both 0 (neutral)
    """
    white_any_control = 0
    black_any_control = 0
    white_legal = 0
    black_legal = 0

    for pt in PIECE_TYPES:
        white_any_control |= control_any[(chess.WHITE, pt)]
        black_any_control |= control_any[(chess.BLACK, pt)]
        white_legal |= legal_control[(chess.WHITE, pt)]
        black_legal |= legal_control[(chess.BLACK, pt)]

    white_pawn_legal = legal_control[(chess.WHITE, chess.PAWN)]
    black_pawn_legal = legal_control[(chess.BLACK, chess.PAWN)]

    white_net = 0
    black_net = 0

    for sq in range(NUM_SQUARES):
        mask = 1 << sq

        w_leg = bool(white_legal & mask)
        b_leg = bool(black_legal & mask)
        w_p = bool(white_pawn_legal & mask)
        b_p = bool(black_pawn_legal & mask)

        score = int(w_leg) - int(b_leg) + int(w_p) - int(b_p)

        if score > 0:
            white_net |= mask
        elif score < 0:
            black_net |= mask
        # score == 0 → neutral

    # ------------------------------------------------------------------
    # OPTIONAL: advanced, material-aware, x-ray-aware net control
    # ------------------------------------------------------------------
    #
    # The idea:
    #
    #   For each target square T:
    #       1. Consider all 8 directions (up/down/left/right + 4 diagonals).
    #       2. Along each direction, walk outward from T, collecting pieces
    #          that could *legally attack* T along that ray:
    #              - Rooks/queens on rank/file.
    #              - Bishops/queens on diagonals.
    #          Stop when we hit the board edge or a non-relevant piece.
    #       3. Separately gather knight attackers (1 jump away).
    #       4. Optionally include kings as last-resort defenders.
    #       5. For each side, we now have an ordered multiset of attackers:
    #              [ (value, kind, origin_square), ... ]
    #          where value ∈ {1=pawn, 3=knight/bishop, 5=rook, 9=queen, 0=king}
    #          and order respects "who can arrive on T first" along each line.
    #
    #   Then we run two "thought experiments":
    #
    #       - Experiment A: White moves first onto T, then Black responds,
    #                       alternating captures on T until one side either
    #                       runs out of attackers or decides to disengage
    #                       (stop recapturing).
    #
    #       - Experiment B: Black moves first onto T, White responds, same
    #                       depletion/disengagement logic.
    #
    #   In each experiment, we:
    #
    #       * Track the sequence of captures and material lost by each side.
    #       * Allow a side to "stop recapturing" as soon as doing so yields
    #         a material edge (disengagement).
    #       * If the side that *initiated* the occupation of T ends up with
    #         a favorable or equal trade *and* still occupies T, they win
    #         that experiment.
    #
    #   Finally:
    #
    #       * If one side wins BOTH experiments (starting first or second),
    #         we declare that side has material net-control of T.
    #       * If the experiments split (1-1) or both are unclear, the
    #         square is treated as neutral.
    #
    # This would fully capture batteries, x-rays, overloaded pieces, and
    # the difference between "nominal" and "practical" control.
    #
    # We could implement this as:
    #
    #   def _ray_attackers(board, target_square, color):
    #       ...
    #
    #   def _knight_attackers(board, target_square, color):
    #       ...
    #
    #   def _simulate_exchange(white_attackers, black_attackers, starter):
    #       ...
    #
    #   def advanced_net_control_for_square(board, sq):
    #       ...
    #
    # and then fill white_net / black_net accordingly. For now this is
    # commented out because of computational cost and complexity, but
    # the conceptual skeleton is here for future upgrades.
    #
    # ------------------------------------------------------------------

    return {
        "white_any_control": white_any_control,
        "black_any_control": black_any_control,
        "white_legal_control": white_legal,
        "black_legal_control": black_legal,
        "white_net_control": white_net,
        "black_net_control": black_net,
    }


# ---------------------------------------------------------------------
# Main encoder: one ply → (64, 8, 8) binary planes
# ---------------------------------------------------------------------


def encode_ply_planes(board: chess.Board, move: chess.Move) -> np.ndarray:
    """
    Encode a single ply (move) into a (64, 8, 8) binary feature tensor.

    Input:
        board : chess.Board in the position *before* the move
        move  : chess.Move to be played from this board (assumed legal)

    Output:
        feats : np.ndarray of shape (64, 8, 8), dtype uint8
    """
    feats = np.zeros((NUM_PLANES, NUM_SQUARES), dtype=np.uint8)

    # ------------------------------------------------------------------
    # 0) Previous move planes (50, 51) from board.move_stack
    # ------------------------------------------------------------------
    if board.move_stack:
        prev = board.move_stack[-1]
        feats[PREV_FROM_PLANE, prev.from_square] = 1
        feats[PREV_TO_PLANE, prev.to_square] = 1
    # else: leave as all zeros for first move

    # ------------------------------------------------------------------
    # 1) Board pre: piece occupancy per color & type (0–11)
    # ------------------------------------------------------------------
    for sq, piece in board.piece_map().items():
        color = piece.color
        pt = piece.piece_type
        plane_idx = _piece_plane_index(BOARD_PRE_BASE, color, pt)
        feats[plane_idx, sq] = 1

    # ------------------------------------------------------------------
    # 2) Presence & control (pre) – 12–35
    # ------------------------------------------------------------------
    presence_pre = compute_presence_bitboards(board)
    control_any_pre, legal_control_pre = compute_control_and_legal_bitboards(board)

    for color in COLORS:
        for pt in PIECE_TYPES:
            presence_bb = presence_pre[(color, pt)]
            control_bb = control_any_pre[(color, pt)]

            plane_presence = _piece_plane_index(PRESENCE_PRE_BASE, color, pt)
            plane_control = _piece_plane_index(CONTROL_PRE_BASE, color, pt)

            _set_bits_from_bb(feats, plane_presence, presence_bb)
            _set_bits_from_bb(feats, plane_control, control_bb)

    # ------------------------------------------------------------------
    # 3) Aggregates pre (36–41)
    # ------------------------------------------------------------------
    agg_pre = compute_aggregate_bitboards(control_any_pre, legal_control_pre)

    _set_bits_from_bb(feats, WHITE_ANY_CONTROL_PRE, agg_pre["white_any_control"])
    _set_bits_from_bb(feats, BLACK_ANY_CONTROL_PRE, agg_pre["black_any_control"])
    _set_bits_from_bb(feats, WHITE_LEGAL_CONTROL_PRE, agg_pre["white_legal_control"])
    _set_bits_from_bb(feats, BLACK_LEGAL_CONTROL_PRE, agg_pre["black_legal_control"])
    _set_bits_from_bb(feats, WHITE_NET_CONTROL_PRE, agg_pre["white_net_control"])
    _set_bits_from_bb(feats, BLACK_NET_CONTROL_PRE, agg_pre["black_net_control"])

    # ------------------------------------------------------------------
    # 4) Rule-state planes pre (52–57)
    # ------------------------------------------------------------------
    if board.has_kingside_castling_rights(chess.WHITE):
        feats[PRE_WHITE_CASTLE_K, :] = 1
    if board.has_queenside_castling_rights(chess.WHITE):
        feats[PRE_WHITE_CASTLE_Q, :] = 1
    if board.has_kingside_castling_rights(chess.BLACK):
        feats[PRE_BLACK_CASTLE_K, :] = 1
    if board.has_queenside_castling_rights(chess.BLACK):
        feats[PRE_BLACK_CASTLE_Q, :] = 1

    if board.turn == chess.WHITE:
        feats[PRE_SIDE_TO_MOVE, :] = 1  # all-ones plane if White to move

    if board.ep_square is not None:
        feats[PRE_EP_TARGET, board.ep_square] = 1

    # ------------------------------------------------------------------
    # 5) Aggregates post + rule-state post (42–47, 58–63)
    # ------------------------------------------------------------------
    board.push(move)
    try:
        control_any_post, legal_control_post = compute_control_and_legal_bitboards(board)
        agg_post = compute_aggregate_bitboards(control_any_post, legal_control_post)

        _set_bits_from_bb(
            feats, WHITE_ANY_CONTROL_POST, agg_post["white_any_control"]
        )
        _set_bits_from_bb(
            feats, BLACK_ANY_CONTROL_POST, agg_post["black_any_control"]
        )
        _set_bits_from_bb(
            feats, WHITE_LEGAL_CONTROL_POST, agg_post["white_legal_control"]
        )
        _set_bits_from_bb(
            feats, BLACK_LEGAL_CONTROL_POST, agg_post["black_legal_control"]
        )
        _set_bits_from_bb(
            feats, WHITE_NET_CONTROL_POST, agg_post["white_net_control"]
        )
        _set_bits_from_bb(
            feats, BLACK_NET_CONTROL_POST, agg_post["black_net_control"]
        )

        # Rule-state post
        if board.has_kingside_castling_rights(chess.WHITE):
            feats[POST_WHITE_CASTLE_K, :] = 1
        if board.has_queenside_castling_rights(chess.WHITE):
            feats[POST_WHITE_CASTLE_Q, :] = 1
        if board.has_kingside_castling_rights(chess.BLACK):
            feats[POST_BLACK_CASTLE_K, :] = 1
        if board.has_queenside_castling_rights(chess.BLACK):
            feats[POST_BLACK_CASTLE_Q, :] = 1

        if board.turn == chess.WHITE:
            feats[POST_SIDE_TO_MOVE, :] = 1

        if board.ep_square is not None:
            feats[POST_EP_TARGET, board.ep_square] = 1

    finally:
        board.pop()

    # ------------------------------------------------------------------
    # 6) From / to squares for current move (48, 49)
    # ------------------------------------------------------------------
    feats[FROM_SQUARE_PLANE, move.from_square] = 1
    feats[TO_SQUARE_PLANE, move.to_square] = 1

    # Reshape to (planes, 8, 8)
    return feats.reshape(NUM_PLANES, 8, 8)

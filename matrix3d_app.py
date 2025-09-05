"""
Matrix & Vector 3D Visualizer
---------------------------------
A single-file Streamlit app to visualize 3D vectors and point clouds and apply
3×3 linear (and affine) transformations. Includes a small matrix calculator.

Run locally:
  1) pip install streamlit numpy plotly
  2) streamlit run matrix3d_app.py

Author: Chema
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st


# ==========================
# Utilities & Math helpers
# ==========================

@dataclass
class Transform:
    name: str
    matrix: np.ndarray  # shape (3, 3)
    shift: np.ndarray | None = None  # optional 3D translation (affine)

    def apply(self, pts: np.ndarray) -> np.ndarray:
        """Apply linear part (and optional translation) to Nx3 points."""
        out = pts @ self.matrix.T
        if self.shift is not None:
            out = out + self.shift
        return out


def _as_np_array(rows: List[List[float]]) -> np.ndarray:
    arr = np.array(rows, dtype=float)
    if arr.shape != (3, 3):
        raise ValueError("Matrix must be 3×3.")
    return arr


def rotation_x(deg: float) -> Transform:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    R = _as_np_array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return Transform(name=f"RotX({deg:.1f}°)", matrix=R)


def rotation_y(deg: float) -> Transform:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    R = _as_np_array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return Transform(name=f"RotY({deg:.1f}°)", matrix=R)


def rotation_z(deg: float) -> Transform:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    R = _as_np_array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return Transform(name=f"RotZ({deg:.1f}°)", matrix=R)


def scale(sx: float, sy: float, sz: float) -> Transform:
    S = _as_np_array([[sx, 0, 0], [0, sy, 0], [0, 0, sz]])
    return Transform(name=f"Scale({sx:.2f},{sy:.2f},{sz:.2f})", matrix=S)


def shear_xy(kx: float, ky: float) -> Transform:
    """Shear in the XY plane (x += kx*y, y += ky*x)."""
    Sh = _as_np_array([[1, kx, 0], [ky, 1, 0], [0, 0, 1]])
    return Transform(name=f"ShearXY(kx={kx:.2f}, ky={ky:.2f})", matrix=Sh)


def identity() -> Transform:
    return Transform(name="Identity", matrix=np.eye(3))


def custom_matrix(m00: float, m01: float, m02: float,
                  m10: float, m11: float, m12: float,
                  m20: float, m21: float, m22: float) -> Transform:
    M = _as_np_array([[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]])
    return Transform(name="Custom3x3", matrix=M)


def translation(tx: float, ty: float, tz: float) -> Transform:
    return Transform(name=f"Translate({tx:.2f},{ty:.2f},{tz:.2f})", matrix=np.eye(3), shift=np.array([tx, ty, tz]))


def compose(transforms: List[Transform]) -> Transform:
    """Compose transforms in the listed order (left-to-right application)."""
    if not transforms:
        return identity()

    M = np.eye(3)
    t = np.zeros(3)
    for tr in transforms:  # apply sequentially
        M = tr.matrix @ M
        if tr.shift is not None:
            t = tr.matrix @ t + tr.shift
        else:
            t = tr.matrix @ t
    shift = t if np.linalg.norm(t) > 1e-12 else None
    return Transform(name=" ∘ ".join([tr.name for tr in transforms]), matrix=M, shift=shift)


# ==========================
# Point cloud generators
# ==========================

def make_axes_vectors(length: float = 1.0) -> np.ndarray:
    return np.array([
        [length, 0, 0],
        [0, length, 0],
        [0, 0, length],
    ], dtype=float)


def make_cube(n_per_edge: int = 5, size: float = 1.0) -> np.ndarray:
    lin = np.linspace(-size, size, n_per_edge)
    X, Y, Z = np.meshgrid(lin, lin, lin)
    pts = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    # keep only shell
    on_shell = (
        (np.isclose(np.abs(pts[:, 0]), size)) |
        (np.isclose(np.abs(pts[:, 1]), size)) |
        (np.isclose(np.abs(pts[:, 2]), size))
    )
    return pts[on_shell]


def make_sphere(n: int = 500, radius: float = 1.0, rng_seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)
    u = rng.uniform(0.0, 1.0, n)
    v = rng.uniform(0.0, 1.0, n)
    theta = 2 * math.pi * u
    phi = np.arccos(2 * v - 1)
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.vstack([x, y, z]).T


def parse_custom_points(txt: str) -> np.ndarray:
    """Parse lines like: x,y,z (comma or space separated)."""
    rows: List[List[float]] = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        # allow space or comma separated
        parts = [p for p in line.replace(",", " ").split(" ") if p]
        if len(parts) != 3:
            raise ValueError(f"Each line must have 3 numbers; got: {line}")
        rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
    if not rows:
        return np.zeros((0, 3))
    return np.array(rows, dtype=float)


# ==========================
# Plotting helpers (Plotly)
# ==========================

def add_vectors(fig: go.Figure, vectors: np.ndarray, name_prefix: str, color: str, width: float = 6.0):
    """Add 3D vectors (lines from origin)."""
    for i, v in enumerate(vectors):
        fig.add_trace(
            go.Scatter3d(
                x=[0, v[0]], y=[0, v[1]], z=[0, v[2]],
                mode="lines+markers",
                line=dict(width=width, color=color),
                marker=dict(size=2),
                name=f"{name_prefix} v{i+1}"
            )
        )


def make_figure(original: np.ndarray,
                transformed: np.ndarray | None,
                orig_vectors: np.ndarray | None,
                xfrm_vectors: np.ndarray | None,
                show_grid: bool = True,
                title: str = "3D Visualization") -> go.Figure:
    fig = go.Figure()

    if original.size > 0:
        fig.add_trace(go.Scatter3d(
            x=original[:, 0], y=original[:, 1], z=original[:, 2],
            mode="markers",
            marker=dict(size=3, opacity=0.8),
            name="Original"
        ))

    if transformed is not None and transformed.size > 0:
        fig.add_trace(go.Scatter3d(
            x=transformed[:, 0], y=transformed[:, 1], z=transformed[:, 2],
            mode="markers",
            marker=dict(size=3, opacity=0.8),
            name="Transformed"
        ))

    if orig_vectors is not None and orig_vectors.size > 0:
        add_vectors(fig, orig_vectors, name_prefix="Basis", color="#1f77b4")

    if xfrm_vectors is not None and xfrm_vectors.size > 0:
        add_vectors(fig, xfrm_vectors, name_prefix="A·Basis", color="#d62728")

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title="x", backgroundcolor="rgb(240,240,240)", gridcolor="white", showbackground=True, zerolinecolor="gray"),
            yaxis=dict(title="y", backgroundcolor="rgb(240,240,240)", gridcolor="white", showbackground=True, zerolinecolor="gray"),
            zaxis=dict(title="z", backgroundcolor="rgb(240,240,240)", gridcolor="white", showbackground=True, zerolinecolor="gray"),
            aspectmode="cube",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


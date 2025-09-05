"""
Matrix & Vector 3D Visualizer
---------------------------------
A single-file Streamlit app to visualize 3D vectors and point clouds and apply
3×3 linear (and affine) transformations. Includes a small matrix calculator.

Run locally:
  1) pip install streamlit numpy plotly
  2) streamlit run matrix3d_app.py

Author: chema
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


# ==========================
# Streamlit UI
# ==========================

def sidebar_transform_builder() -> List[Transform]:
    st.sidebar.header("Build Transform")

    tabs = st.sidebar.tabs(["Preset", "Custom 3×3", "Translation"])
    built: List[Transform] = []

    with tabs[0]:
        st.markdown("**Presets (add in order):**")
        with st.popover("Rotation X/Y/Z"):
            rx = st.slider("Rot X (deg)", -180.0, 180.0, 0.0, 1.0)
            ry = st.slider("Rot Y (deg)", -180.0, 180.0, 0.0, 1.0)
            rz = st.slider("Rot Z (deg)", -180.0, 180.0, 0.0, 1.0)
            if st.button("Add Rotations (X→Y→Z)"):
                if abs(rx) > 1e-9: built.append(rotation_x(rx))
                if abs(ry) > 1e-9: built.append(rotation_y(ry))
                if abs(rz) > 1e-9: built.append(rotation_z(rz))

        with st.popover("Scale"):
            sx = st.number_input("sx", value=1.0, step=0.1, format="%.3f")
            sy = st.number_input("sy", value=1.0, step=0.1, format="%.3f")
            sz = st.number_input("sz", value=1.0, step=0.1, format="%.3f")
            if st.button("Add Scale"):
                built.append(scale(sx, sy, sz))

        with st.popover("Shear XY"):
            kx = st.number_input("kx (x += kx·y)", value=0.0, step=0.1, format="%.3f")
            ky = st.number_input("ky (y += ky·x)", value=0.0, step=0.1, format="%.3f")
            if st.button("Add ShearXY"):
                built.append(shear_xy(kx, ky))

        if st.button("Add Identity"):
            built.append(identity())

    with tabs[1]:
        st.markdown("**Enter a custom 3×3 matrix (row-major):**")
        c1, c2, c3 = st.columns(3)
        m00 = c1.number_input("m00", value=1.0, format="%.3f"); m01 = c2.number_input("m01", value=0.0, format="%.3f"); m02 = c3.number_input("m02", value=0.0, format="%.3f")
        m10 = c1.number_input("m10", value=0.0, format="%.3f"); m11 = c2.number_input("m11", value=1.0, format="%.3f"); m12 = c3.number_input("m12", value=0.0, format="%.3f")
        m20 = c1.number_input("m20", value=0.0, format="%.3f"); m21 = c2.number_input("m21", value=0.0, format="%.3f"); m22 = c3.number_input("m22", value=1.0, format="%.3f")
        if st.button("Add Custom 3×3"):
            built.append(custom_matrix(m00, m01, m02, m10, m11, m12, m20, m21, m22))

    with tabs[2]:
        st.markdown("**Optional translation (affine):**")
        t1, t2, t3 = st.columns(3)
        tx = t1.number_input("tx", value=0.0, step=0.1, format="%.3f")
        ty = t2.number_input("ty", value=0.0, step=0.1, format="%.3f")
        tz = t3.number_input("tz", value=0.0, step=0.1, format="%.3f")
        if st.button("Add Translation"):
            built.append(translation(tx, ty, tz))

    st.sidebar.caption("Tip: Add multiple transforms, then compose them.")
    return built


def render_visualizer():
    st.title("Matrix & Vector 3D Visualizer")

    # Build transforms
    added_now = sidebar_transform_builder()
    if "pipeline" not in st.session_state:
        st.session_state.pipeline: List[Transform] = []

    # Show & manage pipeline
    st.subheader("Transform Pipeline")
    with st.container(border=True):
        cols = st.columns([3, 1])
        cols[0].write(", ".join([t.name for t in st.session_state.pipeline]) or "(empty)")
        if cols[1].button("Clear pipeline", type="secondary"):
            st.session_state.pipeline.clear()
    if added_now:
        st.session_state.pipeline.extend(added_now)
        st.rerun()

    composed = compose(st.session_state.pipeline)

    # Data selection
    st.subheader("Data")
    data_tab, vectors_tab = st.tabs(["Points", "Vectors"])

    with data_tab:
        dataset = st.selectbox("Point set", ["Axes endpoints", "Cube shell", "Sphere surface", "Random Gaussian", "Custom (paste)"])
        pts = np.zeros((0, 3))
        if dataset == "Axes endpoints":
            L = st.slider("Axis length", 0.5, 5.0, 1.0, 0.1)
            pts = make_axes_vectors(L)
        elif dataset == "Cube shell":
            n = st.slider("Points per edge", 3, 12, 6)
            size = st.slider("Half-size", 0.5, 3.0, 1.0, 0.1)
            pts = make_cube(n, size)
        elif dataset == "Sphere surface":
            n = st.slider("# points", 100, 3000, 800, 50)
            radius = st.slider("Radius", 0.5, 3.0, 1.0, 0.1)
            pts = make_sphere(n, radius)
        elif dataset == "Random Gaussian":
            n = st.slider("# points", 50, 5000, 500, 50)
            mean = np.zeros(3)
            cov = np.eye(3)
            rng = np.random.default_rng(42)
            pts = rng.multivariate_normal(mean, cov, size=n)
        else:  # Custom
            st.caption("Paste lines: x,y,z or x y z")
            txt = st.text_area("Points", value="1,0,0\n0,1,0\n0,0,1\n-1,0,0\n0,-1,0\n0,0,-1", height=140)
            try:
                pts = parse_custom_points(txt)
                st.success(f"Parsed {len(pts)} points")
            except Exception as e:
                st.error(str(e))
                pts = np.zeros((0, 3))

    with vectors_tab:
        st.caption("Optional extra vectors (also drawn as arrows from origin)")
        vec_txt = st.text_area("Vectors (one per line)", value="", placeholder="e.g. 1,2,0\n-0.5,0,1.2", height=120)
        try:
            extra_vecs = parse_custom_points(vec_txt) if vec_txt.strip() else np.zeros((0, 3))
        except Exception as e:
            st.error(f"Vectors: {e}")
            extra_vecs = np.zeros((0, 3))

    # Apply
    transformed_pts = composed.apply(pts) if pts.size else np.zeros((0, 3))

    # Basis vectors under A
    basis = np.eye(3)
    basis_scaled = make_axes_vectors(1.0)
    Abasis = composed.apply(basis)

    st.subheader("Plot")
    show_orig = st.checkbox("Show original points", value=True)
    show_xfrm = st.checkbox("Show transformed points", value=True)
    show_basis = st.checkbox("Show basis vectors", value=True)
    show_Abasis = st.checkbox("Show A·basis vectors", value=True)

    orig_to_plot = pts if show_orig else np.zeros((0, 3))
    xfrm_to_plot = transformed_pts if show_xfrm else np.zeros((0, 3))
    basis_to_plot = basis_scaled if show_basis else np.zeros((0, 3))
    Abasis_to_plot = Abasis if show_Abasis else np.zeros((0, 3))

    fig = make_figure(
        original=orig_to_plot,
        transformed=xfrm_to_plot,
        orig_vectors=basis_to_plot if basis_to_plot.size else None,
        xfrm_vectors=Abasis_to_plot if Abasis_to_plot.size else None,
        title="3D: Original vs Transformed"
    )

    # Add user vectors
    if extra_vecs.size:
        add_vectors(fig, extra_vecs, name_prefix="u", color="#2ca02c")
        add_vectors(fig, composed.apply(extra_vecs), name_prefix="A·u", color="#9467bd")

    st.plotly_chart(fig, use_container_width=True)

    # Matrix display
    st.subheader("Composed Transform")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.write(f"**Name:** {composed.name or 'Identity'}")
        st.write("**Matrix A (3×3):**")
        st.dataframe(np.round(composed.matrix, 6))
        if composed.shift is not None:
            st.write("**Translation t:**")
            st.code(np.round(composed.shift, 6))
        st.download_button("Download A as CSV", data="\n".join(
            ",".join(map(str, row)) for row in composed.matrix
        ), file_name="A_3x3.csv", mime="text/csv")
    with c2:
        st.write("**det(A):**", float(np.linalg.det(composed.matrix)))
        try:
            st.write("**rank(A):**", int(np.linalg.matrix_rank(composed.matrix)))
        except Exception:
            st.write("**rank(A):** n/a")
        try:
            st.write("**A⁻¹:**")
            st.dataframe(np.round(np.linalg.inv(composed.matrix), 6))
        except np.linalg.LinAlgError:
            st.info("Matrix is singular; no inverse.")


def render_matrix_calculator():
    st.title("Matrix Calculator (3×3 focus)")
    st.caption("Quick multiply & apply to vectors. Works best for 3×3, but allows general sizes where possible.")

    tab_mul, tab_apply = st.tabs(["Multiply", "Apply to Vectors"])

    with tab_mul:
        st.subheader("A·B")
        dims = st.columns(3)
        rA = dims[0].number_input("rows(A)", min_value=1, max_value=6, value=3)
        cA = dims[1].number_input("cols(A)", min_value=1, max_value=6, value=3)
        rB = dims[2].number_input("rows(B)", min_value=1, max_value=6, value=3)
        cB = st.number_input("cols(B)", min_value=1, max_value=6, value=3)

        if st.session_state.get("A") is None or st.session_state.get("A").shape != (rA, cA):
            st.session_state.A = np.eye(rA, cA)
        if st.session_state.get("B") is None or st.session_state.get("B").shape != (rB, cB):
            st.session_state.B = np.eye(rB, cB)

        st.write("**Matrix A**")
        A = st.data_editor(st.session_state.A, num_rows="dynamic", key="A_editor")
        st.write("**Matrix B**")
        B = st.data_editor(st.session_state.B, num_rows="dynamic", key="B_editor")

        if A.shape[1] != B.shape[0]:
            st.error(f"Incompatible shapes: A is {A.shape}, B is {B.shape} (need cols(A)==rows(B)).")
        else:
            C = A @ B
            st.success(f"Result C = A·B (shape {C.shape})")
            st.dataframe(C)

    with tab_apply:
        st.subheader("y = A·x + t (optional)")
        use_affine = st.checkbox("Include translation t (affine)", value=False)
        A = st.data_editor(np.eye(3), key="A_apply")
        x = st.data_editor(np.array([[1.0], [0.0], [0.0]]), key="x_apply")
        t = None
        if use_affine:
            t = st.data_editor(np.zeros((3, 1)), key="t_apply")
        try:
            y = A @ x
            if use_affine:
                if t is None:
                    t = np.zeros((A.shape[0], 1))
                y = y + t
            st.success("Computed y")
            st.dataframe(y)
        except Exception as e:
            st.error(f"{e}")


def main():
    st.set_page_config(page_title="Matrix & Vector 3D Visualizer", layout="wide")

    page = st.sidebar.radio("Pages", ["Visualizer", "Matrix Calculator", "Help"], index=0)

    if page == "Visualizer":
        render_visualizer()
    elif page == "Matrix Calculator":
        render_matrix_calculator()
    else:
        st.title("Help & Notes")
        st.markdown(
            """
            ### What can I do here?
            - Build 3×3 transformations from presets (rotations, scale, shear) or custom entries.
            - Add an optional translation to create an **affine** transform.
            - Load or paste points and visualize how the transform changes them.
            - Compare original vs transformed point clouds.
            - See how the basis vectors **e₁, e₂, e₃** map under **A**.
            - Download the composed matrix as CSV; inspect det, rank, and inverse.

            ### Tips
            - Order matters. The pipeline composes transforms in the order you add them (left→right).
            - For stable visuals, start with modest scales and shears.
            - Use the **Matrix Calculator** tab for quick products (A·B) or y = A·x + t.

            ### Shortcuts
            - **Identity**: Quickly insert a no-op to anchor your pipeline.
            - **Custom 3×3**: Enter exact entries when reproducing textbook examples.

            ---
            **Built with:** Streamlit + NumPy + Plotly.
            """
        )


if __name__ == "__main__":
    main()

from torch import Tensor
import plotly.graph_objs as go


def plot_sphere(X: Tensor, X_A: Tensor):
    """
    Plot hyperspheres traces

    Args :
        X : Initial (raw) hypersphere X
        X : Guessed (model) hypersphere X_A
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=X[:, 0],
            y=X[:, 1],
            z=X[:, 2],
            mode="markers",
            marker=dict(color="blue"),
            name="raw",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=X_A[:, 0],
            y=X_A[:, 1],
            z=X_A[:, 2],
            mode="markers",
            marker=dict(color="red"),
            name="model",
        )
    )

    fig.show()

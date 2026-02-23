import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

def create_pr_doughnuts(precision, recall):

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}]])

    colours = [
            "#ffffff",
            "#00619A",
            ]

    fig.add_trace(go.Pie(
        values=[1 - precision, precision],
        hole=0.5,
        rotation=0,
        marker=dict(colors=colours),
        showlegend=False,
        textinfo='none'
    ), row=1, col=1)

    fig.add_trace(go.Pie(
        values=[1 - recall, recall],
        hole=0.5,
        rotation=00,
        marker=dict(colors=colours),
        showlegend=False,
        textinfo='none'
    ), row=1, col=2)

    fig.update_layout(
        annotations=[
            dict(text='Precision', x=0.23, y=0.525, showarrow=False, font_size=25, xanchor='center'),
            dict(text=f'{precision*100:.1f}%', x=0.23, y=0.46, showarrow=False, font_size=18, xanchor='center'),
            dict(text='Recall',    x=0.78, y=0.525, showarrow=False, font_size=25, xanchor='center'),
            dict(text=f'{recall*100:.1f}%',    x=0.78, y=0.46, showarrow=False, font_size=18, xanchor='center'),
        ],
        margin=dict(l=0, r=0, t=0, b=0)
    )

    fig.update_traces(hoverinfo = 'none')

    return fig


def create_pr_chart(pr_data, precision, recall):

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=pr_data[:, 0],
        y=pr_data[:, 1],
        mode='lines',
        showlegend = False
    ))

    fig.add_trace(go.Scatter(
        x=[recall * 100],
        y=[precision * 100],
        mode='markers',
        marker=dict(color='red', symbol='circle', size=12),
        showlegend = False

    ))

    fig.update_layout(
        width=1000,
        height=250,
        xaxis_title='Recall (%)',
        yaxis_title='Precision (%)',
        xaxis_range = [0,105],
        yaxis_range = [0, 105],
        margin=dict(l=0, r=0, t=0, b=0),
        )
    
    return fig
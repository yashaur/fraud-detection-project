from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import streamlit as st

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
            dict(text='Precision', x=0.225, y=0.525, showarrow=False, font_size=25, xanchor='center'),
            dict(text=f'{precision*100:.1f}%', x=0.225, y=0.46, showarrow=False, font_size=18, xanchor='center'),
            dict(text='Recall',    x=0.775, y=0.525, showarrow=False, font_size=25, xanchor='center'),
            dict(text=f'{recall*100:.1f}%',    x=0.775, y=0.46, showarrow=False, font_size=18, xanchor='center'),
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
    
    fig.update_traces(hoverinfo = 'none')

    
    return fig



def create_fraud_by_time_chart(stat_data):
    x = stat_data.index
    y = stat_data.values
    text = [str(np.round(val, 2)) + '%' for val in y]
    y_upper_limit = max(y) + 5

    if all(y == 100):
        cscale = 'blues'
    else:
        cscale = 'rdbu'

    fig_hour = go.Figure(
        data=[
            go.Bar(
                x = x,
                y = y,
                marker=dict(
                    color=stat_data.values,
                    colorscale=cscale,
                    showscale=False,
                )
            )
        ]
    )

    fig_hour.update_layout(
        title = 'Fraud Rate (%) by Hour of Day',
        xaxis_title="Hour",
        yaxis_title="Fraud Rate (%)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        # yaxis=dict(range=[0, y_upper_limit]),
        margin=dict(l=20, r=20, t=20, b=20)
    )

    return fig_hour


def create_segment_type_chart(input_df):

    if all(input_df.values == 1000):
        cscale = 'blues'
    else:
        cscale = 'rdbu'

    fig = go.Figure(
        data=[
            go.Bar(
                x=input_df.index,
                y=input_df.values,
                marker=dict(
                    color = input_df.values,
                    colorscale= cscale,
                    showscale=False,
                ),
            )
        ]
    )

    if len(input_df.index) == 2:
        x_axis = 'Time Segment'
    else:
        x_axis = 'Type of Transaction'

    fig.update_layout(
        title = '# Fraud transactions per 1000 transactions (Day vs Night)',
        xaxis_title= x_axis,
        yaxis_title="# Fraud transactions per 1000 transactions",
                margin=dict(l=20, r=20, t=20, b=20)

    )

    return fig


def create_shap_chart(shap_df):

    features_vals = sorted(zip(shap_df.columns, shap_df.values[0]), key = lambda row: row[1], reverse = False)
    features = [row[0] for row in features_vals]
    vals = [row[1] for row in features_vals]

    field_names = st.session_state['field_names'].copy()

    for idx, col in enumerate(features):
        if col in field_names:
            features[idx] = field_names[col]

    fig_shap = go.Figure(
        data=[
            go.Bar(
                y=features,
                x=vals,
                orientation="h",
                marker=dict(
                    color=vals,
                    colorscale="rdbu",
                    showscale=False,
                ),
            )
        ]
    )

    fig_shap.update_layout(
        title="Top Features Driving Fraud Risk",
        xaxis_title="Importance (Higher is better)",
        # yaxis_title="Feature",
    )

    return fig_shap
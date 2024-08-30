import numpy as np
import plotly.graph_objects as go



time = np.linspace(0, 10, 100)
intensity = np.exp(-time / 2)

regs = [(1, 2), (3, 4), (5, 6), (7, 8)]
colors = ['red','yellow','green','blue']

fig = go.Figure()
fig.add_trace(go.Scatter(x=time, y=intensity, mode='lines', name='Exponential Decay'))

for i, ((start, end), color) in enumerate(zip(regs, colors * 2), start=1):
    mask = (time >= start) & (time <= end)
    fig.add_trace(go.Scatter(
        x=time[mask], y=intensity[mask], fill='tozeroy', mode='lines', line=dict(color=color), name=f'I_{i}',
        fillcolor=f'rgba(255, 255, 0, 0.3)' if color == 'yellow' else f'rgba(0, 255, 255, 0.3)',
        showlegend=False))
    
    mid_time = (start + end) / 2
    mid_intensity = np.exp(-mid_time / 2)
    fig.add_annotation(x=mid_time, y=mid_intensity, text=f'I_{i}', showarrow=False, font=dict(size=14))

for i in range(len(regs) - 1):
    start, end = regs[i]
    next_start = regs[i + 1][0]
    
    fig.add_annotation(x=(start + next_start) / 2, y=-0.1, text=r'$dt$', showarrow=False, yshift=-10)
    fig.add_trace(go.Scatter(
        x=[start, next_start], y=[-0.05, -0.05], mode='lines', line=dict(color='black', dash='dash'),
        showlegend=False))
    
    fig.add_annotation(x=(start + end) / 2, y=-0.1, text=r'$g$', showarrow=False, yshift=-10)
    fig.add_trace(go.Scatter(
        x=[start, end], y=[-0.05, -0.05], mode='lines', line=dict(color='black', dash='dash'),
        showlegend=False))

last_start, last_end = regs[-1]
fig.add_annotation(x=(last_start + last_end) / 2, y=-0.1, text=r'$g$', showarrow=False, yshift=-10)
fig.add_trace(go.Scatter(
    x=[last_start, last_end], y=[-0.05, -0.05], mode='lines', line=dict(color='black', dash='dash'),
    showlegend=False))

fig.update_layout(
    xaxis_title="Time (ns)",
    yaxis_title="Intensity (a.u.)",
    title="Rapid Lifetime Determination",
    yaxis_range=[-0.2, 1],
    xaxis_range=[0, 10],
    plot_bgcolor='white',
    margin=dict(l=50, r=50, t=50, b=50)
)

fig.show()

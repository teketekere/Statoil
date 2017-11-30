import plotly.graph_objs as go
import plotly.offline as py
py.init_notebook_mode(connected=True)


def plotmy3d(c, name):
    data = [go.Surface(z=c)]
    layout = go.Layout(
        title=name,
        autosize=False,
        width=700,
        height=700,
        margin=dict(l=65, r=50, b=65, t=90)
    )
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig)

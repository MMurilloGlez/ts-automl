import plotly.graph_objects as go


def plot_test_pred(y_test, pred_manual):

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=list(y_test.index), y=list(y_test.iloc[:, 0]),
                   mode='lines+markers', name="Test"))
    fig.add_trace(
        go.Scatter(x=list(y_test.index), y=list(pred_manual),
                   line=go.scatter.Line(color="green"), name="Pred"))
    fig.show()


def plot_comparison(y_test, y_1=None, y_2=None, y_3=None, y_n=None):

    layout = go.Layout(
                       xaxis=dict(title="Fecha"),
                       yaxis=dict(title="Datos"),
                       font=dict(family='sans serif', size=20))
    fig = go.Figure(layout=layout)

    fig.add_trace(
        go.Scatter(x=list(y_test.index), y=list(y_test.iloc[:, 0]),
                   mode='lines+markers', name='Test'))
    if y_n is not None:
        fig.add_trace(
            go.Scatter(x=list(y_test.index), y=list(y_n),
                       line=go.scatter.Line(color="black"), name='Naive'))

    if y_1 is not None:
        fig.add_trace(
            go.Scatter(x=list(y_test.index), y=list(y_1),
                       line=go.scatter.Line(color="red"), name='LightGBM'))

    if y_2 is not None:
        fig.add_trace(
            go.Scatter(x=list(y_test.index), y=list(y_2),
                       line=go.scatter.Line(color="green"), name='kNN con optimización x30'))

    if y_3 is not None:
        fig.add_trace(
            go.Scatter(x=list(y_test.index), y=list(y_3),
                       line=go.scatter.Line(color="orange"), name='LGBM con optimización x50'))
    fig.show()


def plot_train(y_train):

    layout = go.Layout(
                       xaxis=dict(title="Fecha"),
                       yaxis=dict(title="Datos"),
                       font=dict(family='sans serif', size=20))

    fig = go.Figure(layout=layout)

    fig.add_trace(
        go.Scatter(x=list(y_train.index), y=list(y_train.iloc[:, 0]),
                   mode='lines', name="Train"))

    fig.show()

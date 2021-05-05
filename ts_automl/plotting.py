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

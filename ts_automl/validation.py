"""
Module for validation of the paremetrs fed into the prediction functions.

Will check that all parameters are compatible with the series and prediction
models that are used, and raise warnings or errors if they may cause conflicts.

"""


def parameters_validation(self):

    if isinstance(self.filename, str):
        pass
    else:
        return TypeError("filename must be of type str")

    if isinstance(self.freq, str):
        pass
    else:
        return TypeError("freq must be of type str")

    if isinstance(self.datecol, str):
        pass
    else:
        return TypeError("datecol must be of type str")

    if isinstance(self.targetcol, str):
        pass
    else:
        return TypeError("targetcol must be of type str")

    if isinstance(self.window_length, int):
        if self.window_length < 1:
            raise ValueError(f"`window_length` must be positive integer >= 1,"
                             f"but found: {self.window_length}")
    else:
        raise TypeError("window_length must be int")

    if isinstance(self.horizon, int):
        if self.horizon < 1:
            raise ValueError(f"`horizon` must be positive integer >= 1,"
                             f"but found: {self.horizon}")
    else:
        raise TypeError("horizon must be int")

    if isinstance(self.selected_feat, int):
        if self.selected_feat < 1:
            raise ValueError(f"`selected_feat` must be positive integer >= 1,"
                             f"but found: {self.selected_feat}")
    else:
        raise TypeError("selected_feat must be int")

    if isinstance(self.step, int):
        if self.step < 1:
            raise ValueError(f"`step` must be positive integer >= 1,"
                             f"but found: {self.step}")
    else:
        raise TypeError("step must be int")

    if isinstance(self.plot, bool):
        pass
    else:
        raise TypeError("plot must be bool")

    if isinstance(self.rel_metrics, bool):
        pass
    else:
        raise TypeError("plot must be bool")

def hook(module, inputs, output):
    self._acts[name] = output        # keep the graph: we differentiate w.r.t. this tensor

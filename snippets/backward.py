score = logits[torch.arange(x.size(0), device=x.device), class_idx].sum()   # y^c, pre-softmax
names = list(self._acts)
grads = torch.autograd.grad(score, [self._acts[n] for n in names])          # dy^c / dA^k_ij

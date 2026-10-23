x = x.clone().requires_grad_(True)
class_idx = torch.as_tensor(class_idx, device=x.device).long().reshape(-1).expand(x.size(0))
logits = model(x)
score = logits[torch.arange(x.size(0), device=x.device), class_idx].sum()
(grad,) = torch.autograd.grad(score, x)               # dy^c / dx, same shape as the image
sal = grad.abs().amax(dim=1, keepdim=True)            # max over RGB channels

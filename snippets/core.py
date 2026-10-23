alpha = G.mean(dim=(2, 3), keepdim=True)              # alpha_k^c = (1/Z) sum_ij dy^c/dA^k_ij
cam = F.relu((alpha * A).sum(dim=1, keepdim=True))    # ReLU( sum_k alpha_k^c A^k )

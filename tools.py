import torch
import logging


def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True,
                                   only_inputs=True, )[0]
    else:
        return gradients(gradients(u, x), x, order=order - 1)


# 定义网络
class MLP(torch.nn.Module):
    def __init__(self, seq=None):
        super(MLP, self).__init__()
        if seq is None:
            seq = [1, 50, 50, 50, 50, 1]
        seq = [(seq[i], seq[i + 1]) for i in range(len(seq) - 1)]
        mod_seq = []
        for s in seq[:-1]:
            mod_seq.append(torch.nn.Linear(s[0], s[1]))
            mod_seq.append(torch.nn.Tanh())
        s = seq[-1]
        mod_seq.append(torch.nn.Linear(s[0], s[1]))
        self.net = torch.nn.Sequential(*mod_seq)

    def forward(self, x):
        return self.net(x)


log_format = '[%(asctime)s] [%(levelname)s] %(message)s'
handlers = [logging.FileHandler('train.log', mode='a'), logging.StreamHandler()]
logging.basicConfig(format=log_format, level=logging.INFO, datefmt='%d-%b-%y %H:%M:%S', handlers=handlers)
logger = logging.getLogger(__name__)

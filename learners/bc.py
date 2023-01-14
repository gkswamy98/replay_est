import numpy as np
import torch
from torch import optim, nn
import tqdm

def BC(D_E, pi, loss_fn=nn.MSELoss(), lr=3e-4, steps=int(1e5)):
    optimizer = optim.Adam(pi.parameters(), lr=lr)

    for step in tqdm.tqdm(range(steps)):
        tup = next(iter(D_E))
        states, actions = tup['obs'].to('cuda:0'), tup['acts'].to('cuda:0')
        
        optimizer.zero_grad()
        outputs = pi(states.float())
        loss = loss_fn(outputs, actions.float())
        loss.backward()
        optimizer.step()
    return pi
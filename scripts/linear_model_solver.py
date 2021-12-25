import os
import sys
import tqdm
import torch
import torch.nn as nn
from rich.console import Console

sys.path.append("../modules")

from linear_model import LinearModel


console = Console()

NDIM = 4
NUM_DATA = 1000
BS = 4
NUM_EPOCHS = 5
MAX_NORM = 0.9
VERBOSE_STEP = NUM_DATA // BS // 5


def main():
    # load model
    lm = LinearModel(NDIM, init_mode="random")
    console.rule("[bold red] Initial model parameters")
    console.print(list(lm.named_parameters()))
    console.rule("")

    # load loss function
    criterion = nn.MSELoss()

    # load optimizer
    optim = torch.optim.SGD(lm.parameters(), lr=1e-2)

    # load scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.8)

    # simulate training data and targets
    train_x = torch.randn(NUM_DATA, NDIM)
    train_y = torch.sum(train_x, dim=1).unsqueeze(-1)

    # main training loop
    for e in range(NUM_EPOCHS):
        console.print("[>] [blue]Epoch {}...".format(e))
        for i in tqdm.tqdm(range(0, NUM_DATA, BS)):
            x_ = train_x[i : i + BS]
            y_ = train_y[i : i + BS]

            y_pred = lm(x_)

            loss = criterion(y_pred, y_)

            if i % (BS * VERBOSE_STEP) == 0:
                print("Loss at step {}: {:.6f}".format(i // BS, loss.item()))

            # back-propagation
            optim.zero_grad()  # clear previous gradients
            loss.backward()  # calculate current gradients
            torch.nn.utils.clip_grad_norm_(lm.parameters(), MAX_NORM)  # clip gradients
            optim.step()  # optimize the current parameters

        console.print("[-] Current loss: {:.6f}".format(loss.item()))
        console.print("[-] Current LR: {:.6f}".format(optim.param_groups[0]["lr"]))
        scheduler.step()  # update LR

    console.rule("[bold green] Parameters after training")
    console.print(list(lm.named_parameters()))
    console.rule("")


if __name__ == "__main__":
    main()
from __future__ import annotations

import typer

from . import eval as eval_cli
from . import orchestrate as orchestrate_cli
from . import train as train_cli

app = typer.Typer(help="Adaptive DPO command suite.")
app.add_typer(train_cli.app, name="train")
app.add_typer(eval_cli.app, name="eval")
app.add_typer(orchestrate_cli.app, name="orchestrate")


def entrypoint():
    app()


import pathlib
import click
import os

from model_algorithm.data import fetch_train_test_split
from model_algorithm.models import train, serialize


@click.command()
@click.argument('output', type=click.Path(exists=False, file_okay=False, dir_okay=True))
@click.option('--id', type=str, help="The model id", default="xgboost")
@click.option('--seed', type=int, help="The random seed", default=930525)
def main(output: str, id: str = "xgboost", seed: int = 930525):
    output = pathlib.Path(output)
    output.mkdir(exist_ok=True)
    train_test = fetch_train_test_split(random_seed=seed)
    model = train(train_test)
    serialize(bst=model, output_folder=output, model_id=id)

if __name__ == '__main__':
    main()
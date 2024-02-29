from src.utils import get_transforms, train_one_experiment
from results import best_configs

device = "cuda:0"
workers = 12
epochss = [3, 5, 10, 15]
epochs_reduce_lr = [6, 11]
experiments = [
    x["config"]
    for ncls in [1, 2, 4]
    for x in best_configs(ncls=ncls, nbest=3, nepochs=1, showtxt=False)
]
transforms_train, transforms_test, _ = get_transforms()

for experiment in experiments:
    train_one_experiment(
        experiment=experiment,
        epochss=epochss,
        epochs_reduce_lr=epochs_reduce_lr,
        transforms_train=transforms_train,
        transforms_val=transforms_test,
        transforms_test=transforms_test,
        workers=workers,
        device=device,
        without=None,
        store=False,
    )

from src.utils import get_transforms, train_one_experiment
from results import best_configs

device = "cuda:0"
workers = 12
ncls_list = [1, 2, 4]
epochs_reduce_lr = [6, 11]
transforms_train, transforms_val, _ = get_transforms()

for ncls in ncls_list:
    experiment = best_configs(ncls=ncls, nbest=1, nepochs=None, showtxt=False)[0]
    print(f"train and store the {ncls}-class model")
    train_one_experiment(
        experiment=experiment["config"],
        epochss=[experiment["epochs"]],
        epochs_reduce_lr=epochs_reduce_lr,
        transforms_train=transforms_train,
        transforms_val=transforms_val,
        transforms_test=transforms_val,
        workers=workers,
        device=device,
        without=None,
        store=True,
    )

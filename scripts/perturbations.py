from torch.utils.data import DataLoader
from torchvision import transforms

from src.data import EvaluationDataset
from src.utils import (
    seed_everything,
    get_transforms,
    get_our_trained_model,
    get_generators,
    evaluation,
)

device = "cuda:0"
_, transforms, _ = get_transforms()
for perturb in ["blur", "crop", "compress", "noise", "combined"]:
    seed_everything(0)
    test = [
        (
            g,
            DataLoader(
                EvaluationDataset(g, transforms=transforms, perturb=perturb),
                batch_size=128,
                shuffle=False,
                num_workers=12,
                pin_memory=True,
                drop_last=False,
            ),
        )
        for g in get_generators()
    ]

    for ncls in [1, 2, 4]:
        print(f"{ncls}-class --- {perturb}")
        model = get_our_trained_model(ncls=ncls, device=device)
        model.to(device)
        evaluation(
            model,
            test,
            device,
            training="progan",
            ours=True,
            filename=f"results/perturbations/{perturb}_{ncls}class.pickle",
        )

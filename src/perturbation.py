from torchvision import transforms
import numpy as np
import cv2


def perturbation(kind):
    if kind == "blur":
        p = transforms.Lambda(lambda img: blur(np.array(img)))
    elif kind == "crop":
        p = transforms.Lambda(lambda img: cropping(np.array(img)))
    elif kind == "compress":
        p = transforms.Lambda(lambda img: jpeg(np.array(img)))
    elif kind == "noise":
        p = transforms.Lambda(lambda img: noise(np.array(img)))
    elif kind == "combined":
        return transforms.Compose(
            [
                transforms.Lambda(lambda img: blur(np.array(img))),
                transforms.Lambda(lambda img: cropping(np.array(img))),
                transforms.Lambda(lambda img: jpeg(np.array(img))),
                transforms.Lambda(lambda img: noise(np.array(img))),
                transforms.ToPILImage(),
                transforms.Resize(size=(256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
    return transforms.Compose(
        [
            p,
            transforms.ToPILImage(),
            transforms.Resize(size=(256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def noise(image):
    # variance from U[5.0,20.0]
    variance = np.random.uniform(low=5.0, high=20.0)
    image = np.copy(image).astype(np.float64)
    noise = variance * np.random.randn(*image.shape)
    image += noise
    return np.clip(image, 0.0, 255.0).astype(np.uint8)


def blur(image):
    # kernel size from [3, 5, 7, 9]
    kernel_size = np.random.choice([3, 5, 7, 9])
    return cv2.GaussianBlur(
        image, (kernel_size, kernel_size), sigmaX=cv2.BORDER_DEFAULT
    )


def jpeg(image):
    # qualite factor sampled from U[10, 75]
    factor = np.random.randint(low=10, high=75)
    _, image = cv2.imencode(".jpg", image, [factor, 90])
    return cv2.imdecode(image, 1)


def cropping(image):
    # crop between 5% and 20%
    percentage = np.random.uniform(low=0.05, high=0.2)
    x, y, _ = image.shape
    x_crop = int(x * percentage * 0.5)
    y_crop = int(y * percentage * 0.5)
    cropped = image[x_crop:-x_crop, y_crop:-y_crop]
    resized = cv2.resize(cropped, dsize=(x, y), interpolation=cv2.INTER_CUBIC)
    return resized

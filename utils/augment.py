import numpy as np
import torch


class Cutout:
    def __init__(self, size=16) -> None:
        self.size = int(size)

    def __call__(self, image):
        if torch.is_tensor(image):
            height, width = image.shape[-2:]
            center_y = np.random.randint(0, height)
            center_x = np.random.randint(0, width)

            y0 = max(0, center_y - self.size // 2)
            x0 = max(0, center_x - self.size // 2)
            y1 = min(height, center_y + self.size // 2)
            x1 = min(width, center_x + self.size // 2)

            output = image.clone()
            if output.dim() >= 3 and output.shape[-3] >= 3:
                fill = torch.tensor([125, 122, 113], dtype=output.dtype, device=output.device)
                output[..., :3, y0:y1, x0:x1] = fill.view(3, 1, 1)
            else:
                output[..., y0:y1, x0:x1] = 0
            return output

        image = image.copy()
        width, height = image.size
        center_x = np.random.randint(0, width)
        center_y = np.random.randint(0, height)

        x0 = max(0, center_x - self.size // 2)
        y0 = max(0, center_y - self.size // 2)
        x1 = min(width, center_x + self.size // 2)
        y1 = min(height, center_y + self.size // 2)

        pixels = image.load()
        fill = (125, 122, 113, 0) if image.mode == "RGBA" else (125, 122, 113)
        for x in range(x0, x1):
            for y in range(y0, y1):
                pixels[x, y] = fill
        return image

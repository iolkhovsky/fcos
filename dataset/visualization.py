import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

from common.torch_utils import tensor2numpy


def plot_images_list(images_list, labels_list=None, cols=4, cmap='gray', figsize=(16, 16)):
    sample_size = len(images_list)
    rows = sample_size // cols
    if sample_size % cols:
        rows += 1
    fig, m_axs = plt.subplots(rows, cols, figsize=figsize)
    for idx, (img, c_ax) in enumerate(zip(images_list, m_axs.flatten())):
        c_ax.imshow(img, cmap=cmap)
        label = f"image #{idx}"
        if labels_list is not None:
            label = labels_list[idx]
        c_ax.set_title(f'{label}')
        c_ax.axis('off')


def visualize_boxes(image, boxes=[], labels=[], scores=[], codec=None, width=4): 
    boxes = torch.from_numpy(np.array(boxes))
    image_cwh = torch.from_numpy(
        np.transpose(image, [2, 0, 1])
    )
    boxes_, labels_, colors_ = None, None, None
    vis = image_cwh.clone()

    if len(boxes):
        for obj_idx, (box, label) in enumerate(zip(boxes, labels)):
            label = codec.decode(label)
            color = codec.color(label)
            score = 1.
            if len(scores):
                score = scores[obj_idx]
                label += f": {'{:.2f}'.format(score)}"
            
            obj_width = max(1, int(width * score))

            vis = torchvision.utils.draw_bounding_boxes(
                        image=vis,
                        boxes=torch.stack([box]),
                        labels=[label],
                        width=obj_width,
                        colors=color,
                    )
    return np.transpose(vis.numpy(), [1, 2, 0])


def visualize_batch(imgs_batch, boxes_batch=None, labels_batch=None, scores_batch=None, codec=None, return_images=False):
    images_bhwc = tensor2numpy(imgs_batch).astype(np.uint8)
    vis_images = []
    for image_idx, image in enumerate(images_bhwc):
        if boxes_batch and labels_batch:
            boxes = tensor2numpy(boxes_batch[image_idx])
            labels = tensor2numpy(labels_batch[image_idx])
            scores = []
            if scores_batch:
                scores = tensor2numpy(scores_batch[image_idx])
            
            image = visualize_boxes(image, boxes, labels, scores, codec=codec)
        vis_images.append(image)
    if return_images:
        return vis_images
    else:
        plot_images_list(vis_images)

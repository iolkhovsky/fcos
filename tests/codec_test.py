import torch

from fcos.codec import FcosDetectionsCodec


def test_codec_centers():
    scales = torch.tensor([(1, 2), (3, 4), (5, 6)])
    height, width = 2, 3

    centers = FcosDetectionsCodec._generate_centers(height, width, scales)
    assert list(centers.shape) == [len(scales), height, width, 2]

    for batch_idx in range(len(scales)):
        scale_y, scale_x = scales[batch_idx]
        for y_idx in range(height):
            for x_idx in range(width):
                assert scale_y * (y_idx + 0.5) == centers[batch_idx, y_idx, x_idx, 0]
                assert scale_x * (x_idx + 0.5) == centers[batch_idx, y_idx, x_idx, 1]


def test_codec_decode():
    codec = FcosDetectionsCodec((256, 256), None)

    scales = torch.tensor([(1., 2.), (3., 4.)])
    test_data = torch.tensor(
        [[1, 1, 1, 1],
        [1, 2, 3, 4],
        [4, 3, 2, 1],
        [2, 1, 5, 2]]
    )
    test_ltrb_2x2 = torch.reshape(
        test_data,
        shape=[1, 2, 2, 4]
    )
    test_ltrb = test_ltrb_2x2.repeat(2, 1, 1, 1)
    test_ltrb = torch.permute(test_ltrb, [0, 3, 1, 2])

    decoded = codec.decode(
        test_ltrb, scales
    )
    assert list(decoded.shape) == [len(scales), 4, 4]

    for img_idx, image_detections in enumerate(decoded):
        scale_y, scale_x = scales[img_idx]
        for det_idx, x1y1x2y2 in enumerate(image_detections):
            xmin, ymin, xmax, ymax = x1y1x2y2
            left, top, right, bottom = test_data[det_idx]

            y_center = det_idx // 2
            x_center = det_idx - y_center * 2
            x_center = (x_center + 0.5) * scale_x
            y_center = (y_center + 0.5) * scale_y

            assert xmin == x_center - left
            assert ymin == y_center - top
            assert xmax == x_center + right
            assert ymax == y_center + bottom

from PIL import Image
import numpy as np
import h5py


class LoadData:
    @staticmethod
    def test_data(img_path, only_img=False):
        img_path = str(img_path)
        img = Image.open(img_path).convert('RGB')

        if not only_img:
            amb_gt_path = img_path.replace('.jpg', '.npy').replace('images', 'amb_gt')
            amb_target = np.load(amb_gt_path)
            gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')
            gt_file = h5py.File(gt_path, 'r')
            target = np.asarray(gt_file['density'])
            # w, h = img.width, img.height
            # new_w, new_h = w // 32 * 32, h // 32 * 32
            # img = img.resize((new_w, new_h))
            # amb_target = cv2.resize(amb_target, (new_w, new_h)) * (h / new_h) * (w / new_w)
            # target = cv2.resize(target, (new_w, new_h)) * (h / new_h) * (w / new_w)

            return img, amb_target, target

        return img

    @staticmethod
    def train_data(img_path, crop_size, scale=None, mask=None, only_img=False):
        if not only_img:
            img, amb_target, target = LoadData.test_data(img_path)
        else:
            img = LoadData.test_data(img_path, only_img=True)
            amb_target = None
            target = None

        crop_size = (img.size[0] // 2 // 32 * 32, img.size[1] // 2 // 32 * 32) if not crop_size \
            else (crop_size, crop_size)
        crop_max_x, crop_max_y = img.size[0] - crop_size[0] - 1, img.size[1] - crop_size[1] - 1

        # dx = int(scale['x'] * crop_size[0])
        # dy = int(scale['y'] * crop_size[1])
        dx = int(scale['x'] * crop_max_x)
        dy = int(scale['y'] * crop_max_y)

        if not only_img:
            img = img.crop((dx, dy, crop_size[0] + dx, crop_size[1] + dy))
            amb_target = amb_target[dy: crop_size[1] + dy, dx: crop_size[0] + dx]
            target = target[dy: crop_size[1] + dy, dx: crop_size[0] + dx]
            if not isinstance(mask, type(None)):
                mask = mask[dy: crop_size[1] + dy, dx: crop_size[0] + dx]
            if scale['flip'] >= .5:
                amb_target = np.fliplr(amb_target)
                target = np.fliplr(target)
                if not isinstance(mask, type(None)):
                    mask = np.fliplr(mask)
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            return img, amb_target, target, mask
        else:
            img = img.crop((dx, dy, crop_size[0] + dx, crop_size[1] + dy))
            if scale['flip'] >= .5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            return img

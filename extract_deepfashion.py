import sys
import numpy as np
import caffe
import argparse
import cv2
from tqdm import tqdm
import os
from collections import OrderedDict
import subprocess
import cPickle


class ImageHelper:
    def __init__(self, S, L, means):
        self.S = S
        self.L = L
        self.means = means

    def prepare_image_and_grid_regions_for_network(self, fname, roi=None):
        # Extract image, resize at desired size, and extract roi region if
        # available. Then compute the rmac grid in the net format: ID X Y W H
        I, im_resized = self.load_and_prepare_image(fname, roi)
        if self.L == 0:
            # Encode query in mac format instead of rmac, so only one region
            # Regions are in ID X Y W H format
            R = np.zeros((1, 5), dtype=np.float32)
            R[0, 3] = im_resized.shape[1] - 1
            R[0, 4] = im_resized.shape[0] - 1
        else:
            # Get the region coordinates and feed them to the network.
            all_regions = []
            all_regions.append(self.get_rmac_region_coordinates(im_resized.shape[0], im_resized.shape[1], self.L))
            R = self.pack_regions_for_network(all_regions)
        return I, R

    def get_rmac_features(self, I, R, net):
        net.blobs['data'].reshape(I.shape[0], 3, int(I.shape[2]), int(I.shape[3]))
        net.blobs['data'].data[:] = I
        net.blobs['rois'].reshape(R.shape[0], R.shape[1])
        net.blobs['rois'].data[:] = R.astype(np.float32)
        net.forward(end='rmac/normalized')
        return np.squeeze(net.blobs['rmac/normalized'].data)

    def load_and_prepare_image(self, fname, roi=None):
        # Read image, get aspect ratio, and resize such as the largest side equals S
        im = cv2.imread(fname)
        im_size_hw = np.array(im.shape[0:2])
        ratio = float(self.S)/np.max(im_size_hw)
        new_size = tuple(np.round(im_size_hw * ratio).astype(np.int32))
        im_resized = cv2.resize(im, (new_size[1], new_size[0]))
        # If there is a roi, adapt the roi to the new size and crop. Do not rescale
        # the image once again
        if roi is not None:
            roi = np.round(roi * ratio).astype(np.int32)
            im_resized = im_resized[roi[1]:roi[3], roi[0]:roi[2], :]
        # Transpose for network and subtract mean
        I = im_resized.transpose(2, 0, 1) - self.means
        return I, im_resized

    def pack_regions_for_network(self, all_regions):
        n_regs = np.sum([len(e) for e in all_regions])
        R = np.zeros((n_regs, 5), dtype=np.float32)
        cnt = 0
        # There should be a check of overflow...
        for i, r in enumerate(all_regions):
            try:
                R[cnt:cnt + r.shape[0], 0] = i
                R[cnt:cnt + r.shape[0], 1:] = r
                cnt += r.shape[0]
            except:
                continue
        assert cnt == n_regs
        R = R[:n_regs]
        # regs where in xywh format. R is in xyxy format, where the last coordinate is included. Therefore...
        R[:n_regs, 3] = R[:n_regs, 1] + R[:n_regs, 3] - 1
        R[:n_regs, 4] = R[:n_regs, 2] + R[:n_regs, 4] - 1
        return R

    def get_rmac_region_coordinates(self, H, W, L):
        # Almost verbatim from Tolias et al Matlab implementation.
        # Could be heavily pythonized, but really not worth it...
        # Desired overlap of neighboring regions
        ovr = 0.4
        # Possible regions for the long dimension
        steps = np.array((2, 3, 4, 5, 6, 7), dtype=np.float32)
        w = np.minimum(H, W)

        b = (np.maximum(H, W) - w) / (steps - 1)
        # steps(idx) regions for long dimension. The +1 comes from Matlab
        # 1-indexing...
        idx = np.argmin(np.abs(((w**2 - w * b) / w**2) - ovr)) + 1

        # Region overplus per dimension
        Wd = 0
        Hd = 0
        if H < W:
            Wd = idx
        elif H > W:
            Hd = idx

        regions_xywh = []
        for l in range(1, L+1):
            wl = np.floor(2 * w / (l + 1))
            wl2 = np.floor(wl / 2 - 1)
            # Center coordinates
            if l + Wd - 1 > 0:
                b = (W - wl) / (l + Wd - 1)
            else:
                b = 0
            cenW = np.floor(wl2 + b * np.arange(l - 1 + Wd + 1)) - wl2
            # Center coordinates
            if l + Hd - 1 > 0:
                b = (H - wl) / (l + Hd - 1)
            else:
                b = 0
            cenH = np.floor(wl2 + b * np.arange(l - 1 + Hd + 1)) - wl2

            for i_ in cenH:
                for j_ in cenW:
                    regions_xywh.append([j_, i_, wl, wl])

        # Round the regions. Careful with the borders!
        for i in range(len(regions_xywh)):
            for j in range(4):
                regions_xywh[i][j] = int(round(regions_xywh[i][j]))
            if regions_xywh[i][0] + regions_xywh[i][2] > W:
                regions_xywh[i][0] -= ((regions_xywh[i][0] + regions_xywh[i][2]) - W)
            if regions_xywh[i][1] + regions_xywh[i][3] > H:
                regions_xywh[i][1] -= ((regions_xywh[i][1] + regions_xywh[i][3]) - H)
        return np.array(regions_xywh).astype(np.float32)


class Dataset:
    def __init__(self, path, eval_binary_path):
        self.path = path
        self.eval_binary_path = eval_binary_path
        # Some images from the Paris dataset are corrupted. Standard practice is
        # to ignore them
        self.load()

    def load(self):
        # Load the dataset GT
        self.img_root = '{0}/jpg/'.format(self.path)
        # Get the filenames without the extension
        self.img_filenames = [e[:-4] for e in np.sort(os.listdir(self.img_root))]

        self.N_images = len(self.img_filenames)
        anno_bbox = file('{0}/Anno/list_bbox_consumer2shop.txt'.format(self.path)).readlines()
        anno_landmark = file('{0}/Anno/list_landmarks_consumer2shop.txt'.format(self.path)).readlines()
        self.group_class_item = {}
        self.item_img = {}
        self.img_info = np.zeros((self.N_images, 5), dtype=np.string_)
        self.img_roi = {}
        self.img_landmark = np.zeros((self.N_images, 8, 5), dtype=np.float32)
        for i in range(self.N_images):
            bbox = anno_bbox[i+2].split()
            img_name = bbox[0].split('/')
            if (img_name[1], img_name[2]) not in self.group_class_item:
                self.group_class_item[(img_name[1], img_name[2])] = [img_name[3]]
            if img_name[3] not in self.group_class_item[(img_name[1], img_name[2])]:
                self.group_class_item[(img_name[1], img_name[2])].append(img_name[3])
            if img_name[3] not in self.item_img:
                self.item_img[img_name[3]] = [self.img_filenames[i]]
            if self.img_filenames[i] not in self.item_img[img_name[3]]:
                self.item_img[img_name[3]].append(self.img_filenames[i])
            self.img_info[i] = [img_name[1], img_name[2], img_name[3], bbox[1], bbox[2]]
            bbox = map(int, bbox[3:])
            landmark = anno_landmark[i+2].split()
            landmark = map(int, landmark[3:])
            for j in range(8):
                k = j*3
                if k < len(landmark) and landmark[k] == 0:
                    x = landmark[k+1]
                    y = landmark[k+2]
                    v = landmark[k]
                    if x < bbox[0]:
                         bbox[0] = x
                    if x > bbox[2]:
                         bbox[2] = x
                    if y < bbox[1]:
                         bbox[1] = y
                    if y > bbox[3]:
                         bbox[3] = y
                    self.img_landmark[i][j] = [x, y, x, y, v]
                else:
                    self.img_landmark[i][j] = [0, 0, 0, 0, 1]    
            self.img_roi[self.img_filenames[i]] = np.array([bbox[0], bbox[1], bbox[2], bbox[3]], dtype=np.float)

    def get_filename(self, i):
        return os.path.normpath("{0}/{1}.jpg".format(self.img_root, self.img_filenames[i]))

    def get_roi(self, i):
        return self.img_roi[self.img_filenames[i]]

    def get_landmark(self, i):
        return self.img_landmark[i]

def extract_features(dataset, image_helper, net, args):
    Ss = [args.S, ] if not args.multires else [args.S - 250, args.S, args.S + 250]

    # Second part, dataset
    for S in Ss:
        image_helper.S = S
        out_dataset_fname = "{0}/{1}_S{2}_L{3}_dataset.npy".format(args.temp_dir, args.dataset_name, S, args.L)
        if not os.path.exists(out_dataset_fname):
            dim_features = net.blobs['rmac/normalized'].data.shape[1]
            N_dataset = dataset.N_images
            features_dataset = np.zeros((N_dataset, dim_features), dtype=np.float32)
            for i in tqdm(range(N_dataset), file=sys.stdout, leave=False, dynamic_ncols=True):
                # Load image, process image, get image regions, feed into the network, get descriptor, and store
                roi = dataset.get_roi(i)
                I, R = image_helper.prepare_image_and_grid_regions_for_network(dataset.get_filename(i), roi=roi)
                im = cv2.imread(dataset.get_filename(i))
                im_size_hw = np.array(im.shape[0:2])
                ratio = float(S)/np.max(im_size_hw)
                landmark = dataset.get_landmark(i)
                for lm in landmark:
                    if lm[4] == 0:
                        lm = np.round(lm * ratio)
                        roi = np.round(roi * ratio)
                        x1 = max(0, lm[0] - roi[0] - 2)
                        y1 = max(0, lm[1] - roi[1] - 2)
                        x2 = min(roi[2] - roi[0], lm[2] - roi[0] + 2)
                        y2 = min(roi[3] - roi[1], lm[3] - roi[1] + 2)
                        np.vstack( (R, [0, x1, y1, x2, y2]) )
                features_dataset[i] = image_helper.get_rmac_features(I, R, net)
            np.save(out_dataset_fname, features_dataset)
    features_dataset = np.dstack([np.load("{0}/{1}_S{2}_L{3}_dataset.npy".format(args.temp_dir, args.dataset_name, S, args.L)) for S in Ss]).sum(axis=2)
    features_dataset /= np.sqrt((features_dataset * features_dataset).sum(axis=1))[:, None]
    # Restore the original scale
    image_helper.S = args.S
    return features_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Oxford / Paris')
    parser.add_argument('--gpu', type=int, required=True, help='GPU ID to use (e.g. 0)')
    parser.add_argument('--S', type=int, required=True, help='Resize larger side of image to S pixels (e.g. 800)')
    parser.add_argument('--L', type=int, required=True, help='Use L spatial levels (e.g. 2)')
    parser.add_argument('--proto', type=str, required=True, help='Path to the prototxt file')
    parser.add_argument('--weights', type=str, required=True, help='Path to the caffemodel file')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the Oxford / Paris directory')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name')
    parser.add_argument('--eval_binary', type=str, required=True, help='Path to the compute_ap binary to evaluate Oxford / Paris')
    parser.add_argument('--temp_dir', type=str, required=True, help='Path to a temporary directory to store features and scores')
    parser.add_argument('--multires', dest='multires', action='store_true', help='Enable multiresolution features')
    parser.add_argument('--aqe', type=int, required=False, help='Average query expansion with k neighbors')
    parser.add_argument('--dbe', type=int, required=False, help='Database expansion with k neighbors')
    parser.set_defaults(multires=False)
    args = parser.parse_args()

    if not os.path.exists(args.temp_dir):
        os.makedirs(args.temp_dir)

    # Load and reshape the means to subtract to the inputs
    args.means = np.array([103.93900299,  116.77899933,  123.68000031], dtype=np.float32)[None, :, None, None]

    # Configure caffe and load the network
    caffe.set_device(args.gpu)
    caffe.set_mode_gpu()
    net = caffe.Net(args.proto, args.weights, caffe.TEST)

    # Load the dataset and the image helper
    dataset = Dataset(args.dataset, args.eval_binary)
    image_helper = ImageHelper(args.S, args.L, args.means)

    # Extract features
    features_dataset = extract_features(dataset, image_helper, net, args)

    # Database side expansion?
    if args.dbe is not None and args.dbe > 0:
        # Extend the database features
        # With larger datasets this has to be done in a batched way.
        # and using smarter ways than sorting to take the top k results.
        # For 5k images, not really a problem to do it by brute force
        X = features_dataset.dot(features_dataset.T)
        idx = np.argsort(X, axis=1)[:, ::-1]
        weights = np.hstack(([1], (args.dbe - np.arange(0, args.dbe)) / float(args.dbe)))
        weights_sum = weights.sum()
        features_dataset = np.vstack([np.dot(weights, features_dataset[idx[i, :args.dbe + 1], :]) / weights_sum for i in range(len(features_dataset))])

    output = open('feat_street2shop.pkl', 'wb')
    cPickle.dump(features_dataset, output)
    cPickle.dump(["{0}.jpg".format(i) for i in dataset.img_filenames], output)
    output.close()



# python extract_deepfashion.py --gpu 0 --S 800 --L 2 --proto deploy_resnet101_normpython.prototxt --weights model.caffemodel --dataset ../py-faster-rcnn/data/DeepFashion/Consumer2Shop --eval_binary datasets/evaluation/compute_ap --temp_dir tmp --dataset_name deepfashion --multires --aqe 1 --dbe 20

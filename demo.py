import jittor as jt
if jt.has_cuda:
    jt.flags.use_cuda = 1
import argparse
from dataset_scenenet import SceneNetDataset
from model import LinkNet_BackBone_FuseNet
from tqdm import tqdm
from pathlib import Path
import cv2
import numpy as np

parser = argparse.ArgumentParser(description='LinkNet demo', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataroot', required=True, help='path to scans (should have subfolders photo, depth, dhac and label)')
parser.add_argument('--use_dhac', action='store_true', help='use dhac as depth channel')
parser.add_argument('--store_result', action='store_true', help='store prediction results')
parser.add_argument('--results_dir', type=str, default='./predictions/', help='saves results here.')


def confusion_matrix(gt, pred, num_labels, ignore_label=None):
    valid = (gt != ignore_label)
    return np.bincount(num_labels * gt[valid].astype(int) + pred[valid], minlength=num_labels**2).reshape(num_labels, num_labels)


def getAccuracyPercentage(conf_matrix):
    return 100.0 * np.diag(conf_matrix).sum() / conf_matrix.sum()


if __name__ == '__main__':
    opt = parser.parse_args()
    dataset = SceneNetDataset(opt)
    print("Dataset Loaded! Frames:", len(dataset))

    linknet = LinkNet_BackBone_FuseNet(dataset.num_labels, 4 if opt.use_dhac else 1)
    print("Loading model pretrain weights")
    if opt.use_dhac:
        linknet.load('weights/SceneNet_RGB_DHAC.pkl')
        print("Pretrained Model:'weights/SceneNet_RGB_DHAC.pkl' Loaded")
    # else:
    #     linknet.load('weights/SceneNet_RGB_Depth.pth')
    #     print("Pretrained Model:'weights/SceneNet_RGB_Depth.pth' Loaded")

    conf_mat = np.zeros((dataset.num_labels, dataset.num_labels), dtype=np.int)

    linknet.eval()
    with jt.no_grad():
        for rgb_image, depth_image, gtlabel, frameid in tqdm(dataset, desc="Predicting"):
            predlabel = linknet(rgb_image, depth_image)

            gtlabel = gtlabel[0].numpy()
            predlabel = predlabel[0].argmax(dim=0)[0].numpy()

            if opt.store_result:
                pred_folder = Path(opt.results_dir)
                if not pred_folder.exists():
                    pred_folder.mkdir(parents=True)
                pred_path = pred_folder / (str(int(frameid[0].numpy())) + '.png')
                cv2.imwrite(str(pred_path), predlabel.astype(np.uint8))

            conf_mat += confusion_matrix(gtlabel, predlabel, dataset.num_labels, dataset.ignore_label)

    print("Predict End. Overall Accuracy:%d" % (getAccuracyPercentage(conf_mat)))

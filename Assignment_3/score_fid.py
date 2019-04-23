import argparse
import os
import torchvision
import torchvision.transforms as transforms
import torch
import classify_svhn
from classify_svhn import Classifier
import numpy as np
from scipy import linalg
import scipy

SVHN_PATH = "svhn"
PROCESS_BATCH_SIZE = 32


def get_sample_loader(path, batch_size):
    """
    Loads data from `[path]/samples`
    - Ensure that path contains only one directory
      (This is due ot how the ImageFolder dataset loader
       works)
    - Ensure that ALL of your images are 32 x 32.
      The transform in this function will rescale it to
      32 x 32 if this is not the case.
    Returns an iterator over the tensors of the images
    of dimension (batch_size, 3, 32, 32)
    """
    data = torchvision.datasets.ImageFolder(
        path,
        transform=transforms.Compose([
            transforms.Resize((32, 32), interpolation=2),
            classify_svhn.image_transform
        ])
    )
    data_loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        num_workers=2,
    )
    return data_loader


def get_test_loader(batch_size):
    """
    Downloads (if it doesn't already exist) SVHN test into
    [pwd]/svhn.
    Returns an iterator over the tensors of the images
    of dimension (batch_size, 3, 32, 32)
    """
    testset = torchvision.datasets.SVHN(
        SVHN_PATH, split='test',
        download=True,
        transform=classify_svhn.image_transform
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
    )
    return testloader


def extract_features(classifier, data_loader):
    """
    Iterator of features for each image.
    """
    with torch.no_grad():
        for x, _ in data_loader:
            h = classifier.extract_features(x).numpy()
            for i in range(h.shape[0]):
                yield h[i]

# def calculate_fid_score(sample_feature_iterator,
#                         testset_feature_iterator):
#     """
    
#     """
#     samples = np.array(list(sample_feature_iterator))
#     testset = np.array(list(testset_feature_iterator))
#     #print("Done creating arrays")
#     # means
#     samples_mu = np.mean(samples, axis=0)
#     testset_mu = np.mean(testset, axis=0)
#     # covariance matrices
#     samples_S = np.cov(samples, rowvar=False)
#     testset_S = np.cov(testset, rowvar=False)
    
#     # FID
#     return np.sqrt(np.sum((samples_mu - testset_mu) ** 2) + \
#                     np.trace(samples_S + testset_S - 2 * \
#                              scipy.linalg.sqrtm(samples_S @ testset_S 
#                                                 + 1e-4 * np.eye(samples_mu.shape[0]))))
#     raise NotImplementedError(
#         "TO BE IMPLEMENTED."
#         "Part of Assignment 3 Quantitative Evaluations"
# )

def calculate_fid_score(sample_feature_iterator,
                        testset_feature_iterator):

    samples_s = []
    samples_t = []

    for s in sample_feature_iterator:
        # print(s.shape)
        samples_s.append(s)

    for t in testset_feature_iterator:
        # print(t.shape)
        samples_t.append(t)

    samples_s =  np.array(samples_s)
    samples_t =  np.array(samples_t)

    print('samples')
    print(samples_s.shape)
    print(samples_t.shape)

    mu_1 = np.mean(samples_s, axis=0)
    mu_2 = np.mean(samples_t, axis=0)

    print('mu shape')
    print(mu_1.shape)
    print(mu_2.shape)

    diff = mu_1 - mu_2

    sigma1 = np.cov(samples_s, rowvar=False)
    sigma2 = np.cov(samples_t, rowvar=False)
    print('sigma')
    print(sigma1.shape)
    print(sigma2.shape)

    result = np.sqrt(np.sum((mu_1 - mu_2) ** 2) + np.trace(sigma1 + sigma2 - 2 * scipy.linalg.sqrtm(sigma1 @ sigma2 + 1e-4 * np.eye(mu_1.shape[0]))))
    return result



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Score a directory of images with the FID score.')
    parser.add_argument('--model', type=str, default="svhn_classifier.pt",
                        help='Path to feature extraction model.')
    parser.add_argument('--directory', type=str, default="/network/home/guptagun/code/dl/images/fid/samples/",
                        help='Path to image directory')
    args = parser.parse_args()

    quit = False
    if not os.path.isfile(args.model):
        print("Model file " + args.model + " does not exist.")
        quit = True
    if not os.path.isdir(args.directory):
        print("Directory " + args.directory + " does not exist.")
        quit = True
    if quit:
        exit()
    print("Test")
    classifier = torch.load(args.model, map_location='cpu')
    classifier.eval()

    sample_loader = get_sample_loader(args.directory,
                                      PROCESS_BATCH_SIZE)
    sample_f = extract_features(classifier, sample_loader)

    test_loader = get_test_loader(PROCESS_BATCH_SIZE)
    test_f = extract_features(classifier, test_loader)

    fid_score = calculate_fid_score(sample_f, test_f)
    print("FID score:", fid_score)
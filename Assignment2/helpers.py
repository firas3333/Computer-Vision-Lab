import cv2
import numpy as np
from glob import glob
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt


class ImageHelpers:
    def __init__(self):

        self.sift_object = cv2.xfeatures2d.SIFT_create()

    def gray(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray

    def features(self, image):
        keypoints, descriptors = self.sift_object.detectAndCompute(image, None)
        return [keypoints, descriptors]


class BOVHelpers:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.kmeans_obj = KMeans(n_clusters=n_clusters)
        self.kmeans_ret = None
        self.descriptor_vstack = None
        self.mega_histogram = None
        self.clf = svm.LinearSVC()

    def cluster(self):
        # cluster using KMeans


        self.kmeans_obj = KMeans(self.n_clusters,tol=0.000001)
        self.kmeans_ret = self.kmeans_obj.fit_predict(self.descriptor_vstack)

    def developVocabulary(self, n_images, descriptor_list, kmeans_ret=None):
        self.mega_histogram = np.array([np.zeros(self.n_clusters) for i in range(n_images)])
        old_count = 0
        for i in range(n_images):
            l = len(descriptor_list[i])
            for j in range(l):
                if kmeans_ret is None:
                    idx = self.kmeans_ret[old_count + j]
                else:
                    idx = kmeans_ret[old_count + j]
                self.mega_histogram[i][idx] += 1
            old_count += l
        print ("Vocabulary Histogram Done")

    def standardize(self, std=None):
        if std is None:
            self.scale = StandardScaler().fit(self.mega_histogram)
            self.mega_histogram = self.scale.transform(self.mega_histogram)
        else:
            print ("STD not none. External STD supplied")
            self.mega_histogram = std.transform(self.mega_histogram)

    def formatND(self, l):
        vStack = np.array(l[0])
        for remaining in l[1:]:
            vStack = np.vstack((vStack, remaining))
        self.descriptor_vstack = vStack.copy()
        return vStack
    def train(self, train_labels):
        print ("Training SVM")
        print ("Train labels", train_labels)
        self.clf.fit(self.mega_histogram, train_labels)
        print ("Training completed")

    def predict(self, iplist):
        predictions = self.clf.predict(iplist)
        return predictions
    def plotHist(self, vocabulary=None):
        print ("Plotting histogram")
        if vocabulary is None:
            vocabulary = self.mega_histogram

        x_scalar = np.arange(self.n_clusters)
        y_scalar = np.array([abs(np.sum(vocabulary[:, h], dtype=np.int32)) for h in range(self.n_clusters)])
        plt.bar(x_scalar, y_scalar)
        plt.xlabel("Visual Word Index")
        plt.ylabel("Frequency")
        plt.title("Complete Vocabulary Generated")
        plt.xticks(x_scalar + 0.4, x_scalar)
        plt.show()


class FileHelpers:
    def __init__(self):
        pass

    def getFiles(self, path):
        imlist = {}
        count = 0
        for each in glob(path + "/*"):
                word = each.split("/")[-1]
                print (" #### Reading image category ", word, " ##### ")
                imlist[word] = []
                for imagefile in glob(word +"/*"):
                    im = cv2.imread(imagefile, 0)
                    imlist[word].append(im)
                    count += 1
        return [imlist, count]

from helpers import *
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

class BOV:
    def __init__(self, no_clusters):
        self.no_clusters = no_clusters
        self.train_path = None
        self.test_path = None
        self.im_helper = ImageHelpers()
        self.bov_helper = BOVHelpers(no_clusters)
        self.file_helper = FileHelpers()
        self.images = None
        self.trainImageCount = 0
        self.train_labels = np.array([])
        self.name_dict = {}
        self.descriptor_list = []

    def trainModel(self):
        # read file. prepare file lists.
        self.images, self.trainImageCount= self.file_helper.getFiles(self.train_path)
        # extract SIFT Features from each image
        label_count = 0
        for word, imlist in self.images.items():
            self.name_dict[str(label_count)] = word
            print ("Computing Features for ", word)
            for im in imlist:
                # cv2.imshow("im", im)
                # cv2.waitKey()
                self.train_labels = np.append(self.train_labels, label_count)
                kp, des = self.im_helper.features(im)
                self.descriptor_list.append(des)

            label_count += 1
        # perform clustering
        bov_descriptor_stack = self.bov_helper.formatND(self.descriptor_list)

        self.bov_helper.cluster()

        self.bov_helper.developVocabulary(n_images=self.trainImageCount, descriptor_list=self.descriptor_list)
        # show vocabulary trained
        self.bov_helper.plotHist()

        self.bov_helper.standardize()

        self.bov_helper.train(self.train_labels)


    def recognize(self, test_img, test_image_path=None):
        kp, des = self.im_helper.features(test_img)
        # generate vocab for test image
        vocab = np.array([[0 for i in range(self.no_clusters)]])
        # locate nearest clusters for each of
        # word (feature) present in the image
        # kmeans nearest clusters for N features
        test_ret = self.bov_helper.kmeans_obj.predict(des)
        for each in test_ret:
            vocab[0][each] += 1
        # Scale the features
        vocab = self.bov_helper.scale.transform(vocab)
        # predict
        lb = self.bov_helper.clf.predict(vocab)
        return lb

    def testModel(self):
        true=[]
        self.testImages, self.testImageCount = self.file_helper.getFiles(self.test_path)
        predictions = []
        preds=[]
        for word, imlist in self.testImages.items():
            print ("processing ", word)
            for im in imlist:
                # print imlist[0].shape, imlist[1].shape
                cl = self.recognize(im)
                if 'Airplane' in word:
                    true.append(0.0)
                if 'Elephant' in word:
                    true.append(1.0)
                if 'MotorBike' in word:
                    true.append(2.0)
                preds.append(cl[0])
                predictions.append({
                    'image': im,
                    'class': cl,
                    'object_name': self.name_dict[str(int(cl[0]))]

                })

        for each in predictions:
            # cv2.imshow(each['object_name'], each['image'])
            # cv2.waitKey()
            # cv2.destroyWindow(each['object_name'])
            #
            plt.imshow(cv2.cvtColor(each['image'], cv2.COLOR_GRAY2RGB))
            plt.title(each['object_name'])
            plt.show()
        a = confusion_matrix(true, preds)
        # for i in range (0,3):
        #     tp=a[i][i]
        #     if i==0:
        #         fn=a[i][i+1]+a[i][i+2]
        #         fp=a[i+1][i]+a[i+2][i]
        #     if i==1:
        #         fn = a[i][i - 1] + a[i][i +1]
        #         fp = a[i - 1][i] + a[i + 1][i]
        #     if i==2:
        #         fn = a[i][i - 1] + a[i][i - 2]
        #         fp = a[i - 1][i] + a[i - 2][i]
        #     recall=tp/(tp+fn)
        #     precision=tp/(tp+fp)
            # f=open("threshroc","a+")
            # f.write("%f\n" %recall)
            # f.write("%f\n" %precision)
            # f.write("--------------\n")
        print (a)
        labels = ['airplane', 'elephant','motorbike']
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(a)
        plt.title('Confusion matrix')
        fig.colorbar(cax)
        ax.set_xticklabels([''] + labels)
        ax.set_yticklabels([''] + labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
    def print_vars(self):
        pass


if __name__ == '__main__':
    bov = BOV(no_clusters=50)
    # set training path
    bov.train_path = 'train'
    # set testing path
    bov.test_path = 'test'
    # train
    bov.trainModel()
    # test
    bov.testModel()

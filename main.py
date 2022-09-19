import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


def KNN(norm,label_for_train,data_for_test_nor,label_for_test):
    #K3 + 1 due to 3 species
    K1 = np.zeros([30,])
    K2 = np.zeros([30,])
    K3 = np.zeros([30,])
    K4 = np.zeros([30,])
    avg_acc = 0.0


    for i in range(data_for_test_nor.shape[0]):

        K1[i] = np.argmin(norm[i])
        norm[i][int(K1[i])] = np.inf
        K2[i] = np.argmin(norm[i])
        norm[i][int(K2[i])] = np.inf
        K3[i] = np.argmin(norm[i])
        norm[i][int(K2[i])] = np.inf

        label_K1 = label_for_train[int(K1[i])]
        label_K2 = label_for_train[int(K2[i])]
        label_K3 = label_for_train[int(K3[i])]

        label_arr = [label_K1,label_K2,label_K3]
        setosa_num = 0
        versicolor_num = 0
        virginica_num = 0
        hypothesis = None


        for j in label_arr:
            if j == 'Iris-setosa':
                setosa_num += 1
                if setosa_num == 2:
                    hypothesis = 'Iris-setosa'

            elif j == 'Iris-versicolor':
                versicolor_num += 1
                if versicolor_num == 2:
                    hypothesis = 'Iris-versicolor'
            else:
                virginica_num += 1
                if virginica_num == 2:
                    hypothesis = 'Iris-virginica'

        # print('test data {}  = {} | pred = {}'.format(i+1, hypothesis, hypothesis == label_for_test[i]))

        if label_K1 != label_K2 != label_K3:
            K4[i] = np.argmin(norm[i])
            label_K4 = label_for_train[int(K4[i])]
            hypothesis = label_K4[0]


        avg_acc += hypothesis == label_for_test[i]


    # print(f"model accuracy : {avg_acc / len(label_for_test)}")
    return avg_acc / len(label_for_test)

def KNN_argmax(norm,label_for_train,data_for_test_nor,label_for_test): #argmin -> argmax
    #K3 + 1 due to 3 species
    K1 = np.zeros([30,])
    K2 = np.zeros([30,])
    K3 = np.zeros([30,])
    K4 = np.zeros([30,])
    avg_acc = 0.0


    for i in range(data_for_test_nor.shape[0]):

        K1[i] = np.argmax(norm[i])
        norm[i][int(K1[i])] = np.inf
        K2[i] = np.argmax(norm[i])
        norm[i][int(K2[i])] = np.inf
        K3[i] = np.argmax(norm[i])
        norm[i][int(K2[i])] = np.inf

        label_K1 = label_for_train[int(K1[i])]
        label_K2 = label_for_train[int(K2[i])]
        label_K3 = label_for_train[int(K3[i])]

        label_arr = [label_K1,label_K2,label_K3]
        setosa_num = 0
        versicolor_num = 0
        virginica_num = 0
        hypothesis = None

        if label_K1 != label_K2 != label_K3:
            K4[i] = np.argmax(norm[i])
            label_K4 = label_for_train[int(K4[i])]
            hypothesis = label_K4[0]


        for j in label_arr:
            if j == 'Iris-setosa':
                setosa_num += 1
                if setosa_num == 2:
                    hypothesis = 'Iris-setosa'

            elif j == 'Iris-versicolor':
                versicolor_num += 1
                if versicolor_num == 2:
                    hypothesis = 'Iris-versicolor'
            else:
                virginica_num += 1
                if virginica_num == 2:
                    hypothesis = 'Iris-virginica'

        # print('test data {}  = {} | pred = {}'.format(i+1, hypothesis, hypothesis == label_for_test[i]))
        avg_acc += hypothesis == label_for_test[i]


    # print(f"model accuracy : {avg_acc / len(label_for_test)}")

    return avg_acc / len(label_for_test)


if __name__ == "__main__":

    avg_acc_1, avg_acc_2, avg_acc_3, avg_acc_4 = 0.0, 0.0, 0.0, 0.0

    for _ in tqdm(range(100)):
        csv_data = np.loadtxt('Iris.csv', delimiter=',', dtype='str')[1:]
        np.random.shuffle(csv_data)

        # split into train:test = 120:30
        data_num = csv_data[:, 1:5].astype(np.float32)
        label = csv_data[:, 5:]
        data_for_train = data_num[:120]
        label_for_train = label[:120]
        data_for_test = data_num[120:]
        label_for_test = label[120:]

        # for Z score
        mean = np.mean(data_for_train, axis=0, dtype=np.float32)
        std = np.std(data_for_train, axis=0, dtype=np.float32)

        data_for_train_nor = (data_for_train - mean) / std
        data_for_test_nor = (data_for_test - mean) / std

        # for plot
        """
        df = pd.DataFrame(data_for_train_nor, columns= ['sepal length', 'sepal width', 'petal length', 'petal width'])
    
        pca = PCA(n_components = 2)
        pca.fit(df.iloc[:,:-1])
    
        df_pca = pca.transform(df.iloc[:,:-1])
        df_pca = pd.DataFrame(df_pca, columns = ['component 0', 'component 1'])
        print(df_pca[0])
        #plt.scatter(df_pca[0],df_pca[1])
    
    
        """

        # L1-norm
        norm1 = np.zeros([30, 120])
        for i in range(data_for_test_nor.shape[0]):
            for j in range(data_for_train_nor.shape[0]):
                norm1[i][j] = abs(np.sum(data_for_train_nor[j] - data_for_test_nor[i]))

        # L2-norm

        norm2 = np.zeros([30, 120])
        for i in range(data_for_test_nor.shape[0]):
            for j in range(data_for_train_nor.shape[0]):
                norm2[i][j] = np.sqrt(abs(np.sum(np.square(data_for_train_nor[j]) - np.square(data_for_test_nor[i]))))

        # cosine similarity  ** using no normalization data for positive domain & KNN_argmax
        norm3 = np.zeros([30, 120])
        for i in range(data_for_test.shape[0]):
            for j in range(data_for_train.shape[0]):
                norm3[i][j] = np.dot(data_for_train[j], data_for_test[i]) / (
                            np.sqrt(np.sum(np.square(data_for_train[j]))) * np.sqrt(np.sum(np.square(data_for_test[i]))))

        # inner product ** using normalization data & KNN_argmax
        norm4 = np.zeros([30, 120])
        for i in range(data_for_test_nor.shape[0]):
            for j in range(data_for_train_nor.shape[0]):
                norm4[i][j] = np.dot(data_for_train_nor[j], data_for_test_nor[i])

        avg_acc_1 += KNN(norm1, label_for_train, data_for_test_nor, label_for_test)
        avg_acc_2 += KNN(norm2, label_for_train, data_for_test_nor, label_for_test)
        avg_acc_3 += KNN_argmax(norm3, label_for_train, data_for_test, label_for_test)
        avg_acc_4 += KNN_argmax(norm4, label_for_train, data_for_test_nor, label_for_test)


    x = ["l1_norm", "l2_norm", "cos_sim", "inner_product"]
    y = [float(avg_acc_1) / 100, float(avg_acc_2) / 100, float(avg_acc_3) / 100, float(avg_acc_4) / 100]
    plt.title(f"Accuracy ({100} iter)")
    plt.bar(x, y)
    plt.xlabel("metric")
    plt.ylabel("acc")

    plt.show()



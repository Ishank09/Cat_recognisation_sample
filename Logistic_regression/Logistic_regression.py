import h5py
import numpy as np
from matplotlib import pyplot as plt

def load_dataset():
    train_dataset = h5py.File('dataset/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels
 
    test_dataset = h5py.File('dataset/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels
 
    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def info_dataset( train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes):
    print('before Dim change')
    print('Number of training examples: m_train = ',train_set_x_orig.shape[0])
    print('Number of testing examples: m_test = ',test_set_x_orig.shape[0])
    print('Height/Width of each image: num_px = ',test_set_x_orig.shape[1])
    print('Each image is of size: (' , train_set_x_orig.shape[1] , ',' , train_set_x_orig.shape[2] ,',', train_set_x_orig.shape[3] , ')')
    print('train x shape ', train_set_x_orig.shape)
    print('train y shape', train_set_y_orig.shape)
    print('test x shape ', test_set_x_orig.shape)
    print('test y shape', test_set_y_orig.shape)
    print('---------------------------------------')


#For convenience, you should now reshape images of shape (num_px, num_px, 3) in a numpy-array of shape (num_px ∗∗ num_px ∗∗ 3, 1). After this, our training (and test) dataset is a numpy-array where each column represents a flattened image. There should be m_train (respectively m_test) columns.
#Exercise: Reshape the training and test data sets so that images of size (num_px, num_px, 3) are flattened into single vectors of shape (num_px ∗∗ num_px ∗∗ 3, 1).
#A trick when you want to flatten a matrix X of shape (a,b,c,d) to a matrix X_flatten of shape (b∗∗c∗∗d, a) is to use: 
def change_dim( train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes):
    print('after Dim change')
    train_set_x_orig = train_set_x_orig.reshape(train_set_x_orig.shape[0] , -1).T
    test_set_x_orig = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
    print('train x shape ', train_set_x_orig.shape)
    print('train y shape', train_set_y_orig.shape)
    print('test x shape ', test_set_x_orig.shape)
    print('test y shape', test_set_y_orig.shape)
    print('---------------------------------------')
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

#To represent color images, the red, green and blue channels (RGB) must be specified for each pixel, and so the pixel value is actually a vector of three numbers ranging from 0 to 255.
#One common preprocessing step in machine learning is to center and standardize your dataset, meaning that you substract the mean of the whole numpy array from each example, and then divide each example by the standard deviation of the whole numpy array. But for picture datasets, it is simpler and more convenient and works almost as well to just divide every row of the dataset by 255 (the maximum value of a pixel channel).
#Let's standardize our dataset.
def preprocessing(train_set_x_orig, test_set_x_orig):
    train_set_x_orig = train_set_x_orig/255
    test_set_x_orig = test_set_x_orig/255
    return train_set_x_orig,  test_set_x_orig

 #implement sigmoid(). As you've seen in the figure above, you need to compute sigmoid(wTx+b)=11+e−(wTx+b)sigmoid(wTx+b)=11+e−(wTx+b) to make predictions. Use np.exp().
def sigmoid( z ):
    a = (1 / ( 1 + ( np.exp(-z) ) ))
    return a

def initialize(dim):
    w = np.zeros((dim,1))
    b = 0
    return w,b

#Now that your parameters are initialized, you can do the "forward" and "backward" propagation steps for learning the parameters.
#Exercise: Implement a function propagate() that computes the cost function and its gradient.
#Hints:
#Forward Propagation:
#    You get X
#    You compute A=σ(wTX+b)=(a(1),a(2),...,a(m−1),a(m))A=σ(wTX+b)=(a(1),a(2),...,a(m−1),a(m))
#    You calculate the cost function: J=−1m∑mi=1y(i)log(a(i))+(1−y(i))log(1−a(i))J=−1m∑i=1my(i)log⁡(a(i))+(1−y(i))log⁡(1−a(i))
#Here are the two formulas you will be using:
#∂J∂w=1mX(A−Y)T(7)
#∂J∂w=1mX(A−Y)T
#∂J∂b=1m∑i=1m(a(i)−y(i))
def propagate(w0 , w1 , x , y):
    m = x.shape[1]
    z = w0 + np.dot(x.T,w1)
    a = sigmoid(z)
    print(y.shape, '   ' , a.shape,'  ',w1.shape,'  ',x.shape)
    cost = -1/m * (np.sum(np.dot( y , np.log(a)) + np.dot((1-y) , np.log(a) )))
    dw0 = -1/m * ( np.sum( a - y.T )  )
    #print((a-y).shape,'  ',x.shape)
    dw1 = -1/m * ( np.sum( np.dot( a - y , x )))

    return cost,dw0,dw1

def optimize( w0, w1, learning_rate, num_iterations , x, y):
    for i in range(num_iterations):
        cost,dw0,dw1 = propagate(w0,w1,x,y)
        w0 = w0 - learning_rate * dw0
        w1 = w1 - learning_rate * dw1
        #print(cost)
    return w0,w1,dw0,dw1,cost

def main():
    print("CAT prediction program coursera")
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    info_dataset(train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes)
    index = 5
    plt.imshow(train_set_x_orig[index])
    print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")
    plt.show()
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = change_dim(train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes)
    train_set_x_orig, test_set_x_orig= preprocessing(train_set_x_orig, test_set_x_orig)
    w1, w0 = initialize(train_set_x_orig.shape[0])
    print(optimize(w0,w1,0.01,10,train_set_x_orig,train_set_y))


  
if __name__== "__main__":
  main()

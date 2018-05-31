#1st part -- divide the data file into test and train datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

train_spam_count = test_spam_count = probablity_train_spam = probablity_train_spam = \
probablity_train_not_spam = probablity_test_not_spam = 0.00

default_standard_deviation = 0.000001

input_data = np.loadtxt('spambase.data', delimiter=',', dtype = float)
#np.random.shuffle(input_data)
X, target = input_data[:,:-1], input_data[:,-1]

# Split the data into a training and test set.
# Each of these should have about 2,300 instances i.e., 50% of given data set
train_input, test_input, train_target, test_target = train_test_split(X, target, test_size=0.50, random_state=0)

#2nd part -- Create Probablistic Model
# for train input caclulating no. of spam mails
for i in range(len(train_target)):
    if (train_target[i] == 1):
        train_spam_count += 1
# for train input caclulating no. of spam mails
for i in range(len(test_target)):
    if (test_target[i] == 1):
        test_spam_count += 1
#calculate the probablity of spam mails:
probablity_train_spam = train_spam_count / len(train_target)
probablity_test_spam = test_spam_count/ len(test_target)
#not spam calculation
probablity_train_not_spam = 1 - probablity_train_spam
probablity_test_not_spam = 1 - probablity_test_spam

#dividing the feature into spam and not spam
mean_train_spam, mean_train_not_spam = [] , []
std_dev_train_spam , std_dev_train_not_spam = [] , []

for each_feature in range(train_input.shape[1]):
    feature_spam , feature_not_spam = []  , []
    for each_feature_row in range(len(train_target)):
        if (train_target[each_feature_row] == 1):
            feature_spam.append(train_input[each_feature_row][each_feature])
        else:
            feature_not_spam.append(train_input[each_feature_row][each_feature])
    mean_train_spam.append(np.mean(feature_spam))
    mean_train_not_spam.append(np.mean(feature_not_spam))
    std_dev_train_spam.append(np.std(feature_spam))
    std_dev_train_not_spam.append(np.std(feature_not_spam))

#keeping default value as 0.001 if any value is 0
'''default_standard_deviation
std_dev_train_spam[std_dev_train_spam == 0] = default_standard_deviation
std_dev_train_not_spam[std_dev_train_not_spam == 0] = default_standard_deviation
'''
#3rd Part - Gaussian Equation Implementation

def gauss_value(x,mean,std_deviation):
    if (std_deviation == 0):
        std_deviation = default_standard_deviation
    step_1 = 1.0/float(np.sqrt(2*np.pi)*std_deviation)
    if (step_1 <= 0.0000000000000000000000000000000000001):
        step_1 = 0.0000000000000000000000000000000000001
    step_2 = step_1 * float(np.exp(-((x-mean)**2)/(2*float(std_deviation**2))))

    if (step_2 <= 0.000000000000000000000000000000000000000000001):
        step_2 = 0.00000000000000000000000000000000000000001
    return step_2

class_x = 0
result = [] #to store test output predicted values

for each_row in range(len(test_input)):
    class_1 = np.log(probablity_train_spam)
    class_0 = np.log(probablity_train_not_spam)
    for each_feature in range(test_input.shape[1]):
        x = test_input[each_row][each_feature]
        class_1 += np.log(gauss_value(x,mean_train_spam[each_feature],std_dev_train_spam[each_feature]))
        class_0 += np.log(gauss_value(x,mean_train_not_spam[each_feature],std_dev_train_not_spam[each_feature]))
    class_x = np.argmax([class_0, class_1])
    result.append(class_x)

#Compute a confusion matrix for the test set
cfm = confusion_matrix(test_target, result)
print"Confusion Matrix: "
print cfm

#Calculating accuracy, precision, and recall on the test set
#TP = True Positive, TN = True Negative, FP = False Positive, FN = Flase Negative
TP,TN,FP,FN = 0,0,0,0

for row in range(len(result)):
	if (result[row] == 1 and test_target[row] == 1):
		TP += 1
	elif (result[row] == 0 and test_target[row] == 0 ):
		TN += 1
	elif (result[row] == 1 and test_target[row] == 0 ):
		FP += 1
	else:
		FN += 1

accuracy = float(TP + TN)/(TP+TN+FP+FN)
print ("Accuracy : ", accuracy)
precision = float(TP)/(TP+FP)
recall = float(TP)/(TP+FN)
print ("Precision: ", precision)
print ("Recall   : ", recall)
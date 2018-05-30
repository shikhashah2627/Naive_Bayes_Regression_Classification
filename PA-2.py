#1st part -- divide the data file into test and train datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


train_spam_count = test_spam_count = probablity_train_spam = probablity_train_spam = \
probablity_train_not_spam = probablity_test_not_spam = 0.00

default_standard_deviation = 0.0001

input_data = np.loadtxt('spambase.data', delimiter=',', dtype = float)
np.random.shuffle(input_data)
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
probablity_test_spam = test_spam_count / len(test_target)
#not spam calculation
probablity_train_not_spam = 1 - probablity_train_spam
probablity_test_not_spam = 1 - probablity_test_spam

#dividing the feature into spam and not spam
feature_spam , feature_not_spam = []  , []
mean_train_spam, mean_train_not_spam = [] , []
std_dev_train_spam , std_dev_train_not_spam = [] , []
for each_feature in range(train_input.shape[1]):
    for each_feature_row in range(len(train_target)):
        if (train_target[each_feature_row] == 1):
            feature_spam.append(train_input[each_feature_row][each_feature])
        else:
            feature_not_spam.append(train_input[each_feature_row][each_feature])
    mean_train_spam.append(np.mean(feature_spam))
    mean_train_not_spam.append(np.mean(feature_not_spam))
    std_dev_train_spam.append(np.std(mean_train_spam))
    std_dev_train_not_spam.append(np.std(mean_train_not_spam))

#keeping default value as 0.001 if any value is 0
std_dev_train_spam[std_dev_train_spam == 0] = default_standard_deviation
std_dev_train_not_spam[std_dev_train_not_spam == 0] = default_standard_deviation

#3rd Part - Gaussian Equation Implementation
def gauss_value(x,mean,std_deviation):
    step_1 = x - mean
    print "step_1" 
    print step_1
    step_2 = (step_1 / std_deviation)**2
    step_2 = (float)((-0.5*step_2))
    print "step_2"
    print step_2
    step_3 = np.exp(step_2)
    print "step_3"
    step_3 = (float)(round(step_3,2))

    if (step_3 == 0.00):
        step_3 = 0.01
    else:
        step_3 = round(step_3,2)  
    print step_3
    step_4 = (1/np.sqrt(2*np.pi*std_deviation)) 
    print "step_4"
    print step_4      
    step_5 = (float)(round((step_4*step_3),2))
    print "step_5"
    print step_5
    if (step_5 == 0):
        step_5 = 0.001
    else:
        step_5 = step_5    
    return step_5

class_1 = []
class_0 = []
class_x = 0
result = [] #to store test output predicted values

for each_row in range(len(test_input)):
    class_1 = np.log(probablity_train_spam)
    #class_0 = np.log(probablity_train_not_spam)
    for each_feature in range(train_input.shape[1]):
        x = test_input[each_row][each_feature]
        #half_sum = (((x - mean_train_spam[each_feature])**2)/(2*std_dev_train_spam[each_feature]**2))
        class_1 += np.log(gauss_value(x,mean_train_spam[each_feature],std_dev_train_spam[each_feature]))
        class_0 += np.log(gauss_value(x,mean_train_not_spam[each_feature],std_dev_train_not_spam[each_feature]))
    class_x = np.argmax([class_0, class_1])
    result.append(class_x)


#print(len(result)) #2301

#Compute a confusion matrix for the test set
print("\n ")
cfm = confusion_matrix(test_target, result)
print("\nConfusion Matrix: \n\n", cfm)
print("\n")

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
	else:     # (result[row] == 0 and test_target[row] == 1 ):
		FN += 1

accuracy = float(TP + TN)/(TP+TN+FP+FN)
#precision = float(TP)/(TP+FP)
#recall = float(TP)/(TP+FN)
print ("Accuracy : ", accuracy)
#print ("Precision: ", precision)
#print ("Recall   : ", recall)
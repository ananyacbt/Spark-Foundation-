#PREDICTION USING SUPERVISED ML - PREDICT PERCENTAGE OF A STUDENT BASED ON NO. OF STUDY HOURS

# Step 1 - Import Libraries

from matplotlib import style
import pandas as pd                   #data analysis
import numpy as np                    #manipulating the array objects
import matplotlib.pyplot as plt       #for plotting purpose

#data reading
url = "http://bit.ly/w-data"
student_data = pd.read_csv(url)  

#print("Successfully we have imported data")
#print(student_data)

#Statistical Overview of the data 
#to get numerical values like  count, mean value, standard daviation, min n max values, percentiles
#print(student_data.describe())
#to check null values
#print(student_data.isnull().sum())

# Step 2 - Plotted Graph to Visualize the dataset
#to check relation btn variables hours and scores

student_data.plot(x = "Hours", y = "Scores", style = 'o' , color ="blue")
plt.title("Hours Vs Percentages")
plt.xlabel("Hours Studied")
plt.ylabel("Percentage Score")
#plt.show()

# step 3 - Data Preperation
# prepare the data by extracting values from hour data into X variable and scores in Y variable

X = student_data.iloc[:, :-1].values      #spliting the data using iloc function
Y =student_data.iloc[:,1].values
#number of hours studied
#print(X)
#Scores obtained
#print(Y)
#split into test and train
from sklearn.model_selection import train_test_split      
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1,random_state=0)
#print(X_train,X_test,Y_train,Y_test)

# Step 4 - Training Algorithm
#design and train machine learning model

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
print("Training Complete")

#visualize the model - to get the best line fit for graph
#regresson line plottting
line = regressor.coef_*X+regressor.intercept_
#test data plotting
plt.scatter(X,Y)
plt.plot(X, line, color = "red")
#plt.show()

# Step 5 - Make prediction - test hours predicted randomly

#print(X_test)
Y_pred = regressor.predict(X_test)

#comparing actual vs predicted percentage
df = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})
df

#testing with a given data
Hours = 9.25
own_pred = regressor.predict([[Hours]])
print(f"Number of hours = {Hours}")
print (f"Predicted score of that student = {own_pred[0]}")

# Step 6 - Evaluation of the model
#To evaluate performance of the model by using mean sqaure error

from sklearn import metrics
print("Mean Absolute Error: ", metrics.mean_absolute_error(Y_test,Y_pred))



#importing pandas to get the data from the excel sheet
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import openpyxl
import seaborn as sb
import DataSplitandStandard 
import AIcwDataFormat

FullData = AIcwDataFormat.GetDataforMLP("Skelton")
########################
def RootMeanSquaredError(test,pred):
    RMSError = 0

    for i in range(len(pred)):
        prediction = pred[i]
        DataOutput = test[i]

        RMSError = RMSError + ((prediction - DataOutput)**2)

    RMSError = RMSError/len(pred)

    RMSError = np.sqrt(RMSError)

    return(RMSError)

def MeanSquaredRelativeError(test,pred):
    MSRError = 0

    for i in range(len(pred)):
        prediction = pred[i]
        DataOutput = test[i]
         
        MSRError = MSRError + (( ((prediction)-(DataOutput)) / DataOutput)**2)

    MSRError = MSRError/len(pred)

    return(MSRError)
    
def CoefficientOfEfficiency(test,pred):
    mean = 0
    n=len(test)

    sum1 = 0
    sum2 = 0

    for i in range(len(test)):
        DataOutput = test[i]
         
        mean+= DataOutput/n

    for i in range(len(pred)):
        prediction = pred[i]
        DataOutput = test[i]
        
        sum1 += ((prediction)-(DataOutput))**2
        sum2 += ((DataOutput)-(mean))**2

    CE = 1-(sum1/sum2)
    return CE
    
def CoefficientOfDetermination(test,pred):
    mean = 0
    modelledmean = 0
    n=len(test)

    sum1 = 0
    sum2 = 0
    sum3 = 0
    for i in range(len(pred)):
        prediction = pred[i]
        DataOutput = test[i]
        
        mean += DataOutput / n

        modelledmean += prediction / n

    for i in range(len(pred)):
        prediction = pred[i]
        DataOutput = test[i]
                
        sum1 += ((DataOutput)-(mean))*((prediction)-(modelledmean))
        sum2 += ((DataOutput)-(mean))**2
        sum3 += ((prediction)-(modelledmean))**2
        
    RSqr = ( (sum1) / ((sum2*sum3)**(0.5)) )**2
    return RSqr
#####BASELINE 1 - USING PREVIOUS VALUES TO PREDICT NEXT VALUE

baseline1 = DataSplitandStandard.PredictorsandPredictantsOnly(FullData,["Skelton"],"Skelton")
Baseline1Data = DataSplitandStandard.FormatData(baseline1)

#get data of skelton for that day and for the next day 
B1Predictions=[]
B1Actual=[]
for i in range(len(Baseline1Data)):
    B1Predictions.append(Baseline1Data[i][0])
    B1Actual.append(Baseline1Data[i][1])

#plot against each other 
x = np.linspace(0,250,250)
plt.scatter(B1Predictions,B1Actual,marker='.',color = 'r')
plt.plot(x,x,color='b',label='IDEAL PREDICTIONS')
plt.title("Baseline 1 Predictions vs Actual Readings")
plt.xlabel("Prediction")
plt.ylabel("Actual Readings")
plt.legend()
plt.show()

print("BASELINE1 PERFOM METRICS:")
print ("RMSE: " + str(RootMeanSquaredError(B1Actual,B1Predictions)))
print ("MSRE: " + str(MeanSquaredRelativeError(B1Actual,B1Predictions)))
print ("CE: " + str(CoefficientOfEfficiency(B1Actual,B1Predictions)))
print ("RSqr: " + str(CoefficientOfDetermination(B1Actual,B1Predictions)))
print("")

#####BASELINE 2 - LINEAR REGRESSION MODEL USING SKELTON FROM THE PREVIOUS DAY
baseline2 = DataSplitandStandard.PredictorsandPredictantsOnly(FullData,["Skelton"],"Skelton")

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X = baseline2.drop('predictant (Skelton)',axis=1)
y = baseline2['predictant (Skelton)']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model using predictions
y_test = y_test.reset_index(drop=True)

print("BASELINE2 PERFOM METRICS:")
print ("RMSE: " + str(RootMeanSquaredError(y_test, y_pred)))
print ("MSRE: " + str(MeanSquaredRelativeError(y_test, y_pred)))
print ("CE: " + str(CoefficientOfEfficiency(y_test, y_pred)))
print ("RSqr: " + str(CoefficientOfDetermination(y_test, y_pred)))

x = np.linspace(0,250,250)
plt.scatter(y_pred,y_test,marker='.',color = 'r')
plt.plot(x,x,color='b',label='IDEAL PREDICTIONS')
plt.title("Baseline 2 Predictions vs Actual Readings")
plt.xlabel("Prediction")
plt.ylabel("Actual Readings")
plt.legend()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import openpyxl
import seaborn as sb
import AIcwDataFormat

#get dataframe from the AIcwDataFormat.py file containing all datacolumns
FullData = AIcwDataFormat.GetDataforMLP("Skelton")

def PredictorsandPredictantsOnly(FullDataFrame,PredictorsNames,PredictantName):
    #creates a new dataframe 
    MLPInput = pd.DataFrame()

    #goes through the columns of the predictors you want to use 
    for column in PredictorsNames:
        #adds all data from the column to the new dataframe (keeps as a column)
        MLPInput[column] = FullDataFrame[column]

    #adds the predictant data as a column at the 'end' of the dataframe
    PredictantName = "predictant (" + PredictantName +")"
    #print(PredictantName)
    MLPInput[PredictantName] = FullDataFrame[PredictantName]

    MLPInput = MLPInput.dropna()

    #returns this new dataframe 
    return MLPInput

def RandomiseData(DataFrame):
    #randomises by shiffling the rows about and then resets the index (for easy lookups)
    DataFrame = DataFrame.sample(frac = 1).reset_index(drop = True)
    return DataFrame

def StandardiseData(DataFrame):
    #only need to de-standardise the predictant column so get min max values for the predictant column
    MinMaxValues = []

    MinMaxValues.append(DataFrame[DataFrame.columns[-1]].min())
    MinMaxValues.append(DataFrame[DataFrame.columns[-1]].max())

    #puts all data in the range of 0.1 - 0.9
    DataFrame = DataFrame.apply(lambda x: 0.1 + 0.8 * ((x - x.min()) / (x.max() - x.min())))
    return DataFrame,MinMaxValues

def FormatData(DataFrame):
    #needs to be in the form of an np.array([[input 1, input 2 , ... etc]]) and another np.array([[actual output]]
    Data = []

    #loops through each row of the dataframe
    for row in DataFrame.itertuples(index=False, name="Row"):

        currentrow = []
        inputs = np.zeros([len(row)-1])

        #as the way it dataframe has been formatted/setup above the predictant column is always the last one so 
        #can look at all the rows before the predictant for the input section of this list element 
        i=0
        for x in row[:(len(row))-1]:
            inputs[i] = x
            i+=1

        ActualOutput = row[len(row)-1]

        #appends both to a list to then store in the 'Data' list, so it is easy to access each columns
        #elements to train the MLP
        #the '.reshape(1,-1)' makes it able to be transposed which is needed in the MLP
        currentrow.append(inputs.reshape(1, -1))
        currentrow.append(ActualOutput)

        #adds this rows info to the 'Data' list
        Data.append(currentrow)

    return Data

def SplitData(Data):
    #suggested a 50-25-25 split of training, validation and test data
    #takes the list of data and splits accordingly

    #works out about how many elements are in 1/4th of the Data (double divide [//] as no remainder as cant have list element 1.4 (for example))
    quarter = len(Data)//4

    #makes the datasets using this information
    TrainingData = Data[:2*quarter]
    ValidationData = Data[2*quarter:3*quarter]
    TestData = Data[3*quarter:]

    #returns these datasets 
    return TrainingData,ValidationData,TestData

def FilterRandomStandardiseFormatAndSplitData(PredictantName,PredictorsNames):
    #applys all of the above functions in sequence 

    #gets the cleaned data
    FullData = AIcwDataFormat.GetDataforMLP(PredictantName)
    
    #selects only the predictors and predicants that want 
    FullData = PredictorsandPredictantsOnly(FullData,PredictorsNames,PredictantName)
        
    #randomise the data 
    FullData = RandomiseData(FullData)
        
    #standardise the data, retianing the min and max for the predictant so can destandardise later 
    FullData,MinMaxVal = StandardiseData(FullData)
    
    #format the data into list of lists, to be inputted used by my MLP
    FullData = FormatData(FullData)

    #split the data into the 3 datasets 
    TrainingData,ValidationData,TestData = SplitData(FullData)

    return TrainingData,ValidationData,TestData,MinMaxVal

def DeStandardiseData(Data,MinMax):

    #retreve the max and min values 
    max = MinMax[1]
    min = MinMax[0]

    DeStandData = []

    #destandardise each point using the following equation
    for point in Data:
        point = ((point-0.1)/0.8)*(max-min) + min
        DeStandData.append(point)

    return DeStandData

def DeStandardiseDataPoint(datapoint,MinMax):
        
    #retreve the max and min values 
    max = MinMax[1]
    min = MinMax[0]

    #destandardise the datapoint point using the following equation
    point = ((datapoint-0.1)/0.8)*(max-min) + min

    return point


def plotcolumn(dataframe,columnname):
    #generates array from 1 to amount of datapoints in that column of that dataframe  
    x = np.linspace(1,len(dataframe[columnname]),len(dataframe[columnname]))  
    #plots points
    plt.scatter(x, dataframe[columnname], color='b',marker='.')
    plt.title("Data gathered from " + columnname)
    plt.xlabel("1993-1996")
    plt.ylabel(columnname)
    plt.show()

def plotcorrleation(dataframe,columnname1,columnname2):
    plt.scatter(dataframe[columnname1], dataframe[columnname2], color='b',marker='.')
    plt.title("Correlation of Data gathered from " + columnname1 + " & " + columnname2 + " r=" + str(dataframe[columnname1].corr(dataframe[columnname2])))
    plt.xlabel(columnname2)
    plt.ylabel(columnname1)
    plt.show()

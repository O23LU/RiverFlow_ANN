#importing pandas to get the data from the excel sheet
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import openpyxl
import seaborn as sb
import AIcwDataFormat
import DataSplitandStandard 

def RelativeMeanSquaredErrorML(Data,weights,biases,MinMaxVal):
    MSRError = 0

    for row in Data:
        Inputs = row[0]
        DataOutput = row[1]

        nodeouputs,output = ForwardPassML(Inputs,weights,biases)
        
        #OutputOuput[0][0] just gets the Integer value that is contained in the 1x1 matrix that it returns 
        MSRError = MSRError + ((((DataSplitandStandard.DeStandardiseDataPoint(output[0][0],MinMaxVal)-DataSplitandStandard.DeStandardiseDataPoint(DataOutput,MinMaxVal))/DataSplitandStandard.DeStandardiseDataPoint(DataOutput,MinMaxVal))**2))

    MSRError = MSRError/len(Data)

    return(MSRError)

def PlotPredictionsML(TestData,datatype,weights,biases,MinMaxVal,nodelayout):
    Predictions = []
    ActualData = []
    for row in TestData:
        #the expected ouput
        ActualData.append(DataSplitandStandard.DeStandardiseDataPoint(row[1],MinMaxVal))

        #input
        Inputs = row[0]

        nodeouputs,output = ForwardPassML(Inputs,weights,biases)

        Predictions.append(DataSplitandStandard.DeStandardiseDataPoint(output[0][0],MinMaxVal))
    
    x = np.linspace(1,len(Predictions),len(Predictions))
    plt.scatter(x,ActualData, color='r',marker='.',label = "Actual Data")
    plt.scatter(x, Predictions, color='b',marker='.',label = "Predictions")
    plt.title("multi layer MLP predictions for the "+datatype)
    plt.legend()
    #plt.show()
    pdf.savefig()
    plt.close()

    line = np.linspace(1,250,250)
    plt.scatter(Predictions,ActualData, marker = '.', color = "r")
    plt.title("multi layer MLP predictions for the "+datatype)
    plt.xlabel("Prediction")
    plt.ylabel("Actual Data")
    plt.plot(line,line,label = "IDEAL RESULT", color = "b")
    plt.legend()
    #plt.show()
    pdf.savefig()
    plt.close()

def PlotErrorML(trainingerror,validationerror,errorscale,optimalindex):
    plt.plot(errorscale,trainingerror,color = "red",label = "training data error")
    plt.plot(errorscale,validationerror,color = "blue",label = "validation data error")
    plt.vlines(optimalindex,ymin = 0, ymax = 0.1,color='g',label = "Optimal solution (lowest validation error)")
    plt.xlabel("Epochs")
    plt.ylabel("Relative Mean Squared Error")
    plt.title("MutliLayer MLP")
    plt.yscale("log")
    plt.legend()
    #plt.show()
    pdf.savefig()
    plt.close()

def Sigmoid(x):
    ans = 1/(1+np.exp(-x))
    return ans

def SigmoidDeriv(x):
    return x*(1-x)

def Tanh(x):
    return np.tanh(x)

def TanhDeriv(x):
    #where the input x is actually tanh(x) as tanh'(x) = 1 - (tanh(x))^2
    return 1-(x**2) 

def InitaliseML(InDim,HidDim,OutDim):
    #creates arrays of weights and biases for the MLP
    #these are in the data type 'numpy.ndarray'

    #Hid dim is now an array of numbers representing the amount of hidden nodes in each respective layer 

    #going to have lists of weights where index[0] = input to hidden weight & index[1] = last hidden layer to output node weights
    weights = []

    #going to have lists of biases where index[0] = 1st hidden layer biases & index[1] = output node bias
    biases = []

    #wr = weight range (based on the amount of inputs)
    wr = 2/InDim

    #creating weights 
    weights.append(np.random.uniform(-wr,wr,(InDim,HidDim[0])))

    #loops and recusrivley adds to weights list
    for i in range(1,len(HidDim)):
        weights.append(np.random.uniform(-wr,wr,(HidDim[i-1],HidDim[i])))

    #adds to list the set of weights from the last hidden layer to the output node 
    weights.append(np.random.uniform(-wr,wr,(HidDim[-1],OutDim)))
    
    #creating biases 
    for i in range(0,len(HidDim)):
        biases.append(np.random.uniform(-wr,wr,(1,HidDim[i])))

    #adds to list the bias of output node 
    biases.append(np.random.uniform(-wr,wr,(1,OutDim)))

    return weights,biases
    
def ForwardPassML(Input,weights,biases):
    #input = [1xn]
    #weights = ([nxm],[mxz],...,[ixk],[k x out])
    #biases = ([1xm],[1xz],...,[1xk],[1 x out])

    nodeinputs = []
    nodeoutputs = []

    #adds inital input to node outputs as that is what the input nodes output (makes sense if you think about it) 
    #adds to node inputs so the indexing of the two lists matches up (and can refernace later on back pass)
    nodeoutputs.append(Input)
    nodeinputs.append(Input) 

    #loops through the weight and bias lists to essentially move from the left most layer [inputs] --> right most layer [output(s)]
    for i in range(0,len(weights)):
        # imagne starting from index[0]  = [1xn] * [nxm] + [1xm] --> [1xm]
        # then next iteration at index[1] = [1xm] * [mxz] + [1xz] --> [1xz]
        # keep looping
        # get to index[-1] = [1xk] * [k x out] + [1 x out] --> [1 x out]
        CurrentLayerNodeInputSum = (nodeoutputs[i] @ weights[i]) + biases[i]

        #adds these input values to the list - all the values in the list have matrix dimensions [1xm] where m is the number of nodes in that layer
        nodeinputs.append(CurrentLayerNodeInputSum)

        #the sigmoid funciton is applied element-wise, it doesnt change the dimension of the array 
        #therefore the output values also all have matrix dimensions [1xm] where m is the number of nodes in that layer
        nodeoutputs.append(Tanh(CurrentLayerNodeInputSum)) #- for sigmoid acivation function
    
    #get the output of the MLP by looking at the final element in the 'nodeouputs' list as that corresponds to the output of the nodes in the final layer, which is the output layer 
    output = nodeoutputs[-1]

    return nodeoutputs,output

def BackwardPassML(weights,biases,nodeoutputs,output,actualoutput,lr,beta,omega):
    #weights = ([nxm],[mxz],...,[ixk],[k x out])
    #biases = ([1xm],[1xz],...,[1xk],[1 x out])
    #node outputs = ([1xn],[1xm],[1xz],...,[1xk],[1 x out])
    #output = [1 x out]
    #actual output = [1 x out]
    
    nodedeltas = []

    #output deltas = [1 x out] - [1 x out] *(element wise) [1 x out] --> [1 x out]
    #outputdeltas = (actualoutput-output) * SigmoidDeriv(output)
    #for ReLU activation
    outputdeltas = (actualoutput-output + beta*omega) * TanhDeriv(output)

    nodedeltas.append(outputdeltas)

    #loops through all the layers outputs from the second to last layer to first hidden layer, to generate all the delta values for each layer
    #only go to i = 1 as nodeouputs[0] = inputs (from forward pass)
    for i in range(len(nodeoutputs)-2,0,-1):
        #imagne on first pass = [k x out] * [1 x out].T --> [kx1].T --> [1xk] *(element wise) [1 x k] --> [1 x k]
        #then loop to second pass = [i x k] * [1 x k].T --> [i x 1].T --> [1 x i] *(element wise) [1 x i] --> [1 x i]
        CurrentLayerDeltas = (weights[i] @ nodedeltas[0].T).T * TanhDeriv(nodeoutputs[i])

        #adds the delta values of the current layer to start of the list as we are working backwards but want to preserve the order that nodedeltas[0] correpsons to the delta values of the left most hidden layer 
        nodedeltas.insert(0,CurrentLayerDeltas)

    #so now nodedeltas has the form ([1 x m],[1 x z],...,[1 x k],[1 x out]) (same as node outputs)

    #now we have the delta values can start the backpass 
    for i in range(0,len(weights)):
        #imagine on first pass = [n x m] + ([1 x n].T * [1 x m] * const) --> [n x m] + [n x 1]*[1 x m]*const --> [n x m] + [n x m] --> [n x m]
        #then loop second pass = [m x z] + ([1 x m].T * [1 x z] * const) --> [m x z] + [m x 1]*[1 x z]*const --> [m x z] + [m x z] --> [m x z]
        weights[i] = weights[i] + ((nodeoutputs[i].T @ nodedeltas[i]) * lr)

        #imagne on first pass = [1 x m] + ([1 x m]*const) --> [1 x m] + [1 x m] --> [1 x m]
        #then loop second pass = [1 x z] + ([1 x z]*const) --> [1 x z] + [1 x z] --> [1 x z] 
        biases[i] = biases[i] + (nodedeltas[i] * lr)

    return weights,biases

def TrainMLPML(TrainingData,ValidationData,MinMaxVal,InDim,HidDim,OutDim,Epochs,lr):
    trainingerror=[]
    validationerror=[]
    errorscale=[]

    #takes the lr given as the max value for the lr 
    maxlr = lr

    #in form  --> (2,[2,3,2],1) for example
    weights,biases = InitaliseML(InDim,HidDim,OutDim)

    #at the beginning as beta should only range from around 0.1 to 0
    beta = 0
    omega = 0

    x = 0

    optimalindex=0
    optimalfound=0
    optimalsolution=[]

    #for annealing function to imporove results by meaking lr smaller faster 
    Epochs1=Epochs//1.95

    while x <= Epochs:

        #annealing function
        #lr = 0.01 + (maxlr - 0.01)*(1/(0.9828 + np.exp(-4+((14*x)/Epochs))))
        lr = 0.001 + (maxlr - 0.001)*(1/(0.9828 + np.exp(-4+((14*x)/Epochs1))))

        #updates beta and omega for weight decay 
        SumOfWeightsAndBiasesSquared = 0
        for i in range(0,len(weights)):
            SumOfWeightsAndBiasesSquared += (weights[i]**2).sum()
            SumOfWeightsAndBiasesSquared += (biases[i]**2).sum()
        beta = 1/((x+10000)*lr)

        #represents the total number of weights + biases
        n = InDim*HidDim[0] + HidDim[-1] + HidDim[-1]*OutDim + OutDim
        for i in range(0,len(HidDim)-1):
            n += HidDim[i]*HidDim[i+1] + HidDim[i]
        omega = (1/(2*(n))) * SumOfWeightsAndBiasesSquared

        #loops through training data set
        for row in TrainingData:
            #gets input and acutal value for each row
            Input = row[0]
            actualoutput = row[1]

            #computes predicted value via foward pass function
            nodeoutputs,output = ForwardPassML(Input,weights,biases)

            #updates weights and biases using the predicted and acutal values 
            weights,biases = BackwardPassML(weights,biases,nodeoutputs,output,actualoutput,lr,beta,omega)
        
        #increments loop
        x+=1

        #intervals of a specified amount of epochs - to capture error at these intervals
        if x%25 == 0 and x !=0:

            #generates error value at this epoch for validation and training datasets - for overfitting visualisation
            trainingerror.append(RelativeMeanSquaredErrorML(TrainingData,weights,biases,MinMaxVal))
            validationerror.append(RelativeMeanSquaredErrorML(ValidationData,weights,biases,MinMaxVal))
            errorscale.append(x)

            #for checking if the new point is the optimum (local)
            if x>30 and validationerror[-1]>validationerror[-2] and optimalfound == 0:
                optimalindex = x-25
                optimalfound = validationerror[-2]
                optimalsolution.append(weights)
                optimalsolution.append(biases)

            #if new point goes below the current optimal then is the new optimal
            if validationerror[-1]<optimalfound:
                optimalfound=0
                optimalsolution=[]

            print("Epoch: " + str(x))

    #if no optimal found to date then the most recent has the smallest error and is the optimum
    if optimalfound==0:
        optimalindex = x
        optimalsolution.append(weights)
        optimalsolution.append(biases)

    return trainingerror,validationerror,errorscale,weights,biases,optimalsolution,optimalindex

TrainingData,ValidationData,TestData,MinMaxVal = DataSplitandStandard.FilterRandomStandardiseFormatAndSplitData("Skelton",["average of flowrate","Skelton + Change in Average Rainfall","Skelton + Change in Average Rainfall T- 1 days","Skelton","Skelton T-1days","average of rainfall","average of rainfall T-1days","average of rainfall T-2days","average of flowrate T-1days"])

#BEST NODE STRUCURE LAYOUT
trainingerror,validationerror,errorscale,weights,biases,optimalsolution,optimalindex = TrainMLPML(TrainingData,ValidationData,MinMaxVal,9,[4,4,4],1,30000,0.05)


def RootMeanSquaredError(Data,weights,biases,MinMaxVal):
    RMSError = 0

    for row in Data:
        input = row[0]
        DataOutput = row[1]

        nodeoutputs,output = ForwardPassML(input,weights,biases)
        
        #OutputOuput[0][0] just gets the Integer value that is contained in the 1x1 matrix that it returns 
        RMSError = RMSError + ((((DataSplitandStandard.DeStandardiseDataPoint(output[0][0],MinMaxVal)-DataSplitandStandard.DeStandardiseDataPoint(DataOutput,MinMaxVal)))**2))

    RMSError = RMSError/len(Data)

    RMSError = np.sqrt(RMSError)

    return(RMSError)

def MeanSquaredRelativeError(Data,weights,biases,MinMaxVal):
    MSRError = 0

    for row in Data:
        input = row[0]
        DataOutput = row[1]

        nodeoutputs,output = ForwardPassML(input,weights,biases)
        
        #OutputOuput[0][0] just gets the Integer value that is contained in the 1x1 matrix that it returns 
        MSRError = MSRError + ((((DataSplitandStandard.DeStandardiseDataPoint(output[0][0],MinMaxVal)-DataSplitandStandard.DeStandardiseDataPoint(DataOutput,MinMaxVal))/DataSplitandStandard.DeStandardiseDataPoint(DataOutput,MinMaxVal))**2))

    MSRError = MSRError/len(Data)

    return(MSRError)
    
def CoefficientOfEfficiency(Data,weights,biases,MinMaxVal):
    mean = 0
    n=len(Data)

    sum1 = 0
    sum2 = 0
    for row in Data:
        mean+= row[1]/n

    for row in Data:
        input = row[0]
        DataOutput = row[1]

        nodeoutputs,output = ForwardPassML(input,weights,biases)
        
        sum1 += (DataSplitandStandard.DeStandardiseDataPoint(output[0][0],MinMaxVal)-DataSplitandStandard.DeStandardiseDataPoint(DataOutput,MinMaxVal))**2
        sum2 += (DataSplitandStandard.DeStandardiseDataPoint(DataOutput,MinMaxVal)-DataSplitandStandard.DeStandardiseDataPoint(mean,MinMaxVal))**2

    CE = 1-(sum1/sum2)
    return CE
    
def CoefficientOfDetermination(Data,weights,biases,MinMaxVal):
    mean = 0
    modelledmean = 0
    n=len(Data)

    sum1 = 0
    sum2 = 0
    sum3 = 0
    for row in Data:
        mean += row[1] / n
        input = row[0]

        nodeoutputs,output = ForwardPassML(input,weights,biases)
        modelledmean += output / n

    for row in Data:
        input = row[0]
        DataOutput = row[1]

        nodeoutputs,output = ForwardPassML(input,weights,biases)
        
        sum1 += (DataSplitandStandard.DeStandardiseDataPoint(DataOutput,MinMaxVal)-DataSplitandStandard.DeStandardiseDataPoint(mean,MinMaxVal))*(DataSplitandStandard.DeStandardiseDataPoint(output[0][0],MinMaxVal)-DataSplitandStandard.DeStandardiseDataPoint(modelledmean,MinMaxVal))
        sum2 += (DataSplitandStandard.DeStandardiseDataPoint(DataOutput,MinMaxVal)-DataSplitandStandard.DeStandardiseDataPoint(mean,MinMaxVal))**2
        sum3 += (DataSplitandStandard.DeStandardiseDataPoint(output[0][0],MinMaxVal)-DataSplitandStandard.DeStandardiseDataPoint(modelledmean,MinMaxVal))**2
        
    RSqr = ( (sum1) / ((sum2*sum3)**(0.5)) )**2
    return RSqr


print(weights)
print()
print(biases)

with PdfPages('MLPplots3.pdf') as pdf:
    
    PlotPredictionsML(TrainingData,"training data",weights,biases,MinMaxVal,".")
    PlotPredictionsML(TestData,"test data",weights,biases,MinMaxVal,".")
    PlotErrorML(trainingerror,validationerror,errorscale,optimalindex)
    PlotPredictionsML(TestData,"test data",optimalsolution[0],optimalsolution[1],MinMaxVal,".")
    print("RMSE = "+ str(RootMeanSquaredError(TestData,optimalsolution[0],optimalsolution[1],MinMaxVal)))
    print("MSRE = "+ str(MeanSquaredRelativeError(TestData,optimalsolution[0],optimalsolution[1],MinMaxVal)))
    print("CE = "+ str(CoefficientOfEfficiency(TestData,optimalsolution[0],optimalsolution[1],MinMaxVal)))
    print("RSqr = "+ str(CoefficientOfDetermination(TestData,optimalsolution[0],optimalsolution[1],MinMaxVal)))

    """
    RMSE = 13.06579660977282
    MSRE = 0.041141284522575206
    CE = 0.917188530048142
    RSqr = [[0.91754062]]
    after 15000 epochs (optimalsolution chosen at ~2400 epochs) correct  colour plot
    """

    pdf.close()


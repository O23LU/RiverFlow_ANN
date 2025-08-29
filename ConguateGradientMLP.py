import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import openpyxl
import seaborn as sb
import AIcwDataFormat
import DataSplitandStandard

def Tanh(x):
    return np.tanh(x)

def TanhDeriv(x):
    #where the input x is actually tanh(x) as tanh'(x) = 1 - (tanh(x))^2
    return 1-(x**2) 

def Sigmoid(x):
    ans = 1/(1+np.exp(-x))
    return ans

def SigmoidDeriv(x):
    return x*(1-x)

def Initalise(InDim,HidDim,OutDim):
    #creates arrays of weights and biases for the MLP
    #these are in the data type 'numpy.ndarray'

    #wr = weight range 
    wr1 = 2/(InDim+HidDim)
    wr2 = 2/(HidDim+OutDim)

    InToHidW = np.random.uniform(-wr1,wr1,(InDim*HidDim,1))
    HidToOutW = np.random.uniform(-wr2,wr2,(HidDim*OutDim,1))
    HidB = np.random.uniform(-wr1,wr1,(HidDim,1))
    OutB = np.random.uniform(-wr2,wr2,(OutDim,1))

    #turns all matricies into to a massive matrix with 1 column (is needed for the line search)
    TotalWB = np.vstack((InToHidW,HidB,HidToOutW,OutB))

    return(TotalWB)

def ForwardPass(Input,TotalWB,InDim,HidDim,OutDim):
    #input = [1xn]
    #input to hidden node weights = [nxm]
    #hidden bias = [1xm]
    #hidden to output weights = [mxu]
    #output bias = [1xu]
    """ to get this from the totalwb we need to reshape so can use forward pass from previous mlp (MLPv2.py) """
    InToHidW = TotalWB[0:InDim*HidDim].reshape(InDim,HidDim)
    HidB = TotalWB[InDim*HidDim:InDim*HidDim+HidDim].reshape(1,HidDim)
    HidToOutW = TotalWB[InDim*HidDim+HidDim:InDim*HidDim+HidDim+HidDim*OutDim].reshape(HidDim,OutDim)
    OutB = TotalWB[InDim*HidDim+HidDim+HidDim*OutDim:].reshape(1,OutDim)

    #input * in to hid weights + bias =>  [1xn]*[nxm] +[1xm] =>  [1xm]+[1xm] = [1xm]
    #hiddennodesinput = np.dot(Input,InToHidW) + HidB
    hiddennodesinput = (Input @ InToHidW) + HidB

    #apply activation function
    hiddennodesouput = Tanh(hiddennodesinput)

    #hidden nodes ouput * hidden to output weights = [1xm]*[mxu] = [1xu] + [1xu] ->bias = [1xu] output pre activation function
    outputnodeinput = (hiddennodesouput @ HidToOutW) + OutB

    #apply activation function
    output = Tanh(outputnodeinput)

    return HidToOutW,hiddennodesouput,output

def GetDecentDir(Inputs,n,TotalWB,PrevDecentDirection,InDim,HidDim,OutDim,firstpass):
    #takes whole batch as inputs so need running sum for them 
    DecentDirection = 0
    for row in Inputs:
        #takes input and actual ouptut from dataset 
        inputs = row[0]
        actualoutput = row[1]

        #calculates the output from the MLP using ForwardPass function
        HidToOutW,hiddennodesouput,output = ForwardPass(inputs,TotalWB,InDim,HidDim,OutDim)

        #uses the predicted output and actual ouput to calculate delta values (for finding direction of steepest gradient (decent direction))
        outputdeltas = (actualoutput-output) * (TanhDeriv(output))
        hiddendeltas = (HidToOutW @ outputdeltas.T).T * (TanhDeriv(hiddennodesouput))

        """THIS HAS BEEN TESTED ON THE EXAMPLE ON LEARN SLIDES"""
        #uses the delta values to get the decent direciton for the MLP (different deltas needed at different layers of the MLP)
        directionFirst = (inputs.T @ hiddendeltas).reshape(-1,1)
        directionSecond = (hiddendeltas).reshape(-1,1)
        directionThird = (hiddennodesouput.T @ outputdeltas).reshape(-1,1)
        directionFourth = (outputdeltas).reshape(-1,1)

        #form decent direction in same dimensions as TotalWB (for easy application/addtion to this matrix)
        DecentDirection += (np.vstack((directionFirst,directionSecond,directionThird,directionFourth))/n)
        
    #sets beta to zero incrementally to 'reset' search direction
    if firstpass == 0:
        beta = 0
    else:
        #using the fletcher reeves formula
        g = -1 * DecentDirection
        gprev = -1 * PrevDecentDirection

        #cacluate the beta value for conjugate gradient 
        beta = (((g.T)@(g))[0][0])/(((gprev.T)@(gprev))[0][0])

    #cacluates final decent direction using the beta value of the previous direction
    DecentDirection = DecentDirection + (beta * PrevDecentDirection)

    return DecentDirection

def MSE(DataSet,TotalWB,InDim,HidDim,OutDim,MinMaxVal):
    #Calculates MSE of the MLP on a given Dataset 
    MeanSquaredError = 0
    for row in DataSet:
        input = row[0]
        ActualOutput = row[1]

        HidToOutW,hiddennodesouput,output = ForwardPass(input,TotalWB,InDim,HidDim,OutDim)

        MeanSquaredError += (DataSplitandStandard.DeStandardiseDataPoint(ActualOutput,MinMaxVal)-DataSplitandStandard.DeStandardiseDataPoint(output[0][0],MinMaxVal))**2
        #MeanSquaredError += (ActualOutput-output[0][0])**2

    MeanSquaredError = MeanSquaredError/(len(DataSet))

    return MeanSquaredError

def FindBigInterval(dataset,DecentDirection,TotalWB,InDim,HidDim,OutDim,MinMaxVal):
    #set epsilon to a value (defined in slides)
    epsilon = 0.1/len(TrainingData)

    #defining the bounds of the interval
    n=1
    aprev = 0
    a = 0
    b = (epsilon)

    #uses the bounds to create new MLP to find the error at these values 
    aweights = TotalWB + (DecentDirection * a)
    bweights = TotalWB + (DecentDirection * b)
    
    n = 2

    #loops until the error on the right interval is greater than on the left, indicating minima lies inbetween
    while MSE(dataset,aweights,InDim,HidDim,OutDim,MinMaxVal) > MSE(dataset,bweights,InDim,HidDim,OutDim,MinMaxVal):
        #if minima not found move interval
        aprev = a
        a = b
        b = (2**(n-1)) * epsilon

        aweights = TotalWB + (DecentDirection * a)
        bweights = TotalWB + (DecentDirection * b)
        n+=1
    print("big int found")
    #define interval that minima lies between
    interval = [aprev,b]
    print(interval)
    """FOLLOWING JUST TO VISUALLISE"""
    
    errorval = []
    xaxis = []
    xtick = 0 
    """   
    for i in range(0,(int)(((b/epsilon)+5)/150)):
        print("looping ... ")
        currentmlpvals = TotalWB + (DecentDirection * xtick)

        xaxis.append(xtick)
        errorval.append(MSE(dataset,currentmlpvals,InDim,HidDim,OutDim,MinMaxVal))

        xtick+=epsilon
    

    plt.plot(xaxis, errorval, label='MSE', color='blue')
    plt.xlabel('alpha values')
    plt.ylabel('Mean Squared Error')
    plt.vlines(aprev,ymin = min(errorval)-2, ymax = max(errorval),color='r')
    plt.vlines(b,ymin = min(errorval)-2, ymax = max(errorval),color='r')
    """
    return interval,errorval
    
def ReduceInterval(interval,dataset,DecentDirection,TotalWB,InDim,HidDim,OutDim,MinMaxVal,errorval):
    #Golden Search Selection - interval reduction
    a = interval[0]
    b = interval[1]

    #const = golden ratio - 1
    r = 0.618

    #define C and D as in lecture slides (for golden search selection)
    c = a + (1 - r)*(b - a)
    cweights = TotalWB + (DecentDirection * c)

    d = b - (1 - r)*(b - a)
    dweights = TotalWB + (DecentDirection * d)

    #tolerance of how small to make the interval
    tol = 0.0001
    while (b - a) > tol:
        if MSE(dataset,dweights,InDim,HidDim,OutDim,MinMaxVal) > MSE(dataset,cweights,InDim,HidDim,OutDim,MinMaxVal):
            a = a
            b = d
            d = c
            c = a + (1 - r)*(b - a)
            cweights = TotalWB + (DecentDirection * c)
            dweights = TotalWB + (DecentDirection * d)
        else:
            a = c
            b = b
            c = d
            d = b - (1 - r)*(b - a)
            cweights = TotalWB + (DecentDirection * c)
            dweights = TotalWB + (DecentDirection * d)
    
    #now that we have a  very small interval that alpha(to get the optimal in the given seach direction)
    newinterval = [a,b]
    #can take alpha as the mean of the interval
    alpha = (a+b)/2

    """For visualisation Purposes
    plt.vlines(a,ymin = min(errorval)-2, ymax = max(errorval),color='g')
    plt.vlines(b,ymin = min(errorval)-2, ymax = max(errorval),color='g')
    plt.vlines(alpha,ymin = min(errorval)-2, ymax = max(errorval),color='y')
    plt.legend()
    plt.grid(False)
    pdf.savefig()
    plt.close()
    """
    print ("ALPHA: " + str(alpha))
    print("\n")

    return alpha

def TrainMLPwithConjugateGradient(TrainingDataset,ValidationDataset,InDim,HidDim,OutDim,MinMaxVal):
    #putting all togeather results in this function

    #initalise variables
    TotalWB = Initalise(InDim,HidDim,OutDim)
    n = len(TotalWB)
    m = (len(TrainingDataset))

    #for recording error
    modeltrainingerror = []
    modelvalidationerror = []
    errorxaxis = np.linspace(0,150,150)

    #to initalise the variable
    PrevDecentDirection = 0

    x = 0
    #to define a stop -> can change this to be a tolerance of seeing a similar MSE multiple times in a row
    while x < 150:
        #used to know when to 'reset' search direction
        y = x%n

        #get Decent Direction
        DecentDirection = GetDecentDir(TrainingDataset,m,TotalWB,PrevDecentDirection,InDim,HidDim,OutDim,y)

        #find big interval for the given search direction
        interval,errorval = FindBigInterval(TrainingDataset,DecentDirection,TotalWB,InDim,HidDim,OutDim,MinMaxVal)
        
        #get alpha value that reduces MSE to its Minima in the given search direction
        alpha = ReduceInterval(interval,TrainingDataset,DecentDirection,TotalWB,InDim,HidDim,OutDim,MinMaxVal,errorval)
        
        #record this Previous Decentdirection for use when calcualteing new decent direction using beta (conjugate gradients aspect of the line search)
        PrevDecentDirection = DecentDirection.copy()
        
        #update weights with this optimal alpha value
        TotalWB = TotalWB + (DecentDirection * alpha)

        #record error for this new model (for visualisation only)
        modeltrainingerror.append(MSE(TrainingDataset,TotalWB,InDim,HidDim,OutDim,MinMaxVal))
        modelvalidationerror.append(MSE(ValidationDataset,TotalWB,InDim,HidDim,OutDim,MinMaxVal))

        #loop
        x+=1

        #to ensure progress
        print("Epoch: " + str(x) + ",    MSE:" + str(modeltrainingerror[-1]))

    return TotalWB,modeltrainingerror,modelvalidationerror,errorxaxis

def PlotPredictions(TestData,datatype,TotalWB,InDim,HidDim,OutDim,MinMaxVal):
    Predictions = []
    ActualData = []
    for row in TestData:
        #the expected ouput
        ActualData.append(DataSplitandStandard.DeStandardiseDataPoint(row[1],MinMaxVal))

        #input
        Input = row[0]

        HidToOutW,hiddennodesouput,output = ForwardPass(Input,TotalWB,InDim,HidDim,OutDim)

        Predictions.append(DataSplitandStandard.DeStandardiseDataPoint(output[0][0],MinMaxVal))
    
    x = np.linspace(1,len(Predictions),len(Predictions))
    plt.scatter(x,ActualData, color='r',marker='.',label = "Actual Data")
    plt.scatter(x, Predictions, color='b',marker='.',label = "Predictions")
    plt.title("Single Layer MLP Predictions for the "+datatype)
    plt.legend()
    #plt.show()
    pdf.savefig()
    plt.close()

    line = np.linspace(1,250,250) #to draw an y=x line where 0<=x<=300 
    plt.scatter(Predictions,ActualData,color='r',marker='.')
    plt.plot(line,line,color = "b",label = "ideal data line")
    plt.xlabel("predicted data")
    plt.ylabel("actual data")
    plt.legend()
    #plt.plot()
    pdf.savefig()
    plt.close()

def PlotModelError(modeltrainingerror,modelvalidationerror,errorxaxis):
    plt.plot(errorxaxis,modeltrainingerror, color='r',label = "TrainignData MSE")
    plt.plot(errorxaxis, modelvalidationerror, color='b',label = "ValidationData MSE")
    plt.title("Single Layer MLP (conjugate gradient) MSE over time")
    plt.legend()
    #plt.show()
    pdf.savefig()
    plt.close()

def RootMeanSquaredError(Data,TotalWB,InDim,HidDim,OutDim,MinMaxVal):
    RMSError = 0

    for row in Data:
        input = row[0]
        DataOutput = row[1]

        HidToOutW,hiddennodesouput,output = ForwardPass(input,TotalWB,InDim,HidDim,OutDim)
        
        #OutputOuput[0][0] just gets the Integer value that is contained in the 1x1 matrix that it returns 
        RMSError = RMSError + ((((DataSplitandStandard.DeStandardiseDataPoint(output[0][0],MinMaxVal)-DataSplitandStandard.DeStandardiseDataPoint(DataOutput,MinMaxVal)))**2))

    RMSError = RMSError/len(Data)

    RMSError = RMSError**(0.5)

    return(RMSError)

def MeanSquaredRelativeError(Data,TotalWB,InDim,HidDim,OutDim,MinMaxVal):
    MSRError = 0

    for row in Data:
        input = row[0]
        DataOutput = row[1]

        HidToOutW,hiddennodesouput,output = ForwardPass(input,TotalWB,InDim,HidDim,OutDim)
        
        #OutputOuput[0][0] just gets the Integer value that is contained in the 1x1 matrix that it returns 
        MSRError = MSRError + ((((DataSplitandStandard.DeStandardiseDataPoint(output[0][0],MinMaxVal)-DataSplitandStandard.DeStandardiseDataPoint(DataOutput,MinMaxVal))/DataSplitandStandard.DeStandardiseDataPoint(DataOutput,MinMaxVal))**2))

    MSRError = MSRError/len(Data)

    return(MSRError)
    
def CoefficientOfEfficiency(Data,TotalWB,InDim,HidDim,OutDim,MinMaxVal):
    mean = 0
    n=len(Data)

    sum1 = 0
    sum2 = 0
    for row in Data:
        mean+= row[1]/n

    for row in Data:
        input = row[0]
        DataOutput = row[1]

        HidToOutW,hiddennodesouput,output = ForwardPass(input,TotalWB,InDim,HidDim,OutDim)
        
        sum1 += (DataSplitandStandard.DeStandardiseDataPoint(output[0][0],MinMaxVal)-DataSplitandStandard.DeStandardiseDataPoint(DataOutput,MinMaxVal))**2
        sum2 += (DataSplitandStandard.DeStandardiseDataPoint(DataOutput,MinMaxVal)-DataSplitandStandard.DeStandardiseDataPoint(mean,MinMaxVal))**2

    CE = 1-(sum1/sum2)
    return CE
    
def CoefficientOfDetermination(Data,TotalWB,InDim,HidDim,OutDim,MinMaxVal):
    mean = 0
    modelledmean = 0
    n=len(Data)

    sum1 = 0
    sum2 = 0
    sum3 = 0
    for row in Data:
        mean += row[1] / n
        input = row[0]

        HidToOutW,hiddennodesouput,output = ForwardPass(input,TotalWB,InDim,HidDim,OutDim)
        modelledmean += output / n

    for row in Data:
        input = row[0]
        DataOutput = row[1]

        HidToOutW,hiddennodesouput,output = ForwardPass(input,TotalWB,InDim,HidDim,OutDim)
        
        sum1 += (DataSplitandStandard.DeStandardiseDataPoint(DataOutput,MinMaxVal)-DataSplitandStandard.DeStandardiseDataPoint(mean,MinMaxVal))*(DataSplitandStandard.DeStandardiseDataPoint(output[0][0],MinMaxVal)-DataSplitandStandard.DeStandardiseDataPoint(modelledmean,MinMaxVal))
        sum2 += (DataSplitandStandard.DeStandardiseDataPoint(DataOutput,MinMaxVal)-DataSplitandStandard.DeStandardiseDataPoint(mean,MinMaxVal))**2
        sum3 += (DataSplitandStandard.DeStandardiseDataPoint(output[0][0],MinMaxVal)-DataSplitandStandard.DeStandardiseDataPoint(modelledmean,MinMaxVal))**2
        
    RSqr = ( (sum1) / ((sum2*sum3)**(0.5)) )**2
    return RSqr

def TrainMLPwithConjugateGradientMINIBATCH(TrainingDataset,ValidationDataset,InDim,HidDim,OutDim,MinMaxVal):
    #very similar but changes to small batch insead of full dataset batch 
    batchsize = 1
    size = len(TrainingDataset)//batchsize

    TotalWB = Initalise(InDim,HidDim,OutDim)
    n = len(TotalWB)

    modeltrainingerror = []
    modelvalidationerror = []
    errorxaxis = np.linspace(0,300,300)

    #to initalise the variable
    PrevDecentDirection = 0

    x = 0
    while x < 300:
        #loops as many mini-batches that are contained in the dataset 
        for i in range(0,batchsize):
            #make dataset this batch only 
            TDS = TrainingDataset[size*i:size*(i+1)]

            #used to know when to 'reset' search direction
            y = x%n

            #get Decent Direction
            DecentDirection = GetDecentDir(TrainingDataset,TotalWB,PrevDecentDirection,InDim,HidDim,OutDim,y)

            #find big interval for the given search direction
            interval,errorval = FindBigInterval(TrainingDataset,DecentDirection,TotalWB,InDim,HidDim,OutDim,MinMaxVal)
            
            #get alpha value that reduces MSE to its Minima in the given search direction
            alpha = ReduceInterval(interval,TrainingDataset,DecentDirection,TotalWB,InDim,HidDim,OutDim,MinMaxVal,errorval)
            
            #record this Previous Decentdirection for use when calcualteing new decent direction using beta (conjugate gradients aspect of the line search)
            PrevDecentDirection = DecentDirection.copy()
            
            #update weights with this optimal alpha value
            TotalWB = TotalWB + (DecentDirection * alpha)

        #record error for this new model (for visualisation only)
        modeltrainingerror.append(MSE(TrainingDataset,TotalWB,InDim,HidDim,OutDim,MinMaxVal))
        modelvalidationerror.append(MSE(ValidationDataset,TotalWB,InDim,HidDim,OutDim,MinMaxVal))

        #loop
        x+=1

        #to see progress
        print("Epoch: " + str(x))

    return TotalWB,modeltrainingerror,modelvalidationerror,errorxaxis

TrainingData,ValidationData,TestData,MinMaxVal = DataSplitandStandard.FilterRandomStandardiseFormatAndSplitData("Skelton",["average of flowrate","Skelton + Change in Average Rainfall","Skelton + Change in Average Rainfall T- 1 days","Skelton","Skelton T-1days","average of rainfall","average of rainfall T-1days","average of rainfall T-2days","average of flowrate T-1days"]) 

with PdfPages("ConjugateGradients.pdf") as pdf:
    
    TotalWB,modeltrainingerror,modelvalidationerror,errorxaxis = TrainMLPwithConjugateGradient(TrainingData,ValidationData,9,12,1,MinMaxVal)
    #TotalWB,modeltrainingerror,modelvalidationerror,errorxaxis = TrainMLPwithConjugateGradientMINIBATCH(TrainingData,ValidationData,9,12,1,MinMaxVal)
    
    PlotPredictions(TrainingData,"Training DataSet",TotalWB,9,12,1,MinMaxVal)
    PlotPredictions(TestData,"Test DataSet",TotalWB,9,12,1,MinMaxVal)
    PlotModelError(modeltrainingerror,modelvalidationerror,errorxaxis)
    
    print("RMSE = "+ str(RootMeanSquaredError(TestData,TotalWB,9,12,1,MinMaxVal)))
    print("MSRE = "+ str(MeanSquaredRelativeError(TestData,TotalWB,9,12,1,MinMaxVal)))
    print("CE = "+ str(CoefficientOfEfficiency(TestData,TotalWB,9,12,1,MinMaxVal)))
    print("RSqr = "+ str(CoefficientOfDetermination(TestData,TotalWB,9,12,1,MinMaxVal)))

    """
    NEED TO LOOK INTO THESE VALUES COMPARED TO THE PLOTS

    RMSE = 11.98827281566086
    MSRE = 0.08187754637636985
    CE = 0.9302824029887738
    RSqr = 0.93061814
    for 12 hidden nodes after 100 epochs
    """

    pdf.close()



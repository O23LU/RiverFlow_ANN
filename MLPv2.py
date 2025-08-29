#importing pandas to get the data from the excel sheet
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import openpyxl
import seaborn as sb
import AIcwDataFormat
import DataSplitandStandard

def PlotPredictions(TestData,datatype,InToHidW,HidToOutW,HidB,OutB,MinMaxVal):
    Predictions = []
    ActualData = []
    for row in TestData:
        #the expected ouput
        ActualData.append(DataSplitandStandard.DeStandardiseDataPoint(row[1],MinMaxVal))

        #input
        Input = row[0]

        HiddenOutputs,OutputOutput = ForwardPass(Input,InToHidW,HidToOutW,HidB,OutB)

        Predictions.append(DataSplitandStandard.DeStandardiseDataPoint(OutputOutput[0][0],MinMaxVal))
    
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

def PlotError(PlotXaxis,Trainingerror,Validationerror,name,optimalindex):
    plt.plot(PlotXaxis, Trainingerror, label='Training Data', color='red')
    plt.plot(PlotXaxis, Validationerror, label='Validation Data', color='blue')
    plt.vlines(optimalindex,ymin = 0, ymax = 0.1,color='g',label = "Optimal solution (lowest validation error)")
    plt.title("SINGLE LAYER MLP with " + str(name))
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Relative Error')
    plt.yscale("log")
    plt.legend()
    plt.grid(False)
    pdf.savefig()
    plt.close()

def MeanSquaredRelativeError(Data,InToHidW,HidToOutW,HidB,OutB,MinMaxVal):
    MSRError = 0

    for row in Data:
        input = row[0]
        DataOutput = row[1]

        HiddenOutputs,OutputOutput = ForwardPass(input,InToHidW,HidToOutW,HidB,OutB)
        
        #OutputOuput[0][0] just gets the Integer value that is contained in the 1x1 matrix that it returns 
        MSRError = MSRError + ((((DataSplitandStandard.DeStandardiseDataPoint(OutputOutput[0][0],MinMaxVal)-DataSplitandStandard.DeStandardiseDataPoint(DataOutput,MinMaxVal))/DataSplitandStandard.DeStandardiseDataPoint(DataOutput,MinMaxVal))**2))

    MSRError = MSRError/len(Data)

    return(MSRError)

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

def Initalise(InDim,HidDim,OutDim):
    #creates arrays of weights and biases for the MLP
    #these are in the data type 'numpy.ndarray'

    #wr = weight range 
    wr1 = 2/(InDim+HidDim)
    wr2 = 2/(HidDim+OutDim)

    #initalises in the range given by the dimensions of the MLP
    InToHidW = np.random.uniform(-wr1,wr1,(InDim,HidDim))
    HidToOutW = np.random.uniform(-wr2,wr2,(HidDim,OutDim))
    HidB = np.random.uniform(-wr1,wr1,(1,HidDim))
    OutB = np.random.uniform(-wr2,wr2,(1,OutDim))
    
    InToHidMomentumW = np.random.uniform(0,0,(InDim,HidDim))
    HidToOutMomentumW = np.random.uniform(0,0,(HidDim,OutDim))
    HidBMomentum =  np.random.uniform(0,0,(1,HidDim))
    OutBMomentum = np.random.uniform(0,0,(1,OutDim))

    return(InToHidW,HidToOutW,HidB,OutB,InToHidMomentumW,HidToOutMomentumW,HidBMomentum,OutBMomentum)

def ForwardPass(Input,InToHidW,HidToOutW,HidB,OutB):
    #input = [1xn]
    #input to hidden node weights = [nxm]
    #hidden bias = [1xm]
    #hidden to output weights = [mxu]
    #output bias = [1xu]

    #input * in to hid weights + bias =>  [1xn]*[nxm] +[1xm] =>  [1xm]+[1xm] = [1xm]
    hiddennodesinput = (Input @ InToHidW) + HidB

    #apply activation function
    hiddennodesouput = Tanh(hiddennodesinput)

    #hidden nodes ouput * hidden to output weights = [1xm]*[mxu] = [1xu] + [1xu] ->bias = [1xu] output pre activation function
    outputnodeinput = (hiddennodesouput @ HidToOutW) + OutB
    
    #apply activation function
    output = Tanh(outputnodeinput)

    return hiddennodesouput,output

def BackPass(Input,InToHidW,HidToOutW,HidB,OutB,hiddennodesouput,output,actualoutput,InToHidMomentumW,HidToOutMomentumW,HidBMomentum,OutBMomentum,lr,mc,beta,omega):
    #lr = learning rate (usually around 0.1)
    #mc = momentum constnat (usually around 0.9)

    #input = [1xn]
    #input to hidden node weights = [nxm]
    #hidden bias = [1xm]
    #hidden to output weights = [mxu]
    #output bias = [1xu]

    #input to hidden node momentum weights = [nxm]
    #hidden to ouput node momentum weights 

    #actualoutput = [1xu]
    #output = [1xu]

    #hiddennodeinput = [1xm]
    #hiddennodeouput = [1xm]

    #for sigmoid function
    #actualoutput - output * sigmoidderiv(ouput)[element wise] = [1xu]-[1xu] = [1xu] * [1xu](element wise) = [1xu]
    outputdeltas = (actualoutput-output + beta*omega) * (TanhDeriv(output))

    #hidden to ouput weights * ouput deltas = [mxu]*([1xu].T) = [mx1].T = [1xm] (element wise mult with hidden nodes ouput) [1xm] x [1xm] = [1xm]
    hiddendeltas = (HidToOutW @ outputdeltas.T).T * (TanhDeriv(hiddennodesouput))

    #now all deltas are calculated with dimensions of [1xu] & [1xm] repsectivley

    #[nxm] + ([1xn].T)*[1xm] * const = [nxm] + [nx1]*[1xm] * const = [nxm]+[nxm]*const = [nxm]
    NewInToHidW = InToHidW + (Input.T @ hiddendeltas) * lr + (InToHidMomentumW * mc)

    #[1xm] + [1xm] * const = [1xm]
    NewHidB = HidB + (hiddendeltas * lr) + (HidBMomentum * mc) 

    #[mxu] + ([1xm].T)*[1xu])*const = [mxu] + [mx1]*[1xu]*const = [mxu] + [mxu]*const = [mxu]
    NewHidToOutW = HidToOutW + (hiddennodesouput.T @ outputdeltas) * lr + (HidToOutMomentumW * mc) 

    #[1xu] + [1xu] * const = [1xu]
    NewOutB = OutB +  (outputdeltas * lr) + (OutBMomentum * mc)

    #set the InToHidMomentumW and HidToOutMomentumW to the differenece bewteen newW and the W for the next pass
    InToHidMomentumW = NewInToHidW - InToHidW
    HidToOutMomentumW = NewHidToOutW - HidToOutW
    HidBMomentum = NewHidB - HidB
    OutBMomentum = NewOutB - OutB

    return NewInToHidW,NewHidB,NewHidToOutW,NewOutB,InToHidMomentumW,HidToOutMomentumW,HidBMomentum,OutBMomentum

def trainMLP(TrainingData,ValidationData,MinMaxVal,InDim,HidDim,OutDim,lr,momentumconstant,Epochs):
    InToHidW,HidToOutW,HidB,OutB,InToHidMomentumW,HidToOutMomentumW,HidBMomentum,OutBMomentum = Initalise(InDim,HidDim,OutDim)
    
    #momentum constant defined as a list of lenght 2 that contains [min momentum constant,max momentum constant]
    mcmin = momentumconstant[0]
    mcmax = momentumconstant[1]

    #for the annealing function defined below
    maxlr = lr

    #for plotting error data
    TErrorPlotData1 = []
    VErrorPlotData1 = []
    PlotXaxis = []

    #at the beginning as beta should only range from around 0.1 to 0
    beta = 0
    omega = 0

    optimalindex=0
    optimalfound=0
    optimalsolution=[]
    x = 1
    while x <= Epochs:

        #for 'annealing' going to a min of 0.01 from a specified max - plotted on desmos and pic in repot part of project folder
        lr = 0.005 + (maxlr - 0.005)*(1/(0.9828 + np.exp(-3.5+((10.5*x)/Epochs))))# - drops of slightly quicker than abouve to have more time at smaller lr

        #annealing for the momentum constant - needs testing 
        momentumconstant = mcmin + (mcmax - mcmax) * (1 - (x)/(Epochs*0.55))

        #for weight decay -> updates beta and omega values
        SumOfWeightsAndBiasesSquared = (((InToHidW**2).sum())+((HidB**2).sum())+((HidToOutW**2).sum())+((OutB**2).sum()))
        beta = 1/((x+10000)*lr)
        omega = (1/(2*(InDim*HidDim + HidDim + HidDim*OutDim + OutDim))) * SumOfWeightsAndBiasesSquared

        for row in TrainingData:
            #takes the data for that row of the dataset
            Inputs = row[0]
            ActualOutput = row[1]
            
            #computes the output via forward pass
            HiddenOutputs,OutputOutput = ForwardPass(Inputs,InToHidW,HidToOutW,HidB,OutB)

            #updates weight using the actual output and the computed value from above 
            InToHidW,HidB,HidToOutW,OutB,InToHidMomentumW,HidToOutMomentumW,HidBMomentum,OutBMomentum = BackPass(Inputs,InToHidW,HidToOutW,HidB,OutB,HiddenOutputs,OutputOutput,ActualOutput,InToHidMomentumW,HidToOutMomentumW,HidBMomentum,OutBMomentum,lr,momentumconstant,beta,omega)
        
        #intervals of a specified amount of epochs - to capture error at these intervals
        if x%25 == 0:
            
            #generates error value at this epoch for validation and training datasets - for overfitting visualisation
            TErrorPlotData1.append(MeanSquaredRelativeError(TrainingData,InToHidW,HidToOutW,HidB,OutB,MinMaxVal))
            VErrorPlotData1.append(MeanSquaredRelativeError(ValidationData,InToHidW,HidToOutW,HidB,OutB,MinMaxVal))
            PlotXaxis.append(x)

            print(str(x)+" epochs")
            print("current learning rate: " + str(lr))
            print()

            #for checking if the new point is the optimum (local)
            if x>30 and VErrorPlotData1[-1]>VErrorPlotData1[-2] and optimalfound == 0:
                optimalindex = x-25
                optimalfound = VErrorPlotData1[-2]
                optimalsolution.append(InToHidW)
                optimalsolution.append(HidToOutW)
                optimalsolution.append(HidB)
                optimalsolution.append(OutB)

            #if new point goes below the current optimal then is the new optimal
            if VErrorPlotData1[-1]<optimalfound:
                optimalfound=0
                optimalsolution=[]

        x+=1

    #if no optimal found to date then the most recent has the smallest error and is the optimum
    if optimalfound==0:
        optimalindex = x
        optimalsolution.append(InToHidW)
        optimalsolution.append(HidToOutW)
        optimalsolution.append(HidB)
        optimalsolution.append(OutB)
        

    return InToHidW,HidToOutW,HidB,OutB,OutputOutput,TErrorPlotData1,VErrorPlotData1,PlotXaxis,optimalindex,optimalsolution

######################################################################
def BatchBackPass(Input,HidToOutW,hiddennodesouput,output,actualoutput,InToHidSum,HidToOutSum,HidBSum,OutBSum,beta,omega,batchsize):
    #lr = learning rate (usually around 0.1)
    #mc = momentum constnat (usually around 0.9)

    #input = [1xn]
    #input to hidden node weights = [nxm]
    #hidden bias = [1xm]
    #hidden to output weights = [mxu]
    #output bias = [1xu]

    #input to hidden node momentum weights = [nxm]
    #hidden to ouput node momentum weights 

    #actualoutput = [1xu]
    #output = [1xu]

    #hiddennodeinput = [1xm]
    #hiddennodeouput = [1xm]

    #for sigmoid function
    #actualoutput - output * sigmoidderiv(ouput)[element wise] = [1xu]-[1xu] = [1xu] * [1xu](element wise) = [1xu]
    outputdeltas = (actualoutput-output + beta*omega) * (TanhDeriv(output))
    #outputdeltas = (actualoutput-output + beta*omega) * (SigmoidDeriv(output))

    #hidden to ouput weights * ouput deltas = [mxu]*([1xu].T) = [mx1].T = [1xm] (element wise mult with hidden nodes ouput) [1xm] x [1xm] = [1xm]
    hiddendeltas = (HidToOutW @ outputdeltas.T).T * (TanhDeriv(hiddennodesouput))

    #print(SigmoidDeriv(hiddennodesouput))

    #now all deltas are calculated with dimensions of [1xu] & [1xm] repsectivley

    #adds to the sum in proportion to the batch size, to generate an average weight change to apply at the end of the batch 
    InToHidSum = InToHidSum + (Input.T @ hiddendeltas) / batchsize

    HidBSum = HidBSum + (hiddendeltas) / batchsize

    HidToOutSum = HidToOutSum + (hiddennodesouput.T @ outputdeltas) / batchsize

    OutBSum = OutBSum + (outputdeltas) / batchsize

    return InToHidSum,HidBSum,HidToOutSum,OutBSum

def ApplyBatchBackpass(InToHidSum,HidBSum,HidToOutSum,OutBSum,InToHidW,HidToOutW,HidB,OutB,InToHidMomentumW,HidToOutMomentumW,HidBMomentum,OutBMomentum,lr,mc):
    
    #take the sum which represents the avearage weight change needed to be applies to each weight (generated over the course of the batch)
    NewInToHidW = InToHidW + (InToHidSum)*lr + (InToHidMomentumW * mc)
    NewHidB = HidB + (HidBSum)*lr + (HidBMomentum * mc)
    NewHidToOutW = HidToOutW + (HidToOutSum)*lr + (HidToOutMomentumW * mc)
    NewOutB = OutB + (OutBSum)*lr + (OutBMomentum * mc)

    #set the InToHidMomentumW and HidToOutMomentumW to the differenece bewteen newW and the W for the next pass
    InToHidMomentumW = NewInToHidW - InToHidW
    HidToOutMomentumW = NewHidToOutW - HidToOutW
    HidBMomentum = NewHidB - HidB
    OutBMomentum = NewOutB - OutB

    return NewInToHidW,HidB,NewHidToOutW,OutB,InToHidMomentumW,HidToOutMomentumW,HidBMomentum,OutBMomentum

def BatchtrainMLP(TrainingData,ValidationData,MinMaxVal,InDim,HidDim,OutDim,lr,mc,batchsize,Epochs):
    InToHidW,HidToOutW,HidB,OutB,InToHidMomentumW,HidToOutMomentumW,HidBMomentum,OutBMomentum = Initalise(InDim,HidDim,OutDim)
    
    #momentum constant defined as a list of lenght 2 that contains [min momentum constant,max momentum constant]
    mcmin = mc[0]
    mcmax = mc[1]

    #for the annealing function defined below
    maxlr = lr

    #for plotting error data
    TErrorPlotData1 = []
    VErrorPlotData1 = []
    PlotXaxis = []

    #at the beginning as beta should only range from around 0.1 to 0
    beta = 0
    omega = 0

    optimalindex=0
    optimalfound=0
    optimalsolution=[]
    x = 1
    while x <= Epochs:

        InToHidSum = np.random.uniform(0,0,(InDim,HidDim))
        HidToOutSum = np.random.uniform(0,0,(HidDim,OutDim))
        HidBSum = np.random.uniform(0,0,(1,HidDim))
        OutBSum = np.random.uniform(0,0,(1,OutDim))

        batchprogress=0

        #'annealing' of the bath size
        batchsize = max(batchsize,1)

        #for 'annealing' going to a min of 0.01 from a specified max - plotted on desmos and pic in repot part of project folder
        lr = 0.001 + (maxlr - 0.001)*(1/(0.9828 + np.exp(-4+((8.5*x)/Epochs))))# - drops of slightly quicker than abouve to have more time at smaller lr

        #annealing for the momentum constant - needs testing 
        mc = mcmin + (mcmax - mcmax) * (1 - (x)/(Epochs*0.75))

        #for weight decay:
        SumOfWeightsAndBiasesSquared = (((InToHidW**2).sum())+((HidB**2).sum())+((HidToOutW**2).sum())+((OutB**2).sum()))
        beta = 1/((x+5000)*lr)
        omega = (1/(2*(InDim*HidDim + HidDim + HidDim*OutDim + OutDim))) * SumOfWeightsAndBiasesSquared

        for row in TrainingData:

            Inputs = row[0]
            ActualOutput = row[1]
            
            HiddenOutputs,OutputOutput = ForwardPass(Inputs,InToHidW,HidToOutW,HidB,OutB)

            InToHidSum,HidBSum,HidToOutSum,OutBSum = BatchBackPass(Inputs,HidToOutW,HiddenOutputs,OutputOutput,ActualOutput,InToHidSum,HidToOutSum,HidBSum,OutBSum,beta,omega,batchsize)
            
            batchprogress+=1

            if batchprogress == batchsize:
                batchprogress = 0
                
                InToHidW,HidB,HidToOutW,OutB,InToHidMomentumW,HidToOutMomentumW,HidBMomentum,OutBMomentum = ApplyBatchBackpass(InToHidSum,HidBSum,HidToOutSum,OutBSum,InToHidW,HidToOutW,HidB,OutB,InToHidMomentumW,HidToOutMomentumW,HidBMomentum,OutBMomentum,lr,mc)
                
                InToHidSum = np.random.uniform(0,0,(InDim,HidDim))
                HidToOutSum = np.random.uniform(0,0,(HidDim,OutDim))
                HidBSum = np.random.uniform(0,0,(1,HidDim))
                OutBSum = np.random.uniform(0,0,(1,OutDim))

                #max so it doesnt go to 0 as thats stupid - cant test on nothing
                batchsize = max(batchsize//1.8,1)

        #intervals of a specified amount of epochs - to capture error at these intervals
        if x%25 == 0:
            
            TErrorPlotData1.append(MeanSquaredRelativeError(TrainingData,InToHidW,HidToOutW,HidB,OutB,MinMaxVal))
            VErrorPlotData1.append(MeanSquaredRelativeError(ValidationData,InToHidW,HidToOutW,HidB,OutB,MinMaxVal))
            PlotXaxis.append(x)

            print(str(x)+" epochs")
            print("current learning rate: " + str(lr))
            print()

            if x>30 and VErrorPlotData1[-1]>VErrorPlotData1[-2] and optimalfound == 0:
                optimalindex = x-25
                optimalfound = VErrorPlotData1[-2]
                optimalsolution.append(InToHidW)
                optimalsolution.append(HidToOutW)
                optimalsolution.append(HidB)
                optimalsolution.append(OutB)

            
            if VErrorPlotData1[-1]<optimalfound:
                optimalfound=0
                optimalsolution=[]

        x+=1

    if optimalfound==0:
        optimalindex = x
        optimalsolution.append(InToHidW)
        optimalsolution.append(HidToOutW)
        optimalsolution.append(HidB)
        optimalsolution.append(OutB)


    if optimalfound==0:
        optimalindex = x
        optimalsolution.append(InToHidW)
        optimalsolution.append(HidToOutW)
        optimalsolution.append(HidB)
        optimalsolution.append(OutB)
        
        x+=1

    return InToHidW,HidToOutW,HidB,OutB,OutputOutput,TErrorPlotData1,VErrorPlotData1,PlotXaxis,optimalindex,optimalsolution

######################################################################
def RootMeanSquaredError(Data,InToHidW,HidToOutW,HidB,OutB,MinMaxVal):
    RMSError = 0

    for row in Data:
        input = row[0]
        DataOutput = row[1]

        HiddenOutputs,OutputOutput = ForwardPass(input,InToHidW,HidToOutW,HidB,OutB)
        
        #OutputOuput[0][0] just gets the Integer value that is contained in the 1x1 matrix that it returns 
        RMSError = RMSError + ((((DataSplitandStandard.DeStandardiseDataPoint(OutputOutput[0][0],MinMaxVal)-DataSplitandStandard.DeStandardiseDataPoint(DataOutput,MinMaxVal)))**2))

    RMSError = RMSError/len(Data)

    RMSError = RMSError**(0.5)

    return(RMSError)

def MeanSquaredRelativeError(Data,InToHidW,HidToOutW,HidB,OutB,MinMaxVal):
    MSRError = 0

    for row in Data:
        input = row[0]
        DataOutput = row[1]

        HiddenOutputs,OutputOutput = ForwardPass(input,InToHidW,HidToOutW,HidB,OutB)
        
        #OutputOuput[0][0] just gets the Integer value that is contained in the 1x1 matrix that it returns 
        MSRError = MSRError + ((((DataSplitandStandard.DeStandardiseDataPoint(OutputOutput[0][0],MinMaxVal)-DataSplitandStandard.DeStandardiseDataPoint(DataOutput,MinMaxVal))/DataSplitandStandard.DeStandardiseDataPoint(DataOutput,MinMaxVal))**2))

    MSRError = MSRError/len(Data)

    return(MSRError)
    
def CoefficientOfEfficiency(Data,InToHidW,HidToOutW,HidB,OutB,MinMaxVal):
    mean = 0
    n=len(Data)

    sum1 = 0
    sum2 = 0
    for row in Data:
        mean+= row[1]/n

    for row in Data:
        input = row[0]
        DataOutput = row[1]

        HiddenOutputs,OutputOutput = ForwardPass(input,InToHidW,HidToOutW,HidB,OutB)
        
        sum1 += (DataSplitandStandard.DeStandardiseDataPoint(OutputOutput[0][0],MinMaxVal)-DataSplitandStandard.DeStandardiseDataPoint(DataOutput,MinMaxVal))**2
        sum2 += (DataSplitandStandard.DeStandardiseDataPoint(DataOutput,MinMaxVal)-DataSplitandStandard.DeStandardiseDataPoint(mean,MinMaxVal))**2

    CE = 1-(sum1/sum2)
    return CE
    
def CoefficientOfDetermination(Data,InToHidW,HidToOutW,HidB,OutB,MinMaxVal):
    mean = 0
    modelledmean = 0
    n=len(Data)

    sum1 = 0
    sum2 = 0
    sum3 = 0
    for row in Data:
        mean+= row[1]/n
        input = row[0]

        HiddenOutputs,OutputOutput = ForwardPass(input,InToHidW,HidToOutW,HidB,OutB)
        modelledmean += OutputOutput/n

    for row in Data:
        input = row[0]
        DataOutput = row[1]

        HiddenOutputs,OutputOutput = ForwardPass(input,InToHidW,HidToOutW,HidB,OutB)
        
        sum1 += (DataSplitandStandard.DeStandardiseDataPoint(DataOutput,MinMaxVal)-DataSplitandStandard.DeStandardiseDataPoint(mean,MinMaxVal))*(DataSplitandStandard.DeStandardiseDataPoint(OutputOutput[0][0],MinMaxVal)-DataSplitandStandard.DeStandardiseDataPoint(modelledmean,MinMaxVal))
        sum2 += (DataSplitandStandard.DeStandardiseDataPoint(DataOutput,MinMaxVal)-DataSplitandStandard.DeStandardiseDataPoint(mean,MinMaxVal))**2
        sum3 += (DataSplitandStandard.DeStandardiseDataPoint(OutputOutput[0][0],MinMaxVal)-DataSplitandStandard.DeStandardiseDataPoint(modelledmean,MinMaxVal))**2
        
    RSqr = ( (sum1) / ((sum2*sum3)**(0.5)) )**2
    return RSqr


TrainingData,ValidationData,TestData,MinMaxVal = DataSplitandStandard.FilterRandomStandardiseFormatAndSplitData("Skelton",["average of flowrate","Skelton + Change in Average Rainfall","Skelton + Change in Average Rainfall T- 1 days","Skelton","Skelton T-1days","average of rainfall","average of rainfall T-1days","average of rainfall T-2days","average of flowrate T-1days"]) 

#GOODish RESULTS SO FAR
#,2,7,1,0.1,[0.1,0.9],25000) with ["average of flowrate","testingsomething3"]

#GREAT RESULT TO FAR
#9,12,1,0.1,[0.1,0.9],15000) with ["average of flowrate","Skelton + Change in Average Rainfall","Skelton T-1 days + Change in Average Rainfall T- 1 days","Skelton","Skelton T-1days","average of rainfall","average of rainfall T-1days","average of rainfall T-2days","average of flowrate T-1days"]
#6,10,1,0.1,[0.1,0.9],15000) with ["average of flowrate","Skelton + Change in Average Rainfall","Skelton days + Change in Average Rainfall T- 1 days","Skelton","average of rainfall","Skelton T-1days"]

"""TESTING BATCH LEARNING --> DOESN'T LIKE SIGMOID FUNCTION, TO DO WITH VANISHING GRADIENTS?"""
InToHidW1,HidToOutW1,HidB1,OutB1,OutputOutput,TErrorPlotData1,VErrorPlotData1,PlotXaxis1,optimalindex,optimalsolution = BatchtrainMLP(TrainingData,ValidationData,MinMaxVal,9,12,1,0.05,[0.1,0.9],len(TrainingData),2500)
#InToHidW1,HidToOutW1,HidB1,OutB1,OutputOutput,TErrorPlotData1,VErrorPlotData1,PlotXaxis1,optimalindex,optimalsolution = trainMLP(TrainingData,ValidationData,MinMaxVal,9,12,1,0.1,[0.1,0.9],25000)

with PdfPages("MLPplots2.pdf") as pdf:
    PlotPredictions(TestData,"Test Data",InToHidW1,HidToOutW1,HidB1,OutB1,MinMaxVal)
    PlotError(PlotXaxis1,TErrorPlotData1,VErrorPlotData1,"12 Hidden Nodes",optimalindex)
    PlotPredictions(TestData,"Test Data - optimal solution",optimalsolution[0],optimalsolution[1],optimalsolution[2],optimalsolution[3],MinMaxVal)
    print("RMSE = "+ str(RootMeanSquaredError(TestData,optimalsolution[0],optimalsolution[1],optimalsolution[2],optimalsolution[3],MinMaxVal)))
    print("MSRE = "+ str(MeanSquaredRelativeError(TestData,optimalsolution[0],optimalsolution[1],optimalsolution[2],optimalsolution[3],MinMaxVal)))
    print("CE = "+ str(CoefficientOfEfficiency(TestData,optimalsolution[0],optimalsolution[1],optimalsolution[2],optimalsolution[3],MinMaxVal)))
    print("RSqr = "+ str(CoefficientOfDetermination(TestData,optimalsolution[0],optimalsolution[1],optimalsolution[2],optimalsolution[3],MinMaxVal)))
    """
    RMSE = 12.688807863490386
    MSRE = 0.0643878566361261
    CE = 0.9300416493185523
    RSqr = [[0.93115407]]
    after 10k epochs
    """

    pdf.close() 





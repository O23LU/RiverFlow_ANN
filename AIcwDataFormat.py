#importing pandas to get the data from the excel sheet
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import openpyxl
import seaborn as sb


"""need to randomise the data """

"""
dataframe[columnname].where(condidtion,np.nan) repalces all values that dont meet the condition with nan
"""

# Read the first sheet of an Excel file
#df = pd.read_excel('nuralnetworkdataset.xlsx', usecols = [0,1,2,3,4,5,6,7,8],skiprows = 1)
dataset1 = pd.read_excel('nuralnetworkdataset.xlsx', usecols = [1,2,3,4,5,6,7,8],skiprows = 1)
dataset2 = pd.read_excel('nuralnetworkdataset.xlsx', usecols = [1,2,3,4,5,6,7,8],skiprows = 1)
dataset3 = pd.read_excel('nuralnetworkdataset.xlsx', usecols = [1,2,3,4,5,6,7,8],skiprows = 1)
forreport = pd.read_excel('nuralnetworkdataset.xlsx', usecols = [1,2,3,4,5,6,7,8],skiprows = 1)
forreport1 = pd.read_excel('nuralnetworkdataset.xlsx', usecols = [1,2,3,4,5,6,7,8],skiprows = 1)
forreport2 = pd.read_excel('nuralnetworkdataset.xlsx', usecols = [1,2,3,4,5,6,7,8],skiprows = 1)
forreport3 = pd.read_excel('nuralnetworkdataset.xlsx', usecols = [1,2,3,4,5,6,7,8],skiprows = 1)
forreport4 = pd.read_excel('nuralnetworkdataset.xlsx', usecols = [1,2,3,4,5,6,7,8],skiprows = 1)


def invaliddata(dataframe):
    for column in dataframe.columns:
        #converts non-numeric values to nan values
        dataframe[column] = pd.to_numeric(dataframe[column],errors = 'coerce')

        #find invalid (negative values)
        dataframe[column] = dataframe[column].where(dataframe[column] >= 0, np.nan)

        #interpolating using build it methods (linear,cubic,etc)
        #to fill in the nan values 
        dataframe[column] = dataframe[column].interpolate(method ='linear', limit_direction = 'both')

def extremedata(dataframe,filtercoeff):
    for column in dataframe.columns:
        #calculate the mean and standard deviation of column to remove extreme values
        std = dataframe[column].std()
        mean = dataframe[column].mean()
        
        #create a min and max value of data using this mean & std
        minval = mean - filtercoeff*std
        maxval = mean + filtercoeff*std
        #replace all of these 'extreme'/out of reasonable range values to nan
        #so they can easily be interpolated using the built in pandas function
        dataframe[column] = dataframe[column].where((dataframe[column] <= maxval) & (dataframe[column] >= minval), np.nan)

        #interpolating using build it methods (linear,cubic,etc)
        #to fill in the nan values generated from the filtering method
        dataframe[column] = dataframe[column].interpolate(method = 'linear', limit_direction = 'both')

def Std_RollingWindowCleaning(dataframe, WindowSize, FilterCoeff):
    for column in dataframe.columns:
        #these are column values containing all the rolling window results 
        mean = dataframe[column].rolling(window=WindowSize, min_periods = 1, center = True).mean()
        std = dataframe[column].rolling(window=WindowSize, min_periods = 1, center = True).std(ddof = 1)

        #work out min and max value bounds for the window
        #these are column values containing all the rolling window results 
        minval = mean - FilterCoeff*std
        maxval = mean + FilterCoeff*std

        #replace all of these 'extreme'/out of reasonable range values to nan
        #so they can easily be interpolated using the built in pandas function
        dataframe[column] = dataframe[column].where((dataframe[column] <= maxval) & (dataframe[column] >= minval), np.nan)

        #interpolating using build it methods (linear,cubic,etc)
        #to fill in the nan values generated from the filtering method
        dataframe[column] = dataframe[column].interpolate(method = 'linear', limit_direction = 'both')

def MeanAbsoluteDeviation_RollingWindowCleaning(dataframe, WindowSize, FilterCoeff):
    for column in dataframe.columns:
        #these are column values containing all the rolling window results 
        mean = dataframe[column].rolling(window=WindowSize,min_periods = 1, center = True).mean()

        #median absolute deviation (like std but for median)
        #indicated the average distance between observations and their mean
        mad = dataframe[column].rolling(window=WindowSize,min_periods = 1, center = True).apply(lambda x: np.mean(np.abs(x-np.mean(x))))

        #work out min and max value bounds for the window
        #these are column values containing all the rolling window results 
        minval = mean - FilterCoeff*mad
        maxval = mean + FilterCoeff*mad

        #replace all of these 'extreme'/out of reasonable range values to nan
        #so they can easily be interpolated using the built in pandas function
        dataframe[column] = dataframe[column].where((dataframe[column] <= maxval) & (dataframe[column] >= minval), np.nan)

        #interpolating using build it methods (linear,cubic,etc)
        #to fill in the nan values generated from the filtering method
        dataframe[column] = dataframe[column].interpolate(method = 'linear', limit_direction = 'both')

def MedianAbsoluteDeviation_RollingWindowCleaning(dataframe, WindowSize, FilterCoeff):
    for column in dataframe.columns:
        #these are column values containing all the rolling window results   ,min_periods = 0
        median = dataframe[column].rolling(window=WindowSize, min_periods = 1, center = True).median()

        #median absolute deviation (like std but for median)
        #indicated the average distance between observations and their mean
        mad = dataframe[column].rolling(window=WindowSize, min_periods = 1, center = True).apply(lambda x: np.median(np.abs(x-np.median(x))))

        #work out min and max value bounds for the window
        #these are column values containing all the rolling window results 
        minval = median - FilterCoeff*mad
        maxval = median + FilterCoeff*mad

        #replace all of these 'extreme'/out of reasonable range values to nan
        #so they can easily be interpolated using the built in pandas function
        dataframe[column] = dataframe[column].where((dataframe[column] <= maxval) & (dataframe[column] >= minval), np.nan)
        #interpolating using build it methods (linear,cubic,etc)
        #to fill in the nan values generated from the filtering method
        dataframe[column] = dataframe[column].interpolate(method = 'linear')

def InterQuartileRange_RollingWindowCleaning(dataframe, WindowSize):
    for column in dataframe.columns:
        #gets the first and third quartile of the data from the rolling window
        Q1 = dataframe[column].rolling(window=WindowSize, min_periods = 1, center = True).quantile(0.25)
        Q3 = dataframe[column].rolling(window=WindowSize, min_periods = 1, center = True).quantile(0.75)

        #calculates the interquartilerange for this centeral value using 
        #data from the rolling window
        IQR = Q3-Q1

        #work out min and max value bounds for the window
        #these are column values containing all the rolling window results 
        #uses +- 1.5* IQR as this coveres 99.3% of the data and is the 
        #widley accepted value when useing IQR to identify outliers 
        minval = Q1 - 1.5*IQR
        maxval = Q3 + 1.5*IQR

        #replace all of these 'extreme'/out of reasonable range values to nan
        #so they can easily be interpolated using the built in pandas function
        dataframe[column] = dataframe[column].where((dataframe[column] <= maxval) & (dataframe[column] >= minval), np.nan)

        #interpolating using build it methods (linear,cubic,etc)
        #to fill in the nan values generated from the filtering method
        dataframe[column] = dataframe[column].interpolate(method = 'linear')

def setpredictant(dataframe,predictantname):
    #makes a list of the new order of the columns (just to have predictant at start)
    newcolumnorder=[]

    predictantcolname = "predictant ("+predictantname+")"

    dataframe[predictantcolname] = dataframe[predictantname]
    #shifts the predicntat by -1 to align so that it represents the next days reading (to comapare my MLP's predictions to)
    dataframe[predictantcolname] = dataframe[predictantcolname].shift(-1)

    for name in dataframe.columns:
        newcolumnorder.append(name)

    #gets the index for the instertion of new columnnames in the correct place 
    predictantnameindex = newcolumnorder.index(predictantcolname)
    newcolumnorder.insert(0,newcolumnorder.pop(predictantnameindex))
    return dataframe[newcolumnorder]

def addlag(dataframe,columntoaddlag):
    #makes the new order of the columns to the t-days are next to the reading for that day, easier to view on the heat map
    newcolumnorder=[]
    for name in dataframe.columns:
        newcolumnorder.append(name)

    #gets the index for the instertion of new columnnames in the correct place 
    laggedcolumnindex = newcolumnorder.index(columntoaddlag)+1

    #loops through to add lag up to 5 days, can go more but correlation only decreases 
    for i in range(1,3):
        laggedname = columntoaddlag + " T-" + str(i) + "days"
        #.shift lags data by a given amount
        dataframe[laggedname] = dataframe[columntoaddlag].shift(i)
        newcolumnorder.insert(laggedcolumnindex,laggedname)
        laggedcolumnindex+=1

    return dataframe[newcolumnorder]

def averagecolumns(dataframe,columnstoaverage,name):
    dataframe["average of " + name] = dataframe[columnstoaverage].mean(axis = 1)
    return dataframe

def plotcolumn(dataframe,columnname):
    #generates array from 1 to amount of datapoints in that column of that dataframe  
    x = np.linspace(1,len(dataframe[columnname]),len(dataframe[columnname]))  
    #plots points
    plt.scatter(x, dataframe[columnname], color='b',marker='.')
    plt.title("Data gathered from " + columnname)
    plt.xlabel("1993-1996")
    #plt.ylabel(columnname)
    plt.ylabel("Daily Mean Flow Rate for " + str(columnname))
    pdf.savefig()
    plt.close()
    #plt.show()

def plotcorrleation(dataframe,columnname1,columnname2):
    plt.scatter(dataframe[columnname1], dataframe[columnname2], color='b',marker='.')
    plt.title("Correlation of Data gathered from " + columnname1 + " & " + columnname2 + " r=" + str(dataframe[columnname1].corr(dataframe[columnname2])))
    plt.xlabel(columnname2)
    plt.ylabel(columnname1)
    pdf.savefig()
    plt.close()
    #plt.show()

def plotcorrelheatmap(dataframe):
    plt.figure(figsize=(18,9.5))  # Optional: Adjust the size of the heatmap
    sb.heatmap(dataframe.corr(), annot=True,  linewidths=0.5)#cmap='seismic',
    plt.title('Correlation Heatmap')
    plt.xticks(rotation=45,fontsize=10,ha = 'right')
    plt.tight_layout()
    pdf.savefig()
    plt.close()
    #plt.show()

def GetDataforMLP(PredictantName):
    Dataframe = pd.read_excel('nuralnetworkdataset.xlsx', usecols = [1,2,3,4,5,6,7,8],skiprows = 1)
    invaliddata(Dataframe)
    extremedata(Dataframe,3.5)

    #now for the choice of what rolling window cleaning method
    InterQuartileRange_RollingWindowCleaning(Dataframe,30)

    Dataframe = setpredictant(Dataframe,PredictantName)

    rainfall = ["Arkengarthdale","East Cowton","Malham Tarn","Snaizeholme"]
    flowrate = ["Crakehill","Skip Bridge","Westwick"]

    #creates all the lagged and average columns I want to use 
    Dataframe = averagecolumns(Dataframe,rainfall,"rainfall")
    Dataframe = averagecolumns(Dataframe,flowrate,"flowrate")
    Dataframe = addlag(Dataframe,"average of rainfall") # --> column name is "average of rainfall T-1days"
    Dataframe = addlag(Dataframe,"average of flowrate")
    Dataframe = addlag(Dataframe,"Skelton")
    Dataframe["Change in Average Rainfall"] = Dataframe["average of rainfall"] - Dataframe["average of rainfall T-1days"]
    Dataframe["Skelton + Change in Average Rainfall"] = Dataframe[["Skelton","Change in Average Rainfall"]].sum(axis=1)
    Dataframe["Change in Average Rainfall T-1days"] = Dataframe["average of rainfall T-1days"] - Dataframe["average of rainfall T-2days"]
    Dataframe["Skelton T-1 days + Change in Average Rainfall T- 1 days"] = Dataframe[["Skelton T-1days","Change in Average Rainfall T-1days"]].sum(axis=1)
    Dataframe["Skelton + Change in Average Rainfall T- 1 days"] = Dataframe[["Skelton","Change in Average Rainfall T-1days"]].sum(axis=1)

    #remove Nan values
    Dataframe.dropna()
    
    return Dataframe



def reportplotinvalidcolumn(original_dataframe, columnname):
    dataframe = original_dataframe.copy()  # Make a copy to avoid modifying original data
    
    original_values = dataframe[columnname].copy()  # Store original values before filtering
    
    # Identify negative values
    negative_mask = dataframe[columnname] < 0
    dataframe[columnname] = dataframe[columnname].where(~negative_mask, np.nan)  # Replace negative values with NaN
    
    dataframe[columnname] = dataframe[columnname].interpolate(method='linear')  # Interpolate missing values
    
    x = np.arange(len(dataframe[columnname]))  # X-axis values
    y = dataframe[columnname]  # Final interpolated data
    
    # Identify interpolated values (values that were originally NaN due to negatives)
    interpolated_mask = original_values.isna() | negative_mask
    
    plt.scatter(x[~interpolated_mask & ~negative_mask], y[~interpolated_mask & ~negative_mask], color='b', marker='.', label='Original Data')  # Original data in blue
    plt.scatter(x[interpolated_mask], y[interpolated_mask], color='r', marker='.', label='Interpolated Data')  # Interpolated data in red
    plt.scatter(x[negative_mask], original_values[negative_mask], color='g', marker='.', label='Invalid Data')  # Negative values in green
    
    plt.title("Invalid Filtered Data for " + columnname)
    plt.xlabel("1993 -> 1996")
    plt.ylabel("Daily Mean Flow Rate for " + str(columnname))
    plt.legend()
    pdf.savefig()
    #plt.show()
    plt.close()

def reportplotextremecolumn(original_dataframe, columnname, filtercoeff):
    dataframe = original_dataframe.copy()  # Make a copy to avoid modifying original data
    
    original_values = dataframe[columnname].copy()  # Store original values before filtering
    
    # Calculate mean and standard deviation
    mean = dataframe[columnname].mean()
    std = dataframe[columnname].std()
    
    # Define min and max thresholds
    minval = mean - filtercoeff * std
    maxval = mean + filtercoeff * std
    
    # Identify extreme values (outside the range)
    extreme_mask = (dataframe[columnname] < minval) | (dataframe[columnname] > maxval)
    dataframe[columnname] = dataframe[columnname].where(~extreme_mask, np.nan)  # Replace extreme values with NaN
    
    dataframe[columnname] = dataframe[columnname].interpolate(method='linear', limit_direction='both')  # Interpolate missing values
    
    x = np.arange(len(dataframe[columnname]))  # X-axis values
    y = dataframe[columnname]  # Final interpolated data
    
    # Identify interpolated values (values that were originally NaN or extreme)
    interpolated_mask = original_values.isna() | extreme_mask
    
    plt.scatter(x[~interpolated_mask & ~extreme_mask], y[~interpolated_mask & ~extreme_mask], color='b', marker='.', label='Original Data')  # Original data in blue
    plt.scatter(x[interpolated_mask], y[interpolated_mask], color='r',marker = '.', label='Interpolated Data')  # Interpolated data in red
    plt.scatter(x[extreme_mask], original_values[extreme_mask], color='g',marker = '.', label='Extreme Data')  # Extreme values in green
    
    plt.title("Extreme Value Filtered Data for " + columnname)
    plt.xlabel("1993 -> 1996")
    plt.ylabel("Daily Mean Flow Rate for " + str(columnname))
    plt.legend()
    pdf.savefig()
    plt.close()

def reportployIQRcolumn(original_dataframe, columnname, WindowSize):
    dataframe = original_dataframe.copy()  # Make a copy to avoid modifying original data
    
    original_values = dataframe[columnname].copy()  # Store original values before filtering
    
    # Compute rolling IQR-based filtering
    Q1 = dataframe[columnname].rolling(window=WindowSize, min_periods=1, center=True).quantile(0.25)
    Q3 = dataframe[columnname].rolling(window=WindowSize, min_periods=1, center=True).quantile(0.75)
    IQR = Q3 - Q1
    
    # Define min and max thresholds
    minval = Q1 - 1.5 * IQR
    maxval = Q3 + 1.5 * IQR
    
    # Identify extreme values (outliers)
    extreme_mask = (dataframe[columnname] < minval) | (dataframe[columnname] > maxval)
    dataframe[columnname] = dataframe[columnname].where(~extreme_mask, np.nan)  # Replace outliers with NaN
    
    dataframe[columnname] = dataframe[columnname].interpolate(method='linear')  # Interpolate missing values
    
    x = np.arange(len(dataframe[columnname]))  # X-axis values
    y = dataframe[columnname]  # Final interpolated data
    
    # Identify interpolated values (values that were originally NaN or extreme)
    interpolated_mask = original_values.isna() | extreme_mask
    
    plt.scatter(x[~interpolated_mask & ~extreme_mask], y[~interpolated_mask & ~extreme_mask], color='b', marker='.', label='Original Data')  # Original data in blue
    plt.scatter(x[interpolated_mask], y[interpolated_mask], color='r',marker = '.', label='Interpolated Data')  # Interpolated data in red
    plt.scatter(x[extreme_mask], original_values[extreme_mask], color='g',marker = '.', label='Extreme Data')  # Outliers in green
    
    plt.title("IQR Rolling Window Filtered Data for " + columnname)
    plt.xlabel("1993 -> 1996")
    plt.ylabel("Daily Mean Flow Rate for " + str(columnname))
    plt.legend()
    pdf.savefig()
    plt.close()



with PdfPages('plots.pdf') as pdf:

    #FOR COMPARING MODELS
    #in order of RMSE,MSRE,CE,RSqr 
    #Model1=[13.873514686079172,0.08074751967408132,0.9127633718826161,0.9147048] #single layer with all improvements
    Model1=[12.688807863490386,0.0643878566361261,0.9300416493185523,0.93115407]# -second image
    Model2=[12.518969910580065,0.03898638041785662,0.9192373114339087,0.92065227] #multilayer
    Model3=[11.28827281566086,0.08067754637636985,0.9312824029887738,0.93261814] #single layer with conjugate gradients 

    Baseline1=[19.04802043,0.05526891,0.84115758,0.8474717] #using yesterdays value
    Baseline2=[18.36885288987064,0.09054526930900181,0.8536185582747414,0.8540777355853788] #linear interpolation using yesterdays value

    #########-change this with my data-########
    barWidth = 0.15
    #fig = plt.subplots(figsize =(12, 8)) 

    br1 = np.arange(1) 
    br2 = [x + barWidth for x in br1] 
    br3 = [x + barWidth for x in br2] 

    plt.bar(br1, Model1[0], color ='r', width = barWidth, edgecolor ='black', label ='Model 1') 
    plt.bar(br2, Model2[0], color ='g', width = barWidth, edgecolor ='black', label ='Model 2') 
    plt.bar(br3, Model3[0], color ='b', width = barWidth, edgecolor ='black', label ='Model 3') 
    plt.xlabel('Model Metrics') 
    plt.ylabel('Value') 
    plt.xticks([r + barWidth for r in range(1)], 
            ['RMSE'])#, 'MSRE', 'CE', 'RSqr'])

    plt.legend()
    pdf.savefig()
    plt.close()
    #plt.show()

    br1 = np.arange(1) 
    br2 = [x + barWidth for x in br1] 
    br3 = [x + barWidth for x in br2] 

    plt.bar(br1, Model1[1], color ='r', width = barWidth, edgecolor ='black', label ='Model 1') 
    plt.bar(br2, Model2[1], color ='g', width = barWidth, edgecolor ='black', label ='Model 2') 
    plt.bar(br3, Model3[1], color ='b', width = barWidth, edgecolor ='black', label ='Model 3') 
    plt.xlabel('Model Metrics') 
    plt.ylabel('Value') 
    plt.xticks([r + barWidth for r in range(1)], 
            #['RMSE', 
            ['MSRE'])#, 'CE', 'RSqr'])

    plt.legend()
    pdf.savefig()
    plt.close()
    #plt.show()

    br1 = np.arange(2) 
    br2 = [x + barWidth for x in br1] 
    br3 = [x + barWidth for x in br2] 

    plt.bar(br1, Model1[2:], color ='r', width = barWidth, edgecolor ='black', label ='Model 1') 
    plt.bar(br2, Model2[2:], color ='g', width = barWidth, edgecolor ='black', label ='Model 2') 
    plt.bar(br3, Model3[2:], color ='b', width = barWidth, edgecolor ='black', label ='Model 3') 
    plt.xlabel('Model Metrics') 
    plt.ylabel('Value') 
    plt.xticks([r + barWidth for r in range(2)], 
            #['RMSE', 'MSRE', 
            ['CE', 'RSqr'])

    plt.legend()
    pdf.savefig()
    plt.close()
    #plt.show()

#############NOW POLOTS COMPARING BASELINE AND CHOSEN MODEL ##############

    br1 = np.arange(1) 
    br2 = [x + barWidth for x in br1] 
    br3 = [x + barWidth for x in br2] 
    br4 = [x + barWidth for x in br3] 
    br5 = [x + barWidth for x in br4] 

    plt.bar(br1, Baseline1[0], color ='orange', width = barWidth, edgecolor ='black', label ='Baseline 1') 
    plt.bar(br2, Baseline2[0], color ='pink', width = barWidth, edgecolor ='black', label ='Baseline 2') 
    plt.bar(br3, Model1[0], color ='r', width = barWidth, edgecolor ='black', label ='Model 1') 
    plt.bar(br4, Model2[0], color ='g', width = barWidth, edgecolor ='black', label ='Model 2') 
    plt.bar(br5, Model3[0], color ='b', width = barWidth, edgecolor ='black', label ='Model 3') 
    plt.xlabel('Model Metrics') 
    plt.ylabel('RMSE') 
    #plt.xticks([r + barWidth for r in range(1)], 
    #        ['RMSE'])#, 'MSRE', 'CE', 'RSqr'])

    plt.legend()
    pdf.savefig()
    plt.close()
    #plt.show()

    br1 = np.arange(1) 
    br2 = [x + barWidth for x in br1] 
    br3 = [x + barWidth for x in br2] 
    br4 = [x + barWidth for x in br3] 
    br5 = [x + barWidth for x in br4] 

    plt.bar(br1, Baseline1[1], color ='orange', width = barWidth, edgecolor ='black', label ='Baseline 1') 
    plt.bar(br2, Baseline2[1], color ='pink', width = barWidth, edgecolor ='black', label ='Baseline 2') 
    plt.bar(br3, Model1[1], color ='r', width = barWidth, edgecolor ='black', label ='Model 1') 
    plt.bar(br4, Model2[1], color ='g', width = barWidth, edgecolor ='black', label ='Model 2') 
    plt.bar(br5, Model3[1], color ='b', width = barWidth, edgecolor ='black', label ='Model 3') 
    plt.xlabel('Model Metrics') 
    plt.ylabel('MSRE') 
    plt.yticks([0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1])
    #plt.xticks([r + barWidth for r in range(1)], 
            #['RMSE', 
    #        ['MSRE'])#, 'CE', 'RSqr'])

    plt.legend()
    pdf.savefig()
    plt.close()
    #plt.show()

    br1 = np.arange(2) 
    br2 = [x + barWidth for x in br1] 
    br3 = [x + barWidth for x in br2] 
    br4 = [x + barWidth for x in br3] 
    br5 = [x + barWidth for x in br4] 

    plt.bar(br1, Baseline1[2:], color ='orange', width = barWidth, edgecolor ='black', label ='Baseline 1') 
    plt.bar(br2, Baseline2[2:], color ='pink', width = barWidth, edgecolor ='black', label ='Baseline 2') 
    plt.bar(br3, Model1[2:], color ='r', width = barWidth, edgecolor ='black', label ='Model 1') 
    plt.bar(br4, Model2[2:], color ='g', width = barWidth, edgecolor ='black', label ='Model 2') 
    plt.bar(br5, Model3[2:], color ='b', width = barWidth, edgecolor ='black', label ='Model 3') 
    plt.xlabel('Model Metrics') 
    plt.ylabel('Value') 
    plt.yticks([0.6,0.7,0.8,0.9,1.0])
    plt.ylim(bottom=0.6)
    plt.xticks([r + barWidth for r in range(2)], 
            #['RMSE', 'MSRE', 
            ['CE', 'RSqr'])

    plt.legend()
    pdf.savefig()
    plt.close()





    pdf.close()




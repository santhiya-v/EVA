import pandas as pd

# Eg: xlabel - "Epochs"; ylabel - "Accuracy/Losses"
def plotGraph(data, title, xlabel, ylabel):
    allData = pd.DataFrame(data)
    fig1 = allData.plot(figsize=(15,10), kind = 'line', title = title)
    fig1.set_xlabel(xlabel)
    fig1.set_ylabel(ylabel)
    fig1
'''
Created on 20 Nov. 2017

@author: Alvin UTS
'''
import pandas as pd

def readData(fileName, rows):
    rawData = pd.read_csv(fileName, nrows = rows)

    selectedData = rawData[["marital_status_itemid","ethnicity_itemid","overall_payor_group_itemid","religion_itemid","admission_type_itemid","admission_source_itemid","sex","hospital_expire_flg","stay","freq"]]
    selectedData = selectedData.applymap(str)
    selectedData["stay"][selectedData["stay"] == "00:00:00"] = "0 day"
        
    
    selectedData["stay"] = selectedData["stay"].str.split((" "), expand=True)[0]
    
    selectedData.stay = pd.to_numeric(selectedData.stay, errors='coerce')
    selectedData.freq = pd.to_numeric(selectedData.freq, errors='coerce')
    print 'Data Reading Done', '\n'
    return selectedData







    



# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 12:27:42 2022

@author: HP
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#importing excel file from the URL
def solution(filename, countries, columns):
    """
    Parameters
    ----------
    filename : string
        filename.
    countries : list
        list of countries to be considered.
    columns : list
        list of countries to be used.
    indicator : string
        indicator to be retrieved from the dataset.
    Returns
    -------
    df : pandas dataframe
        original dataframe.
    dataframe
        transposed datframe.
    """
    df = pd.read_excel(filename, sheet_name='Data', skiprows=3)
    df = df[columns]
    df.set_index('Country Name', inplace = True)
    df = df.loc[countries]
    return df, df.transpose()

# The file links of the indicators are stored in the variables below
# importing excel file from the URL

""" This method generates data frame from excel by passing a filepath"""

filename_1 = 'C:/Users/HP/Downloads/API_SL.TLF.TOTL.IN_DS2_en_excel_v2_4757954.xls'

filename_2 = 'C:/Users/HP/Downloads/API_EG.USE.ELEC.KH.PC_DS2_en_excel_v2_4753998 (1).xls'

filename_3 = 'C:/Users/HP/Downloads/API_NY.GDP.PCAP.KD.ZG_DS2_en_excel_v2_4757603.xls'

# These are 5 sample countries to be considered as i ensured the countries are selected across regions.
countries = ['Zimbabwe','India','Canada','Australia','Japan']

# These are sample years to be considered ranging from 2005 - 2014 (10 years).
columns = ['Country Name', '2005','2006','2007','2008','2009','2010','2011','2012','2013','2014']

# These are sample indicators to be considered (labour force, electric power consumption and GDP per capita).
indicators = ['Total labour force', 'Electric power consumption (kWh per capita)','GDP per capita growth(annual %)']

""" This method generates attributes for the function which was used to produce different plots """
# The attributes for the functions were passed and were used to produce various plots for this analysis

cnty_tot_labour_force, year_tot_labour_force = solution(filename_1,countries,columns)
cnty_elec_pow_consmptn, year_elec_pow_consmptn = solution(filename_2,countries,columns)
cnty_gdp_per_capita,year_gdp_per_capita = solution(filename_3,countries,columns)

# Line plot
# importing line plot
# The below multiple line plot was created using matplotlib functions to show a plot of the total labour force of 5 different countries
""" This method plots a line plot by accepting 4 parameters"""
plt.figure(figsize=(10,7),dpi=300)
for i in range(len(countries)):
# This line plot shows Y axis against X axis with Y-axis being the total labour force for 5 different countries and the X-axis being the years(2005-2014)
    plt.plot(year_tot_labour_force.index,year_tot_labour_force[countries[i]],label=countries[i]) 
plt.legend(bbox_to_anchor=(1,1))
plt.title('Trend of the total labour force for these 5 countries', fontsize=20, fontweight='bold')
plt.xlabel('Year', fontsize=20)
plt.ylabel('total labour force', fontsize=20)
plt.savefig('total labour force.png')
plt.show()

# The below multiple line plot is created using matplotlib functions to show a plot of GDP per capita for 5 different countries 
plt.figure(figsize=(10,7),dpi=300)
for i in range(len(countries)):
# This line plot shows Y axis against X axis with Y-axis being the GDP per capita for 5 different countries and the X-axis being the years(2005-2014) 
    plt.plot(year_gdp_per_capita.index,year_gdp_per_capita[countries[i]],label=countries[i]) 
plt.legend(bbox_to_anchor=(1,1))
plt.title('GDP PER CAPITA', fontsize=20, fontweight='bold')
plt.xlabel('Year', fontsize=20)
plt.ylabel('GDP', fontsize=20)
plt.savefig('GDP_per_capita.png')
plt.show()

# Bar plot
# Importing bar plot 
# The parameters below are requirements for the multiple bar plots below
labels_array = ['Zimbabwe','India','Canada','Australia','Japan']
label = ['2006', '2010', '2014']
width = 0.25
# This is the length of label_array
x_values = np.arange(len(labels_array)) 

# The matplotlib functions below plot a multiple bar plot for Total labour force of 5 different countries
# This bar plot shows Y axis against X axis with Y-axis being the total labour force for 5 different countries and the X-axis being the countries and some selected years

# This sets the figsize of the bar plots
fig, ax  = plt.subplots(figsize=(14,12)) 
""" This method plots a bar plot by accepting 6 parameters"""
plt.bar(x_values - width, cnty_tot_labour_force['2006'], width, label=label[0]) # this dictates the size of the plots
plt.bar(x_values, cnty_tot_labour_force['2010'], width, label=label[1])
plt.bar(x_values + width, cnty_tot_labour_force['2014'], width, label=label[2])
    
plt.title('Multiple bar plots showing total labour force across different countries', fontsize=20, fontweight='bold')
plt.ylabel('Total labour force', fontsize=20)
plt.xticks(x_values, labels_array)

plt.legend()
ax.tick_params(bottom=False, left=True)
# The multiple bar plot is displayed below
plt.show() 

# The matplotlib functions below plots a multiple bar plot for electric power consumption of 5 different countries(KWh per capita)
# This bar plot shows Y axis against X axis with Y-axis being the electric power consumption(KWh per capita) for 5 different countries and the X-axis being the countries and some selected years
""" This method plots a bar chart by accepting 6 parameters"""
 # This sets the figsize of the bar plots
fig, ax  = plt.subplots(figsize=(14,12)) 
plt.bar(x_values - width, cnty_elec_pow_consmptn['2006'], width, label=label[0]) 
plt.bar(x_values, cnty_elec_pow_consmptn['2010'], width, label=label[1])
plt.bar(x_values + width, cnty_elec_pow_consmptn['2014'], width, label=label[2])
    
plt.title('Multiple bar plots showing electric power consumption across different countries', fontsize=20, fontweight='bold')
plt.ylabel('Elec_power_Consumption', fontsize=20)
plt.xticks(x_values, labels_array)

plt.legend()
ax.tick_params(bottom=False, left=True)
# The multiple bar plot is displayed below
plt.show()

# A dataframe was created using Canada which takes 3 indicators as parameters
""" This method was created by accepting 3 parameters"""
df_canada = pd.DataFrame({'Labour force': year_tot_labour_force['Canada'],
        'Electricity power consumption': year_elec_pow_consmptn['Canada'],
        'GDP Per Capita': year_gdp_per_capita['Canada']})
print(df_canada)

# The correlation matrix of the dataframe is determined
canada_corr = df_canada.corr()
print(canada_corr)

# Heatmap is done for the above correlation matrix
plt.figure(figsize=(7, 7))
sns.set(font_scale=1.0)
sns.heatmap(canada_corr, annot=True) # seaborn is used to produce the heatmap of the canada indicators
plt.title('Heatmap of Canada', fontsize=22)

# A dataframe was created using Zimbabwe which takes 3 indicators as parameters
df_Zimbabwe = pd.DataFrame({'Labour force': year_tot_labour_force['Zimbabwe'],
        'Electricity power consumption': year_elec_pow_consmptn['Zimbabwe'],
        'GDP Per Capita': year_gdp_per_capita['Zimbabwe']})
print(df_Zimbabwe)

# The correlation matrix of the dataframe is determined
Zimbabwe_corr = df_Zimbabwe.corr()
print(Zimbabwe_corr)

# Heatmap is done for the above correlation matrix
plt.figure(figsize=(7, 7))
sns.set(font_scale=1.0)
sns.heatmap(canada_corr, annot=True) # seaborn is used to produce the heatmap of the canada indicators
plt.title('Heatmap of zimbabwe', fontsize=22)
              
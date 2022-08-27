from cProfile import label
from cmath import nan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

### CHECK FOR INFORMATION LEAK ISSUE - see file INFORMATION LEAK ISSUE.txt ###

# SE_main_v2: Since v1: Here changed how categorical variables are relabelled in the course of dummy coding

# write up: beispiel für einfache auswertung und ableitung ovn erkenntnissen: https://neptune.ai/blog/how-to-implement-customer-churn-prediction

# narrative: nicht domäne extrem in Vordergrund stellen; eher muss klarwerden: geht darum wie man daten vorbereitet und visualisiert, domänenunabhängig

# before publishing results make REALLY sure that they are correctly computed AND plotted! don't trust the plots on their own when checking!

# Loading relevant variables from data into dataframe, while replacing missing values with NaN"
# initially was replacing relevant values here with na_values argument, but then realized can be misleading,
# need to investigate for each clm: what do missing value expressions indicate, how can they be replaced

# 26.03.: It seems that I actually have to set this whole thing up differently: Just comparing population means. If
# I do that, and should use two-sample t-tests, I should point out in narrative that this is also used in A/B testing!

# No missing value imputation: Um Ergebnissverzerrung auszuschließen; dafür den Minimum-Parameter in der Correlation Table Erzwugug verwendet (?)

# GitHub repository 'housekeeping': wie gliedern/speichern professionalle Data Scientists/Programmierer ihren Code? Sicher nicht alles in einem File wie hier oder? 

# important: it seem that what I labelled dummy variable/encoding is actually in some cases actually dummy encoding, but in
# other cases 'ordinal encoding'. Make sure to get the terminology right before putting this online! (if in doubt, avoid terminology)

# hervorheben in Beschreibung: die ausgeprägte "breite" (?) des Datensatzes, Feature-Anzahl
# SQL: ganz am Ende nachträglich: in SQL Datenbank laden und von da direkt abfragen, damit ich sagen kann: SQL-Datenbank Projekt

se_red = pd.read_csv(
    filepath_or_buffer = "C:\\Users\\marc.feldmann\\Documents\\data_science_local\\SE\\se_pooled.csv",
    delimiter = ",",
    usecols = ["id", "ssyjahr", "dem01_h", "dem02_h", "dem03_v21", "dem05", "dem09", "dem11a_h", "par03_v21", "par04_v21", "par05_h", "par06_h", "par07_h", "par08_h", "par09", "par10", "stu01a_h", "stu02_h", "stu03_h", "stu04", "stu05", "stu10_h", "stu11b", "stu11c_v20", "stu11d", "stu11e", "stu11f", "stu11g", "stu11h_v21", "stu12_v21", "stu13_h", "stu16a_h", "stu16b_h", "ped01_v21", "ped03", "fin01a", "fin01b", "fin01c", "fin01d", "fin01e", "fin01f_h", "fin01g", "fin01h", "fin01j", "fin01o", "fin02a_h", "fin02b_h", "fin02c_h", "fin02d_h", "fin02e_h", "fin02f_h", "fin02g_h", "fin02h_h", "fin02i_h", "fin03a_h", "fin03b_h", "fin03c_h", "fin03d_h", "fin03e_h", "fin03f_h", "fin03g_h", "fin03h_h", "fin03i_h", "fin04a", "fin04b", "fin04c", "fin04d_v21", "fin04e_v21", "fin04f_v21", "fin04i", "baf01_h", "baf05a", "baf05b", "baf05c", "baf05e", "baf05f_v20", "baf05g", "baf05h_v21", "baf05i_v21", "tim02a_v21", "tim02b_v21", "tim02c_v21", "tim02d_v21", "tim02e_v21", "tim02f_v21", "tim02g_v21", "tim03a_v21", "tim03b_v21", "tim03c_v21", "tim03d_v21", "tim03e_v21", "tim03f_v21", "tim03g_v21", "job02_h", "job03a_v21", "job03b_v21", "job03g_v21", "job03n_v21", "job05a", "job05b", "job05c", "job05d", "job05e", "job05f", "job05g", "job05h", "abr01_h", "abr02a_v21", "abr02b_v21", "abr02d_v21", "abr03a_v21", "abr03b_v21", "abr03d_v21", "abr08a_v21", "abr08b_v21", "abr08c_v21", "abr08d_v21", "abr08e_v21", "abr08f_v21", "abr08g_v21", "abr08h_v21", "abr08i_v21", "abr09a_v21", "abr09b_v21", "abr09c_v21", "abr09d_v21", "abr09e_v21", "abr09f_v21", "abr09g_v21", "abr09h_v21", "abr09i_v21", "abr11_h", "abr12_h", "abr13a_v21", "abr13b_v21", "abr13c_v21", "abr13d_v21", "abr13e_v21", "abr13f_v21", "lan01_v21", "liv01_v21", "adv01a_v21", "adv01b_v21", "adv01c_v21", "adv01d_v21", "adv01e_v21", "adv01g_v21", "adv01i_v21", "adv01j_v21", "adv01k_v21", "adv01l_v21", "adv01m_v21", "adv01n_v21", "adv01o_v21", "adv01p_v21", "adv02_v21"],
#    na_values = ["verweigert", "interviewabbruch", "unbekannter fehlender Wert", "nicht bestimmbar", "keine Angabe (Antwortkategorie)", "keine Angabe", "weiß nicht", "nicht genannt", "ssy-spezifisch fehlend", "splitbedingt fehlend", "filterbedingt fehlend", "designbedingt fehlend", "kohortenbedingt fehlend"],
    encoding = "utf-8"
)

# Dropping all data except most recent survey data
se_red_17 = se_red[se_red["ssyjahr"] == 2016]
se_red_17 = se_red_17.reset_index()
se_red_17 = se_red_17.drop("ssyjahr", 1)

# Checking data types
print(se_red_17.dtypes)

# Checking for duplicate rows
print(se_red_17[se_red_17.duplicated()].shape)


# ------- DATA EXPLORATION AND CLEANING ------- #
# for each column: 
# a) explore unique values and/or distribution, based on that:
# b) interpret/replace missing values (see docu file)
# c) aggregate and/or rename feature values to make better understandable in later output
# d) make sure that existing nominal and ordinal categorical variables are coded numerically (e.g., Likert-scale variables such as "Beratungsbedarf")
# e) declare column data type
## in docu: give rationale why I was not simply replacing all relevant values with read_csv parameter (as I did originally:)
## because different types of 'missing value', need to be inspected individually; also: same type can mean different things
## in different features (insb. "splitbedingt fehlend"; auch: e.g. "unbestimmbar" in Geschlecht oder e.g. Schulabschluss des Vaters)  

## Geschlecht (dem01_h)
round(se_red_17["dem01_h"].value_counts(normalize=True, dropna=False), 3)
se_red_17.loc[se_red_17["dem01_h"] == "nicht bestimmbar", "dem01_h"] = "divers"
se_red_17["dem01_h"].dtype

## Alter (dem02_h)
se_red_17["dem02_h"].dtype
round(se_red_17["dem02_h"].value_counts(normalize=True, dropna=False), 3)

### Declare "keine Angabe" (n = 123/27583) as NaN
se_red_17.loc[se_red_17["dem02_h"] == "keine Angabe", "dem02_h"] = np.nan

### replace identified non-numerical values in age (dem02_h) with imputed, randomized numerical values; assumption: uniform age distribution
### value "40-49 Jahre" (n = 236/27583)

### iterate through positions in dem02_h column and replace with integer sampled from uniform distribution
list = se_red_17[se_red_17["dem02_h"]=="40-49 Jahre"].index.values
for i in list:
    se_red_17.iloc[i, se_red_17.columns.get_loc("dem02_h")] = str(np.random.randint(40, 49))

### repeat for  value "50 Jahre und Älter" (n = 123/27583); assumption: uniform age distribution, max. population age = 83     
list = se_red_17[se_red_17["dem02_h"]=="50 Jahre und älter"].index.values
for i in list:
    se_red_17.iloc[i, se_red_17.columns.get_loc("dem02_h")] = str(np.random.randint(50, 83))

se_red_17["dem02_h"] = pd.to_numeric(se_red_17["dem02_h"], errors='coerce')


## Familienstand (demo03_v21)
se_red_17["dem03_v21"].dtype
round(se_red_17["dem03_v21"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs
se_red_17.loc[se_red_17["dem03_v21"] == "keine Angabe", "dem03_v21"] = np.nan
se_red_17.loc[se_red_17["dem03_v21"] == "keine Angabe (Antwortkategorie)", "dem03_v21"] = np.nan
se_red_17.loc[se_red_17["dem03_v21"] == "Interviewabbruch", "dem03_v21"] = np.nan

### Aggregate "feste Partnerbeziehung" and "verheiratet/eingetr. Partnerschaft" as "feste Partnerbeziehung" 
se_red_17.loc[se_red_17["dem03_v21"] == "verheiratet/eingetr. Partnerschaft", "dem03_v21"] = "feste Partnerbeziehung"

### Code binary: "1" if in relationship
se_red_17.loc[se_red_17["dem03_v21"] == "feste Partnerbeziehung", "dem03_v21"] = 1
se_red_17.loc[se_red_17["dem03_v21"] == "ohne Partnerbeziehung", "dem03_v21"] = 0
se_red_17["dem03_v21"] = pd.to_numeric(se_red_17["dem03_v21"], errors='coerce')
se_red_17.rename(columns={"dem03_v21": "DUM_dem03_v21"}, inplace=True)


## Kinder (dem05)
se_red_17["dem05"].dtype
round(se_red_17["dem05"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs
se_red_17.loc[se_red_17["dem05"] == "keine Angabe", "dem05"] = np.nan
se_red_17.loc[se_red_17["dem05"] == "unbekannter fehlender Wert", "dem05"] = np.nan
se_red_17.loc[se_red_17["dem05"] == "Interviewabbruch", "dem05"] = np.nan

### Code binary: "1" if has kids
se_red_17.loc[se_red_17["dem05"] == "ja", "dem05"] = 1
se_red_17.loc[se_red_17["dem05"] == "nein", "dem05"] = 0
se_red_17["dem05"] = pd.to_numeric(se_red_17["dem05"], errors='coerce')
se_red_17.rename(columns={"dem05": "DUM_dem05"}, inplace=True)


## Geschwister (dem09)
se_red_17["dem09"].dtype
round(se_red_17["dem09"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs
se_red_17.loc[se_red_17["dem09"] == "keine Angabe", "dem09"] = np.nan
se_red_17.loc[se_red_17["dem09"] == "Interviewabbruch", "dem09"] = np.nan

### Code binary: "1" if has siblings
se_red_17.loc[se_red_17["dem09"] == "ja", "dem09"] = 1
se_red_17.loc[se_red_17["dem09"] == "nein", "dem09"] = 0
se_red_17["dem09"] = pd.to_numeric(se_red_17["dem09"], errors='coerce')
se_red_17.rename(columns={"dem09": "DUM_dem09"}, inplace=True)


## dt. Staatsangehörigkeit (dem11a_h)
se_red_17["dem11a_h"].dtype
round(se_red_17["dem11a_h"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs
se_red_17.loc[se_red_17["dem11a_h"] == "nicht bestimmbar", "dem11a_h"] = np.nan


## Vater: hoechster Schulabschluss (par03_v21)
se_red_17["par03_v21"].dtype
round(se_red_17["par03_v21"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs
se_red_17.loc[se_red_17["par03_v21"] == "keine Angabe", "par03_v21"] = np.nan
se_red_17.loc[se_red_17["par03_v21"] == "weiß nicht", "par03_v21"] = np.nan
se_red_17.loc[se_red_17["par03_v21"] == "Interviewabbruch", "par03_v21"] = np.nan


## Mutter: hoechster Schulabschluss (par04_v21)
se_red_17["par04_v21"].dtype
round(se_red_17["par04_v21"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs
se_red_17.loc[se_red_17["par04_v21"] == "keine Angabe", "par04_v21"] = np.nan
se_red_17.loc[se_red_17["par04_v21"] == "weiß nicht", "par04_v21"] = np.nan
se_red_17.loc[se_red_17["par04_v21"] == "Interviewabbruch", "par04_v21"] = np.nan


## Vater: hoechster berufl. Abschluss (par05_h)
se_red_17["par05_h"].dtype
round(se_red_17["par05_h"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs
se_red_17.loc[se_red_17["par05_h"] == "keine Angabe", "par05_h"] = np.nan
se_red_17.loc[se_red_17["par05_h"] == "weiß nicht/nicht bekannt", "par05_h"] = np.nan
se_red_17.loc[se_red_17["par05_h"] == "Interviewabbruch", "par05_h"] = np.nan


## Mutter: hoechster berufl. Abschluss (par05_h)
se_red_17["par06_h"].dtype
round(se_red_17["par06_h"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs
se_red_17.loc[se_red_17["par06_h"] == "keine Angabe", "par06_h"] = np.nan
se_red_17.loc[se_red_17["par06_h"] == "weiß nicht/nicht bekannt", "par06_h"] = np.nan
se_red_17.loc[se_red_17["par06_h"] == "Interviewabbruch", "par06_h"] = np.nan


## Vater: berufl. Position (par07_h)
se_red_17["par07_h"].dtype
round(se_red_17["par07_h"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs
se_red_17.loc[se_red_17["par07_h"] == "keine Angabe", "par07_h"] = np.nan
se_red_17.loc[se_red_17["par07_h"] == "weiß nicht", "par07_h"] = np.nan
se_red_17.loc[se_red_17["par07_h"] == "Interviewabbruch", "par07_h"] = np.nan

### Aggregate "Selbständiger" and "freiberuflich tätig" as "Selbständiger" 
se_red_17.loc[se_red_17["par07_h"] == "freiberuflich tätig", "par07_h"] = "Selbständiger"


## Mutter: berufl. Position (par08_h)
se_red_17["par08_h"].dtype
round(se_red_17["par08_h"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs
se_red_17.loc[se_red_17["par08_h"] == "keine Angabe", "par08_h"] = np.nan
se_red_17.loc[se_red_17["par08_h"] == "weiß nicht", "par08_h"] = np.nan
se_red_17.loc[se_red_17["par08_h"] == "Interviewabbruch", "par08_h"] = np.nan

### Aggregate "Selbständige" and "freiberuflich tätig" as "Selbständige" 
se_red_17.loc[se_red_17["par08_h"] == "freiberuflich tätig", "par08_h"] = "Selbständige"


## Vater: Staatsangehörigkeit (par09)
se_red_17["par09"].dtype
round(se_red_17["par09"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs
se_red_17.loc[se_red_17["par09"] == "nicht bestimmbar", "par09"] = np.nan
se_red_17.loc[se_red_17["par09"] == "weiß nicht", "par09"] = np.nan


## Mutter: Staatsangehörigkeit (par10)
se_red_17["par10"].dtype
round(se_red_17["par10"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs
se_red_17.loc[se_red_17["par10"] == "nicht bestimmbar", "par10"] = np.nan
se_red_17.loc[se_red_17["par10"] == "weiß nicht", "par10"] = np.nan


## 1. Studienfach (stu01a_h)
se_red_17["stu01a_h"].dtype
round(se_red_17["stu01a_h"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs / rename values
se_red_17.loc[se_red_17["stu01a_h"] == "keine Angabe", "stu01a_h"] = np.nan
se_red_17.loc[se_red_17["stu01a_h"] == "unbekannter fehlender Wert", "stu01a_h"] = np.nan
se_red_17.loc[se_red_17["stu01a_h"] == "außerh. amtl. Fächergruppen", "stu01a_h"] = "andere"


## angestrebter Abschluss (stu02_h)
se_red_17["stu02_h"].dtype
round(se_red_17["stu02_h"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs / rename values
se_red_17.loc[se_red_17["stu02_h"] == "keine Angabe", "stu02_h"] = np.nan
se_red_17.loc[se_red_17["stu02_h"] == "Bachelor (nicht Lehramt)", "stu02_h"] = "Bachelor"
se_red_17.loc[se_red_17["stu02_h"] == "Bachelor mit Ziel Lehramt", "stu02_h"] = "Bachelor"
se_red_17.loc[se_red_17["stu02_h"] == "Master (nicht Lehramt)", "stu02_h"] = "Master"
se_red_17.loc[se_red_17["stu02_h"] == "Master mit Ziel Lehramt", "stu02_h"] = "Master"
se_red_17.loc[se_red_17["stu02_h"] == "Staatsexamen (ohne Lehramt)", "stu02_h"] = "Staatsexamen"
se_red_17.loc[se_red_17["stu02_h"] == "Staatsexamen mit Ziel Lehramt", "stu02_h"] = "Staatsexamen"
se_red_17.loc[se_red_17["stu02_h"] == "Diplom Univ./Kunsthochsch.", "stu02_h"] = "Diplom"
se_red_17.loc[se_red_17["stu02_h"] == "Fachhochschuldiplom", "stu02_h"] = "Diplom"


## vorhandener Abschluss (stu03_h)
se_red_17["stu03_h"].dtype
round(se_red_17["stu03_h"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs / rename values
se_red_17.loc[se_red_17["stu03_h"] == "keine Angabe", "stu03_h"] = np.nan
se_red_17.loc[se_red_17["stu03_h"] == "splitbedingt fehlend", "stu03_h"] = "kein vorhandener Abschluss"
se_red_17.loc[se_red_17["stu03_h"] == "filterbedingt fehlend", "stu03_h"] = "kein vorhandener Abschluss"
se_red_17.loc[se_red_17["stu03_h"] == "Bachelor (nicht Lehramt)", "stu03_h"] = "Bachelor"
se_red_17.loc[se_red_17["stu03_h"] == "Bachelor (mit Ziel Lehramt)", "stu03_h"] = "Bachelor"
se_red_17.loc[se_red_17["stu03_h"] == "Master (nicht Lehramt)", "stu03_h"] = "Master"
se_red_17.loc[se_red_17["stu03_h"] == "Master (mit Ziel Lehramt)", "stu03_h"] = "Master"
se_red_17.loc[se_red_17["stu03_h"] == "Staatsexamen (ohne Lehramt)", "stu03_h"] = "Staatsexamen"
se_red_17.loc[se_red_17["stu03_h"] == "Staatsexamen für ein Lehramt", "stu03_h"] = "Staatsexamen"
se_red_17.loc[se_red_17["stu03_h"] == "Diplom Universität/Kunsthochsch. o. Ä.", "stu03_h"] = "Diplom"
se_red_17.loc[se_red_17["stu03_h"] == "Fachhochschuldiplom", "stu03_h"] = "Diplom"


## Fachsemester (stu04)
se_red_17["stu04"].dtype
round(se_red_17["stu04"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs
se_red_17.loc[se_red_17["stu04"] == "keine Angabe", "stu04"] = np.nan
se_red_17.loc[se_red_17["stu04"] == "unbekannter fehlender Wert", "stu04"] = np.nan
se_red_17["stu04"] = pd.to_numeric(se_red_17["stu04"], errors='coerce')


## Hochschulsemester (stu05)
se_red_17["stu05"].dtype
round(se_red_17["stu05"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs
se_red_17.loc[se_red_17["stu05"] == "keine Angabe", "stu05"] = np.nan
se_red_17.loc[se_red_17["stu05"] == "nicht bestimmbar", "stu05"] = np.nan
se_red_17["stu05"] = pd.to_numeric(se_red_17["stu05"], errors='coerce')


## Studienunterbrechung (stu10_h)
se_red_17["stu10_h"].unique()
se_red_17["stu10_h"].dtype
round(se_red_17["stu10_h"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs
se_red_17.loc[se_red_17["stu10_h"] == "keine Angabe", "stu10_h"] = np.nan
se_red_17.loc[se_red_17["stu10_h"] == "unbekannter fehlender Wert", "stu10_h"] = np.nan

### Code binary: "1" if has interrupted studies
se_red_17.loc[se_red_17["stu10_h"] == "ja", "stu10_h"] = 1
se_red_17.loc[se_red_17["stu10_h"] == "nein", "stu10_h"] = 0
se_red_17["stu10_h"] = pd.to_numeric(se_red_17["stu10_h"], errors='coerce')
se_red_17.rename(columns={"stu10_h": "DUM_stu10_h"}, inplace=True)

## Unterbrechungsgrund (stu11X)
### Declare NaNs / rename values

#### for all columns that start with stu11, show unique values
#### to identify values  which need to be replaced with np.nan
list = ["stu11b", "stu11c_v20", "stu11d", "stu11e", "stu11f", "stu11g", "stu11h_v21"]
for i in list:
    round(se_red_17[i].value_counts(normalize=True, dropna=False), 3)

#### create new loop to replace with np.nan and code binary, "1" if interruption reason mentioned
for i in list:
    se_red_17.loc[se_red_17[i] == "filterbedingt fehlend", i] = np.nan
    se_red_17.loc[se_red_17[i] == "splitbedingt fehlend", i] = np.nan
    se_red_17.loc[se_red_17[i] == "keine Angabe", i] = np.nan
    se_red_17.loc[se_red_17[i] == "genannt", i] = 1
    se_red_17.loc[se_red_17[i] == "nicht genannt", i] = 0
    se_red_17[i] = pd.to_numeric(se_red_17[i], errors='coerce')

se_red_17.rename(columns={"stu11b": "DUM_stu11b", "stu11c_v20": "DUM_stu11c_v20", "stu11d": "DUM_stu11d", "stu11e": "DUM_stu11e", "stu11f": "DUM_stu11f", "stu11g": "DUM_stu11g", "stu11h_v21": "DUM_stu11h_v21"}, inplace=True)


## Gesamtdauer Studienunterbrechung (Semester) (stu12_v21)
se_red_17["stu12_v21"].dtype
round(se_red_17["stu12_v21"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs / rename values
se_red_17.loc[se_red_17["stu12_v21"] == "nicht bestimmbar", "stu12_v21"] = np.nan
se_red_17.loc[se_red_17["stu12_v21"] == "splitbedingt fehlend", "stu12_v21"] = np.nan

se_red_17["stu12_v21"] = pd.to_numeric(se_red_17["stu12_v21"], errors='coerce')


## Hochschule mind. 1 Mal gewechselt? (stu13_h)
se_red_17["stu13_h"].dtype
round(se_red_17["stu13_h"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs
se_red_17.loc[se_red_17["stu13_h"] == "keine Angabe", "stu13_h"] = np.nan

### Code binary: "1" if has changed higher education institution at least once
se_red_17.loc[se_red_17["stu13_h"] == "ja, ein- od. mehrmals", "stu13_h"] = 1
se_red_17.loc[se_red_17["stu13_h"] == "nein", "stu13_h"] = 0
se_red_17["stu13_h"] = pd.to_numeric(se_red_17["stu13_h"], errors='coerce')
se_red_17.rename(columns={"stu13_h": "DUM_stu13_h"}, inplace=True)


## Hochschulart (stu16a_h)
se_red_17["stu16a_h"].dtype
round(se_red_17["stu16a_h"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs
se_red_17.loc[se_red_17["stu16a_h"] == "keine Angabe", "stu16a_h"] = np.nan
se_red_17.loc[se_red_17["stu16a_h"] == "nicht bestimmbar", "stu16a_h"] = np.nan

### Code binary: "1" if university, "0" if university of applied science
se_red_17.loc[se_red_17["stu16a_h"] == "Universität", "stu16a_h"] = 1
se_red_17.loc[se_red_17["stu16a_h"] == "Fachhochschule", "stu16a_h"] = 0
se_red_17["stu16a_h"] = pd.to_numeric(se_red_17["stu16a_h"], errors='coerce')
se_red_17.rename(columns={"stu16a_h": "DUM_stu16a_h"}, inplace=True)


## Hochschulregion (stu16b_h)
se_red_17["stu16b_h"].dtype
round(se_red_17["stu16b_h"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs / rename values
se_red_17.loc[se_red_17["stu16b_h"] == "Ostdeutschland, inkl. Berlin", "stu16b_h"] = "Ostdeutschland"
se_red_17.loc[se_red_17["stu16b_h"] == "nicht bestimmbar", "stu16b_h"] = np.nan


## Art der Studienberechtigung (ped01_v21)
se_red_17["ped01_v21"].dtype
round(se_red_17["ped01_v21"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs
se_red_17.loc[se_red_17["ped01_v21"] == "keine Angabe", "ped01_v21"] = "andere Studienberechtigung"
se_red_17.loc[se_red_17["ped01_v21"] == "filterbedingt fehlend", "ped01_v21"] = "andere Studienberechtigung"
se_red_17.loc[se_red_17["ped01_v21"] == "splitbedingt fehlend", "ped01_v21"] = "andere Studienberechtigung"


## Berufsausbildung vor Erstimmatrikulation (ped03)
se_red_17["ped03"].dtype
round(se_red_17["ped03"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs
se_red_17.loc[se_red_17["ped03"] == "keine Angabe", "ped03"] = np.nan
se_red_17.loc[se_red_17["ped03"] == "filterbedingt fehlend", "ped03"] = np.nan

### Code binary: "1" if has completed vocational training before studies
se_red_17.loc[se_red_17["ped03"] == "ja", "ped03"] = 1
se_red_17.loc[se_red_17["ped03"] == "nein", "ped03"] = 0
se_red_17["ped03"] = pd.to_numeric(se_red_17["ped03"], errors='coerce')
se_red_17.rename(columns={"ped03": "DUM_ped03"}, inplace=True)


## Barmittel (fin01X)
### Declare NaNs

#### for all columns that start with fin01, show unique values
#### to identify values  which need to be replaced with np.nan
list = ["fin01a", "fin01b", "fin01c", "fin01d", "fin01e", "fin01f_h", "fin01g", "fin01h", "fin01j", "fin01o"]
for i in list:
    se_red_17[i].unique()

#### create new loop to replace with np.nan and enforce to_numeric in all columns starting with fin01
for i in list:
    se_red_17.loc[se_red_17[i] == "keine Angabe", i] = np.nan
    se_red_17.loc[se_red_17[i] == "unbekannter fehlender Wert", i] = np.nan
    se_red_17.loc[se_red_17[i] == "splitbedingt fehlend", i] = np.nan
    se_red_17.loc[se_red_17[i] == "filterbedingt fehlend", i] = np.nan
    se_red_17[i] = pd.to_numeric(se_red_17[i], errors='coerce')
    se_red_17[i] = round(se_red_17[i], 3)


## Barmittel (fin02X)
### Declare NaNs

#### for all columns that start with fin02, show unique values
#### to identify values  which need to be replaced with np.nan
list = ["fin02a_h", "fin02b_h", "fin02c_h", "fin02d_h", "fin02e_h", "fin02f_h", "fin02g_h", "fin02h_h", "fin02i_h"]
for i in list:
    se_red_17[i].unique()

#### create new loop to replace with np.nan and enforce to_numeric in all columns starting with fin01
for i in list:
    se_red_17.loc[se_red_17[i] == "keine Angabe", i] = np.nan
    se_red_17.loc[se_red_17[i] == "unbekannter fehlender Wert", i] = np.nan
    se_red_17.loc[se_red_17[i] == "splitbedingt fehlend", i] = np.nan
    se_red_17.loc[se_red_17[i] == "filterbedingt fehlend", i] = np.nan
    se_red_17[i] = pd.to_numeric(se_red_17[i], errors='coerce')
    se_red_17[i] = round(se_red_17[i], 3)


## Barmittel (fin03X)
### Declare NaNs

#### for all columns that start with fin03, show unique values
#### to identify values  which need to be replaced with np.nan
list = ["fin03a_h", "fin03b_h", "fin03c_h", "fin03d_h", "fin03e_h", "fin03f_h", "fin03g_h", "fin03h_h", "fin03i_h"]
for i in list:
    round(se_red_17[i].value_counts(normalize=True, dropna=False), 3)

#### create new loop to replace with np.nan and enforce to_numeric in all columns starting with fin01
for i in list:
    se_red_17.loc[se_red_17[i] == "keine Angabe", i] = np.nan
    se_red_17.loc[se_red_17[i] == "unbekannter fehlender Wert", i] = np.nan
    se_red_17.loc[se_red_17[i] == "splitbedingt fehlend", i] = np.nan
    se_red_17.loc[se_red_17[i] == "filterbedingt fehlend", i] = np.nan
    se_red_17[i] = pd.to_numeric(se_red_17[i], errors='coerce')
    se_red_17[i] = round(se_red_17[i], 3)


## Barmittel (fin04X)
### Declare NaNs

#### for all columns that start with fin04, show unique values
#### to identify values  which need to be replaced with np.nan
list = ["fin04a", "fin04b", "fin04c", "fin04d_v21", "fin04e_v21", "fin04f_v21", "fin04i"]
for i in list:
    round(se_red_17[i].value_counts(normalize=True, dropna=False), 3)

#### create new loop to replace with np.nan and enforce to_numeric in all columns starting with fin01
for i in list:
    se_red_17.loc[se_red_17[i] == "keine Angabe", i] = np.nan
    se_red_17.loc[se_red_17[i] == "passt nicht", i] = np.nan
    se_red_17.loc[se_red_17[i] == "splitbedingt fehlend", i] = np.nan
    se_red_17.loc[se_red_17[i] == "filterbedingt fehlend", i] = np.nan
    se_red_17.loc[se_red_17[i] == "trifft völlig zu", i] = 5
    se_red_17.loc[se_red_17[i] == "Pos. 4", i] = 4
    se_red_17.loc[se_red_17[i] == "Pos. 3", i] = 3
    se_red_17.loc[se_red_17[i] == "Pos. 2", i] = 2
    se_red_17.loc[se_red_17[i] == "trifft gar nicht zu", i] = 1
    se_red_17[i] = pd.to_numeric(se_red_17[i], errors='coerce')


## BAfoeG-Förderung im aktuellen SoSe? (baf01_h)
se_red_17["baf01_h"].dtype
round(se_red_17["baf01_h"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs and code binary, "1" if has job in current semester
se_red_17.loc[se_red_17["baf01_h"] == "keine Angabe", "baf01_h"] = np.nan
se_red_17.loc[se_red_17["baf01_h"] == "ja", "baf01_h"] = 1
se_red_17.loc[se_red_17["baf01_h"] == "nein", "baf01_h"] = 0
se_red_17.loc[se_red_17["baf01_h"] == "Antrag noch nicht entschieden", "baf01_h"] = 0
se_red_17["baf01_h"] = pd.to_numeric(se_red_17["baf01_h"], errors='coerce')
se_red_17.rename(columns={"baf01_h": "DUM_baf01_h"}, inplace=True)


## keine BAfoeG-Förderung Gründe (baf05X)
### Declare NaNs / rename values

#### for all columns that start with baf05, show unique values
#### to identify values  which need to be replaced with np.nan
list = ["baf05a", "baf05b", "baf05c", "baf05e", "baf05f_v20", "baf05g", "baf05h_v21", "baf05i_v21"]
for i in list:
    round(se_red_17[i].value_counts(normalize=True, dropna=False), 3)

#### create new loop to 
### replace with np.nan, enforce to_numeric in all columns starting with fin01, and code binary: "1" if reason for not applying for BafoeG mentioned
for i in list:
    se_red_17.loc[se_red_17[i] == "unbekannter fehlender Wert", i] = np.nan
    se_red_17.loc[se_red_17[i] == "splitbedingt fehlend", i] = np.nan
    se_red_17.loc[se_red_17[i] == "filterbedingt fehlend", i] = np.nan
    se_red_17.loc[se_red_17[i] == "genannt", i] = 1
    se_red_17.loc[se_red_17[i] == "nicht genannt", i] = 0
    se_red_17[i] = pd.to_numeric(se_red_17[i], errors='coerce')

se_red_17.rename(columns={"baf05a": "DUM_baf05a", "baf05b": "DUM_baf05b", "baf05c": "DUM_baf05c", "baf05e": "DUM_baf05e", "baf05f_v20": "DUM_baf05f_v20", "baf05g": "DUM_baf05g", "baf05h_v21": "DUM_baf05h_v21", "baf05i": "DUM_baf05i_v21", }, inplace=True)


## Studienaufwand in h/Tag (jenseits von Lehrveranstaltungen) (tim02X)
### Declare NaNs

#### for all columns that start with tim02, show unique values
#### to identify values  which need to be replaced with np.nan
list = ["tim02a_v21", "tim02b_v21", "tim02c_v21", "tim02d_v21", "tim02e_v21", "tim02f_v21", "tim02g_v21"]
for i in list:
    round(se_red_17[i].value_counts(normalize=True, dropna=False), 3)

#### create new loop to replace with np.nan and enforce to_numeric in all columns starting with fin01
for i in list:
    se_red_17.loc[se_red_17[i] == "unbekannter fehlender Wert", i] = np.nan
    se_red_17.loc[se_red_17[i] == "Interviewabbruch", i] = np.nan
    se_red_17.loc[se_red_17[i] == "keine Angabe", i] = np.nan
    se_red_17.loc[se_red_17[i] == "splitbedingt fehlend", i] = np.nan
    se_red_17[i] = pd.to_numeric(se_red_17[i], errors='coerce')


## Erwerbstätigkeit im laufenden Semester (job02_h)
se_red_17["job02_h"].dtype
round(se_red_17["job02_h"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs and code binary, "1" if has job in current semester
se_red_17.loc[se_red_17["job02_h"] == "keine Angabe", "job02_h"] = np.nan
se_red_17.loc[se_red_17["job02_h"] == "splitbedingt fehlend", "job02_h"] = np.nan
se_red_17.loc[se_red_17["job02_h"] == "ja", "job02_h"] = 1
se_red_17.loc[se_red_17["job02_h"] == "nein", "job02_h"] = 0
se_red_17["job02_h"] = pd.to_numeric(se_red_17["job02_h"], errors='coerce')
se_red_17.rename(columns={"job02_h": "DUM_job02_h"}, inplace=True)


## Erwerbstätigkeit in h/Tag (tim03X)
### Declare NaNs

#### for all columns that start with tim02, show unique values
#### to identify values  which need to be replaced with np.nan
list = ["tim03a_v21", "tim03b_v21", "tim03c_v21", "tim03d_v21", "tim03e_v21", "tim03f_v21", "tim03g_v21"]
for i in list:
    round(se_red_17[i].value_counts(normalize=True, dropna=False), 3)

#### create new loop to replace with np.nan and enforce to_numeric
for i in list:
    se_red_17.loc[se_red_17[i] == "unbekannter fehlender Wert", i] = np.nan
    se_red_17.loc[se_red_17[i] == "Interviewabbruch", i] = np.nan
    se_red_17.loc[se_red_17[i] == "keine Angabe", i] = np.nan
    se_red_17.loc[se_red_17[i] == "splitbedingt fehlend", i] = np.nan
    se_red_17[i] = pd.to_numeric(se_red_17[i], errors='coerce')


## Aushilfstätigkeit/Jobben (job03a_v21)
se_red_17["job03a_v21"].dtype
round(se_red_17["job03a_v21"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs
se_red_17.loc[se_red_17["job03a_v21"] == "keine Angabe", "job03a_v21"] = np.nan

### Code binary: "1" if job in current semester is a temp job
se_red_17.loc[se_red_17["job03a_v21"] == "genannt", "job03a_v21"] = 1
se_red_17.loc[se_red_17["job03a_v21"] == "nicht genannt", "job03a_v21"] = 0
se_red_17["job03a_v21"] = pd.to_numeric(se_red_17["job03a_v21"], errors='coerce')
se_red_17.rename(columns={"job03a_v21": "DUM_job03a_v21"}, inplace=True)


## studentische/wiss. Hilfskraft (job03b_v21)
se_red_17["job03b_v21"].dtype
round(se_red_17["job03b_v21"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs
se_red_17.loc[se_red_17["job03b_v21"] == "keine Angabe", "job03b_v21"] = np.nan

### Code binary: "1" if job in current semester is a research assistant job
se_red_17.loc[se_red_17["job03b_v21"] == "genannt", "job03b_v21"] = 1
se_red_17.loc[se_red_17["job03b_v21"] == "nicht genannt", "job03b_v21"] = 0
se_red_17["job03b_v21"] = pd.to_numeric(se_red_17["job03b_v21"], errors='coerce')
se_red_17.rename(columns={"job03b_v21": "DUM_job03b_v21"}, inplace=True)


## Praktikum/Volontariat (job03g_v21)
se_red_17["job03g_v21"].dtype
round(se_red_17["job03g_v21"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs
se_red_17.loc[se_red_17["job03g_v21"] == "keine Angabe", "job03g_v21"] = np.nan

### Code binary: "1" if job in current semester is an internship (paid/unpaid)
se_red_17.loc[se_red_17["job03g_v21"] == "genannt", "job03g_v21"] = 1
se_red_17.loc[se_red_17["job03g_v21"] == "nicht genannt", "job03g_v21"] = 0
se_red_17["job03g_v21"] = pd.to_numeric(se_red_17["job03g_v21"], errors='coerce')
se_red_17.rename(columns={"job03g_v21": "DUM_job03g_v21"}, inplace=True)


## berufsbegleitendes Studium (job03n_v21)
se_red_17["job03n_v21"].dtype
round(se_red_17["job03n_v21"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs / rename values
se_red_17.loc[se_red_17["job03n_v21"] == "keine Angabe", "job03n_v21"] = np.nan
se_red_17.loc[se_red_17["job03n_v21"] == "filterbedingt fehlend", "job03n_v21"] = np.nan
se_red_17.loc[se_red_17["job03n_v21"] == "splitbedingt fehlend", "job03n_v21"] = np.nan


## Erwerbsgrund(job05X)
### Declare NaNs

#### for all columns that start with job05, show unique values
#### to identify values  which need to be replaced with np.nan
list = ["job05a", "job05b", "job05c", "job05d", "job05e", "job05f", "job05g", "job05h"]
for i in list:
    round(se_red_17[i].value_counts(normalize=True, dropna=False), 3)

#### create new loop to replace with np.nan
for i in list:
    se_red_17.loc[se_red_17[i] == "keine Angabe", i] = np.nan
    se_red_17.loc[se_red_17[i] == "passt nicht", i] = np.nan
    se_red_17.loc[se_red_17[i] == "splitbedingt fehlend", i] = np.nan
    se_red_17.loc[se_red_17[i] == "filterbedingt fehlend", i] = np.nan
    se_red_17.loc[se_red_17[i] == "trifft völlig zu", i] = 5
    se_red_17.loc[se_red_17[i] == "Pos. 4", i] = 4
    se_red_17.loc[se_red_17[i] == "Pos. 3", i] = 3
    se_red_17.loc[se_red_17[i] == "Pos. 2", i] = 2
    se_red_17.loc[se_red_17[i] == "trifft gar nicht zu", i] = 1
    se_red_17[i] = pd.to_numeric(se_red_17[i], errors='coerce')


## AUSLAND
## Studienbezogener Auslandsaufenthalt? (abr01_h)
se_red_17["abr01_h"].dtype
round(se_red_17["abr01_h"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs
se_red_17.loc[se_red_17["abr01_h"] == "keine Angabe", "abr01_h"] = np.nan
se_red_17.loc[se_red_17["abr01_h"] == "Interviewabbruch", "abr01_h"] = np.nan

### Code binary: "1" if has been abroad during studies
se_red_17.loc[se_red_17["abr01_h"] == "ja", "abr01_h"] = 1
se_red_17.loc[se_red_17["abr01_h"] == "nein", "abr01_h"] = 0
se_red_17["abr01_h"] = pd.to_numeric(se_red_17["abr01_h"], errors='coerce')
se_red_17.rename(columns={"abr01_h": "DUM_abr01_h"}, inplace=True)


## Auslandsaufenthalt Programmbestandteil? (abr11_h)
se_red_17["abr11_h"].dtype
round(se_red_17["abr11_h"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs / rename values
se_red_17.loc[se_red_17["abr11_h"] == "filterbedingt fehlend", "abr11_h"] = np.nan
se_red_17.loc[se_red_17["abr11_h"] == "keine Angabe", "abr11_h"] = np.nan
se_red_17.loc[se_red_17["abr11_h"] == "Interviewabbruch", "abr11_h"] = np.nan

### Code binary: "1" if going abroad is part of study programme
se_red_17.loc[se_red_17["abr11_h"] == "ja", "abr11_h"] = 1
se_red_17.loc[se_red_17["abr11_h"] == "nein", "abr11_h"] = 0
se_red_17["abr11_h"] = pd.to_numeric(se_red_17["abr11_h"], errors='coerce')
se_red_17.rename(columns={"abr11_h": "DUM_abr11_h"}, inplace=True)


## (weiterer) Auslandsaufenthalt beabsichtigt? (abr12_h)
se_red_17["abr12_h"].dtype
round(se_red_17["abr12_h"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs / rename values
se_red_17.loc[se_red_17["abr12_h"] == "keine Angabe", "abr12_h"] = np.nan
se_red_17.loc[se_red_17["abr12_h"] == "Interviewabbruch", "abr12_h"] = np.nan


## Hindernis Auslandsaufenthalt (abr13X)
## Filter/Split: Studierende, die noch nicht studienbezogen im Ausland waren
## und auch keinen Auslandsaufenthalt planen
### Declare NaNs / rename values

#### for all columns that start with abr13, show unique values
#### to identify values  which need to be replaced with np.nan
list = ["abr13a_v21", "abr13b_v21", "abr13c_v21", "abr13d_v21", "abr13e_v21", "abr13f_v21"]
for i in list:
    round(se_red_17[i].value_counts(normalize=True, dropna=False), 3)

round(se_red_17["abr13a_v21"].value_counts(normalize=True, dropna=False), 3)

#### create new loop to replace with np.nan
for i in list:
    se_red_17.loc[se_red_17[i] == "unbekannter fehlender Wert", i] = np.nan
    se_red_17.loc[se_red_17[i] == "keine Angabe", i] = np.nan
    se_red_17.loc[se_red_17[i] == "Interviewabbruch", i] = np.nan
    se_red_17.loc[se_red_17[i] == "weiß nicht", i] = np.nan
    se_red_17.loc[se_red_17[i] == "splitbedingt fehlend", i] = np.nan
    se_red_17.loc[se_red_17[i] == "filterbedingt fehlend", i] = np.nan
    se_red_17.loc[se_red_17[i] == "trifft völlig zu", i] = 5
    se_red_17.loc[se_red_17[i] == "Pos. 4", i] = 4
    se_red_17.loc[se_red_17[i] == "Pos. 3", i] = 3
    se_red_17.loc[se_red_17[i] == "Pos. 2", i] = 2
    se_red_17.loc[se_red_17[i] == "trifft gar nicht zu", i] = 1
    se_red_17[i] = pd.to_numeric(se_red_17[i], errors='coerce')


## AUSLANDSSTUDIUM als Unterform des 'studienbezogenen Auslandsaufenthalts' (abr02a_v21)
se_red_17["abr02a_v21"].dtype
round(se_red_17["abr02a_v21"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs / rename values
se_red_17.loc[se_red_17["abr02a_v21"] == "keine Angabe", "abr02a_v21"] = np.nan
se_red_17.loc[se_red_17["abr02a_v21"] == "Interviewabbruch", "abr02a_v21"] = np.nan
se_red_17.loc[se_red_17["abr02a_v21"] == "filterbedingt fehlend", "abr02a_v21"] = "kein studienbezogener Auslandsaufenthalt"

### Code binary: "1" if lived abroad during studies to study at other university
se_red_17.loc[se_red_17["abr02a_v21"] == "genannt", "abr02a_v21"] = 1
se_red_17.loc[se_red_17["abr02a_v21"] == "nicht genannt", "abr02a_v21"] = 0
se_red_17["abr02a_v21"] = pd.to_numeric(se_red_17["abr02a_v21"], errors='coerce')
se_red_17.rename(columns={"abr02a_v21": "DUM_abr02a_v21"}, inplace=True)


## Finanzquelle Auslandsstudium (abr08X)
### Declare NaNs / rename values

#### for all columns that start with abr08, show unique values
#### to identify values  which need to be replaced with np.nan
list = ["abr08a_v21", "abr08b_v21", "abr08c_v21", "abr08d_v21", "abr08e_v21", "abr08f_v21", "abr08g_v21", "abr08h_v21", "abr08i_v21"]
for i in list:
    round(se_red_17[i].value_counts(normalize=True, dropna=False), 3)

#### create new loop to replace with np.nan
for i in list:
    se_red_17.loc[se_red_17[i] == "unbekannter fehlender Wert", i] = np.nan
    se_red_17.loc[se_red_17[i] == "keine Angabe", i] = np.nan
    se_red_17.loc[se_red_17[i] == "Interviewabbruch", i] = np.nan
    se_red_17.loc[se_red_17[i] == "splitbedingt fehlend", i] = np.nan
    se_red_17.loc[se_red_17[i] == "filterbedingt fehlend", i] = np.nan
    se_red_17.loc[se_red_17[i] == "genannt", i] = 1
    se_red_17.loc[se_red_17[i] == "nicht genannt", i] = 0
    se_red_17[i] = pd.to_numeric(se_red_17[i], errors='coerce')

se_red_17.rename(columns={"abr08a_v21": "DUM_abr08a_v21", "abr08b_v21": "DUM_abr08b_v21", "abr08c_v21": "DUM_abr08c_v21", "abr08d_v21": "DUM_abr08d_v21", "abr08e_v21": "DUM_abr08e_v21", "abr08f_v21": "DUM_abr08f_v21", "abr08g_v21": "DUM_abr08g_v21", "abr08h_v21": "DUM_abr08h_v21", "abr08i_v21": "DUM_abr08i_v21", }, inplace=True)


# Auslandsstudium/-studien: Dauer in Monaten (abr03a_v21)
se_red_17["abr03a_v21"].dtype
round(se_red_17["abr03a_v21"].value_counts(normalize=True, dropna=False), 3)
se_red_17["abr03a_v21"] = pd.to_numeric(se_red_17["abr03a_v21"], errors='coerce')

### Declare NaNs / rename values
se_red_17.loc[se_red_17["abr03a_v21"] == "keine Angabe", "abr03a_v21"] = np.nan
se_red_17.loc[se_red_17["abr03a_v21"] == "unbekannter fehlender Wert", "abr03a_v21"] = np.nan
se_red_17.loc[se_red_17["abr03a_v21"] == "Interviewabbruch", "abr03a_v21"] = np.nan
se_red_17.loc[se_red_17["abr03a_v21"] == "splitbedingt fehlend", "abr03a_v21"] = np.nan
se_red_17["abr03a_v21"] = pd.to_numeric(se_red_17["abr03a_v21"], errors='coerce')


## AUSLANDSPRAKTIKUM als Unterform des 'studienbezogenen Auslandsaufenthalts' (abr02b_v21)
se_red_17["abr02b_v21"].dtype
round(se_red_17["abr02b_v21"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs / rename values
se_red_17.loc[se_red_17["abr02b_v21"] == "keine Angabe", "abr02b_v21"] = np.nan
se_red_17.loc[se_red_17["abr02b_v21"] == "filterbedingt fehlend", "abr02b_v21"] = "kein studienbezogener Auslandsaufenthalt"
se_red_17.loc[se_red_17["abr02b_v21"] == "nicht genannt", "abr02b_v21"] = "kein Praktikum im Ausland während studienbezogenem Auslandsaufenthalt"
se_red_17.loc[se_red_17["abr02b_v21"] == "Interviewabbruch", "abr02b_v21"] = np.nan

### Code binary: "1" if lived abroad during studies to do internship
se_red_17.loc[se_red_17["abr02b_v21"] == "genannt", "abr02b_v21"] = 1
se_red_17.loc[se_red_17["abr02b_v21"] == "nicht genannt", "abr02b_v21"] = 0
se_red_17["abr02b_v21"] = pd.to_numeric(se_red_17["abr02b_v21"], errors='coerce')
se_red_17.rename(columns={"abr02b_v21": "DUM_abr02b_v21"}, inplace=True)


## Finanzquelle Auslandspraktikum (abr09X)
### Declare NaNs / rename values

#### for all columns that start with abr08, show unique values
#### to identify values  which need to be replaced with np.nan
list = ["abr09a_v21", "abr09b_v21", "abr09c_v21", "abr09d_v21", "abr09e_v21", "abr09f_v21", "abr09g_v21", "abr09h_v21", "abr09i_v21"]
for i in list:
    round(se_red_17[i].value_counts(normalize=True, dropna=False), 3)

#### create new loop to replace with np.nan
for i in list:
    se_red_17.loc[se_red_17[i] == "unbekannter fehlender Wert", i] = np.nan
    se_red_17.loc[se_red_17[i] == "keine Angabe", i] = np.nan
    se_red_17.loc[se_red_17[i] == "Interviewabbruch", i] = np.nan
    se_red_17.loc[se_red_17[i] == "splitbedingt fehlend", i] = "kein Auslandspraktikum"
    se_red_17.loc[se_red_17[i] == "filterbedingt fehlend", i] = "kein Auslandspraktikum"
    se_red_17.loc[se_red_17[i] == "genannt", i] = 1
    se_red_17.loc[se_red_17[i] == "nicht genannt", i] = 0
    se_red_17[i] = pd.to_numeric(se_red_17[i], errors='coerce')

se_red_17.rename(columns={"abr09a_v21": "DUM_abr09a_v21", "abr09b_v21": "DUM_abr09b_v21", "abr09c_v21": "DUM_abr09c_v21", "abr09d_v21": "DUM_abr09d_v21", "abr09e_v21": "DUM_abr09e_v21", "abr09f_v21": "DUM_abr09f_v21", "abr09g_v21": "DUM_abr09g_v21", "abr09h_v21": "DUM_abr09h_v21", "abr09i_v21": "DUM_abr09i_v21", }, inplace=True)


# Auslandspraktikum/-praktika: Dauer in Monaten (abr03b_v21)
se_red_17["abr03b_v21"].dtype
round(se_red_17["abr03b_v21"].value_counts(normalize=True, dropna=False), 3)
se_red_17["abr03b_v21"] = pd.to_numeric(se_red_17["abr03b_v21"], errors='coerce')

### Declare NaNs / rename values
se_red_17.loc[se_red_17["abr03b_v21"] == "keine Angabe", "abr03b_v21"] = np.nan
se_red_17.loc[se_red_17["abr03b_v21"] == "unbekannter fehlender Wert", "abr03b_v21"] = np.nan
se_red_17.loc[se_red_17["abr03b_v21"] == "Interviewabbruch", "abr03b_v21"] = np.nan
se_red_17.loc[se_red_17["abr03b_v21"] == "splitbedingt fehlend", "abr03b_v21"] = np.nan
se_red_17["abr03b_v21"] = pd.to_numeric(se_red_17["abr03b_v21"], errors='coerce')


## SONSTIGE AUSLANDSAUFENTHALTE als Unterform des 'studienbezogenen Auslandsaufenthalts' (abr02d_v21)
se_red_17["abr02d_v21"].dtype
round(se_red_17["abr02d_v21"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs / rename values
se_red_17.loc[se_red_17["abr02d_v21"] == "keine Angabe", "abr02d_v21"] = np.nan
se_red_17.loc[se_red_17["abr02d_v21"] == "filterbedingt fehlend", "abr02d_v21"] = "kein studienbezogener Auslandsaufenthalt"
se_red_17.loc[se_red_17["abr02d_v21"] == "nicht genannt", "abr02d_v21"] = "keine sonstige Aktivitäten (e.g., Summerschool, Exkursion) im Ausland während studienbezogenem Auslandsaufenthalt"
se_red_17.loc[se_red_17["abr02d_v21"] == "Interviewabbruch", "abr02d_v21"] = np.nan

### Code binary: "1" if lived abroad during studies for reasons other than studying or internship (e.g. summer school, excursion)
se_red_17.loc[se_red_17["abr02d_v21"] == "genannt", "abr02d_v21"] = 1
se_red_17.loc[se_red_17["abr02d_v21"] == "nicht genannt", "abr02d_v21"] = 0
se_red_17["abr02d_v21"] = pd.to_numeric(se_red_17["abr02d_v21"], errors='coerce')
se_red_17.rename(columns={"abr02d_v21": "DUM_abr02d_v21"}, inplace=True)


# Sonstige Auslandsaufenthalte: Dauer in Monaten (abr03d_v21)
se_red_17["abr03d_v21"].dtype
round(se_red_17["abr03d_v21"].value_counts(normalize=True, dropna=False), 3)
se_red_17["abr03d_v21"] = pd.to_numeric(se_red_17["abr03d_v21"], errors='coerce')

### Declare NaNs / rename values
se_red_17.loc[se_red_17["abr03d_v21"] == "keine Angabe", "abr03d_v21"] = np.nan
se_red_17.loc[se_red_17["abr03d_v21"] == "unbekannter fehlender Wert", "abr03d_v21"] = np.nan
se_red_17.loc[se_red_17["abr03d_v21"] == "Interviewabbruch", "abr03d_v21"] = np.nan
se_red_17.loc[se_red_17["abr03d_v21"] == "splitbedingt fehlend", "abr03d_v21"] = np.nan
se_red_17["abr03d_v21"] = pd.to_numeric(se_red_17["abr03d_v21"], errors='coerce')


## Sprachkenntnisse: Englisch (lan01_v21)
se_red_17["lan01_v21"].dtype
round(se_red_17["lan01_v21"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs / rename values
se_red_17.loc[se_red_17["lan01_v21"] == "keine Angabe", "lan01_v21"] = np.nan
se_red_17.loc[se_red_17["lan01_v21"] == "trifft nicht zu", "lan01_v21"] = "keine"
se_red_17.loc[se_red_17["lan01_v21"] == "Interviewabbruch", "lan01_v21"] = np.nan
se_red_17.loc[se_red_17["lan01_v21"] == "sehr gute Kenntnisse", "lan01_v21"] = 5
se_red_17.loc[se_red_17["lan01_v21"] == "Pos. 4", "lan01_v21"] = 4
se_red_17.loc[se_red_17["lan01_v21"] == "Pos. 3", "lan01_v21"] = 3
se_red_17.loc[se_red_17["lan01_v21"] == "Pos. 2", "lan01_v21"] = 2
se_red_17.loc[se_red_17["lan01_v21"] == "Grundkenntnisse", "lan01_v21"] = 1
se_red_17.loc[se_red_17["lan01_v21"] == "trifft nicht zu", "lan01_v21"] = 0
se_red_17["lan01_v21"] = pd.to_numeric(se_red_17["lan01_v21"], errors='coerce')


## Wohnform (liv01_v21)
se_red_17["liv01_v21"].dtype
round(se_red_17["liv01_v21"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs / rename values
se_red_17.loc[se_red_17["liv01_v21"] == "nicht bestimmbar", "liv01_v21"] = np.nan
se_red_17.loc[se_red_17["liv01_v21"] == "trifft nicht zu", "liv01_v21"] = "keine"
se_red_17.loc[se_red_17["liv01_v21"] == "Interviewabbruch", "liv01_v21"] = np.nan


## Beratungsbedarf ("Schwierigkeiten", "Belastungen") (adv01X)
### Declare NaNs / rename values

#### for all columns that start with adv01, show unique values
#### to identify values  which need to be replaced with np.nan
list = ["adv01a_v21", "adv01b_v21", "adv01c_v21", "adv01d_v21", "adv01e_v21", "adv01g_v21", "adv01i_v21", "adv01j_v21", "adv01k_v21", "adv01l_v21", "adv01m_v21", "adv01n_v21", "adv01o_v21", "adv01p_v21", "adv01k_v21", "adv02_v21"]
for i in list:
    round(se_red_17[i].value_counts(normalize=True, dropna=False), 3)

#### create new loop to replace with np.nan
for i in list:
    se_red_17.loc[se_red_17[i] == "unbekannter fehlender Wert", i] = np.nan
    se_red_17.loc[se_red_17[i] == "keine Angabe", i] = np.nan
    se_red_17.loc[se_red_17[i] == "Interviewabbruch", i] = np.nan
    se_red_17.loc[se_red_17[i] == "splitbedingt fehlend", i] = np.nan
    se_red_17.loc[se_red_17[i] == "genannt", i] = 1
    se_red_17.loc[se_red_17[i] == "nicht genannt", i] = 0
    se_red_17[i] = pd.to_numeric(se_red_17[i], errors='coerce')

se_red_17.rename(columns={"adv01a_v21": "DUM_adv01a_v21", "adv01b_v21": "DUM_adv01b_v21", "adv01c_v21": "DUM_adv01c_v21", "adv01d_v21": "DUM_adv01d_v21", "adv01e_v21": "DUM_adv01e_v21", "adv01g_v21": "DUM_adv01g_v21", "adv01i_v21": "DUM_adv01i_v21", "adv01j_v21": "DUM_adv01j_v21", "adv01k_v21": "DUM_adv01k_v21", "adv01l_v21": "DUM_adv01l_v21", "adv01m_v21": "DUM_adv01m_v21", "adv01n_v21": "DUM_adv01n_v21", "adv01o_v21": "DUM_adv01o_v21", "adv01p_v21": "DUM_adv01p_v21", "adv01k_v21": "DUM_adv01k_v21", "adv02_v21": "DUM_adv02_v21"}, inplace=True)



# CREATE FEATURES
## "First Generation Student" (dem99_c): infer from par05_h and par06_h


## create new column and set all cells to "1" to denote First Generation Student
se_red_17.insert(8, "dem99_c", 1)


## if either father or mother has "akademischer Abschluss": set cell in dem99_c in same row to "0" (non-FGS)
list = se_red_17[se_red_17["par05_h"]=="akademischer Abschluss"].index.values
for i in list:
    se_red_17.iloc[i, se_red_17.columns.get_loc("dem99_c")] = 0

list = se_red_17[se_red_17["par06_h"]=="akademischer Abschluss"].index.values
for i in list:
    se_red_17.iloc[i, se_red_17.columns.get_loc("dem99_c")] = 0

## if both mother and father columns are NaN: NaN
list = se_red_17[se_red_17["par05_h"].isnull()].index.tolist()
for i in list:
    if pd.isnull(se_red_17.iloc[i, se_red_17.columns.get_loc("par06_h")]):
        se_red_17["dem99_c"].iloc[i] = np.nan

list = se_red_17[se_red_17["par06_h"].isnull()].index.tolist()
for i in list:
    if pd.isnull(se_red_17.iloc[i, se_red_17.columns.get_loc("par05_h")]):
        se_red_17["dem99_c"].iloc[i] = np.nan

se_red_17["dem99_c"] = pd.to_numeric(se_red_17["dem99_c"], errors='coerce')

## produce summary statistics - plausible?
round(se_red_17["dem99_c"].value_counts(normalize=True, dropna=False), 3)


# Zus.fassen zu Wochenwert und Tagesspalten löschen: sonst Studienaufwand, Erwerbstätigkeit 
## Sonst Studienaufwand (tim02X)
se_red_17.insert(87, "tim02z_c", 0)
se_red_17["tim02z_c"] = se_red_17["tim02a_v21"] + se_red_17["tim02b_v21"] + se_red_17["tim02c_v21"] + se_red_17["tim02d_v21"] + se_red_17["tim02e_v21"] + se_red_17["tim02f_v21"] + se_red_17["tim02g_v21"]

## Erwerbstätigkeit (tim03X)
se_red_17.insert(95, "tim03z_c", 0)
se_red_17["tim03z_c"] = se_red_17["tim03a_v21"] + se_red_17["tim03b_v21"] + se_red_17["tim03c_v21"] + se_red_17["tim03d_v21"] + se_red_17["tim03e_v21"] + se_red_17["tim03f_v21"] + se_red_17["tim03g_v21"]

## drop weekday features
se_red_17.drop(["tim03a_v21", "tim03b_v21", "tim03c_v21", "tim03d_v21", "tim03e_v21", "tim03f_v21", "tim03g_v21"], axis=1, inplace=True)
se_red_17.drop(["tim02a_v21", "tim02b_v21", "tim02c_v21", "tim02d_v21", "tim02e_v21", "tim02f_v21", "tim02g_v21"], axis=1, inplace=True)
se_red_17.reset_index()


# Zus.fassen zu Summe und Einzelspalten löschen: Ausgaben 
## Ausgaben (EUR/Monat) (fin02X)
se_red_17.columns.get_loc("fin02a_h")
se_red_17.insert(46, "fin02z_c", 0)
se_red_17["fin02z_c"] = se_red_17["fin02a_h"] + se_red_17["fin02b_h"] + se_red_17["fin02c_h"] + se_red_17["fin02d_h"] + se_red_17["fin02e_h"] + se_red_17["fin02f_h"] + se_red_17["fin02g_h"] + se_red_17["fin02h_h"] + se_red_17["fin02i_h"]

## drop individual columns
se_red_17.drop(["fin02a_h", "fin02b_h", "fin02c_h", "fin02d_h", "fin02e_h", "fin02f_h", "fin02g_h", "fin02h_h", "fin02i_h", ], axis=1, inplace=True)
se_red_17.reset_index()


# Check right colum types? Nans where needed?
print(se_red_17.info(max_cols=145))
se_red_17_dum = se_red_17

## Dummy-code nominal categorical variables
##  - create new binary variable for each unique value
##  - assign numerical values to ordinal categorical variables (e.g., Likert scales), if necessary
##  - clean the original columns for which I now have dummy variables; reset dataframe index values

# create list with variables that require dummies, create loop iterating through list: name dummies after original variable, insert dummy clms,
# this will add new variables - dummy variables, one per category of the dummified categorical variable, and drop the original variable
list = ["dem01_h", "dem11a_h", "par03_v21", "par04_v21", "par05_h", "par06_h", "par07_h", "par08_h", "par09", "par10", "stu01a_h", "stu02_h", "stu03_h", "stu16b_h", "ped01_v21", "job03n_v21", "abr12_h", "liv01_v21"]
for i in list:
    pref = "DUM_" + i
    se_red_17_dum = pd.get_dummies(se_red_17_dum, prefix=pref, columns=[i], dummy_na=True)

se_red_17_dum.columns.values

# rearrange dataframe so that dummy columns are at position of original variables were, instead of end of dataframe
se_red_17_dum = se_red_17_dum.reindex(columns=["index", "id", "dem99_c", "DUM_dem01_h_männlich", "DUM_dem01_h_weiblich", "DUM_dem01_h_divers", "DUM_dem01_h_nan", "dem02_h", "DUM_dem03_v21", "DUM_dem05", "DUM_dem09", "DUM_dem11a_h_andere Staatsangehörigkeit", "DUM_dem11a_h_deutsche Staatsangehörigkeit", "DUM_dem11a_h_deutsche u. andere Staatsangeh.", "DUM_dem11a_h_nan", "DUM_par03_v21_Fachhochschulreife", "DUM_par03_v21_Haupt-/Volksschulabschluss", "DUM_par03_v21_Realschulabschluss, mittlere Reife", "DUM_par03_v21_allg./fachg. Hochschulreife", "DUM_par03_v21_anderer Schulabschluss", "DUM_par03_v21_keinen Schulabschluss", "DUM_par03_v21_nan", "DUM_par04_v21_Fachhochschulreife", "DUM_par04_v21_Haupt-/Volksschulabschluss", "DUM_par04_v21_Realschulabschluss, mittlere Reife", "DUM_par04_v21_allg./fachg. Hochschulreife", "DUM_par04_v21_anderer Schulabschluss", "DUM_par04_v21_keinen Schulabschluss", "DUM_par04_v21_nan", "DUM_par05_h_akademischer Abschluss", "DUM_par05_h_keinen beruflichen Abschluss", "DUM_par05_h_nicht-akademischer Berufsabschluss", "DUM_par05_h_nan", "DUM_par06_h_akademischer Abschluss", "DUM_par06_h_keinen beruflichen Abschluss", "DUM_par06_h_nicht-akademischer Berufsabschluss", "DUM_par06_h_nan", "DUM_par07_h_Angestellter", "DUM_par07_h_Arbeiter", "DUM_par07_h_Beamter", "DUM_par07_h_Selbständiger", "DUM_par07_h_nie berufstätig gewesen", "DUM_par07_h_nan", "DUM_par08_h_Angestellte", "DUM_par08_h_Arbeiterin", "DUM_par08_h_Beamtin", "DUM_par08_h_Selbständige", "DUM_par08_h_nie berufstätig gewesen", "DUM_par08_h_nan", "DUM_par09_andere Staatsangehörigkeit", "DUM_par09_deutsche Staatsangehörigkeit", "DUM_par09_deutsche u. andere Staatsangeh.", "DUM_par09_nan", "DUM_par10_andere Staatsangehörigkeit", "DUM_par10_deutsche Staatsangehörigkeit", "DUM_par10_deutsche u. andere Staatsangeh.", "DUM_par10_nan", "DUM_stu01a_h_Agrar-, Forst-, Ernährg.wiss.", "DUM_stu01a_h_Humanmedizin/Gesundheitswiss.", "DUM_stu01a_h_Ingenieurwiss.", "DUM_stu01a_h_Kunst, Kunstwiss.", "DUM_stu01a_h_Mathematik,  Naturwiss.", "DUM_stu01a_h_Rechts-, Wirtsch.-, Sozialwiss.", "DUM_stu01a_h_Sport", "DUM_stu01a_h_Sprach-, Kulturwiss.", "DUM_stu01a_h_andere", "DUM_stu01a_h_nan", "DUM_stu02_h_Bachelor", "DUM_stu02_h_Diplom", "DUM_stu02_h_Magister", "DUM_stu02_h_Master", "DUM_stu02_h_Staatsexamen", "DUM_stu02_h_anderer Abschluss", "DUM_stu02_h_keinen Abschluss", "DUM_stu02_h_nan", "DUM_stu03_h_Bachelor", "DUM_stu03_h_Diplom", "DUM_stu03_h_Magister", "DUM_stu03_h_Master", "DUM_stu03_h_Promotion", "DUM_stu03_h_Staatsexamen", "DUM_stu03_h_kein vorhandener Abschluss", "DUM_stu03_h_sonstiger Abschluss", "DUM_stu03_h_nan", "stu04", "stu05", "DUM_stu10_h", "DUM_stu11b", "DUM_stu11c_v20", "DUM_stu11d", "DUM_stu11e", "DUM_stu11f", "DUM_stu11g", "DUM_stu11h_v21", "stu12_v21", "DUM_stu13_h", "DUM_stu16a_h", "DUM_stu16b_h_Norddeutschland", "DUM_stu16b_h_Ostdeutschland", "DUM_ped01_v21_Fachhochschulreife", "DUM_ped01_v21_allg. Hochschulreife", "DUM_ped01_v21_andere Studienberechtigung", "DUM_ped01_v21_berufl. Qualifikation", "DUM_ped01_v21_fachg. Hochschulreife", "DUM_ped01_v21_nan", "DUM_ped03", "fin01a", "fin01b", "fin01c", "fin01d", "fin01e", "fin01f_h", "fin01g", "fin01h", "fin01j", "fin01o", "fin02a_h", "fin02b_h", "fin02c_h", "fin02d_h", "fin02e_h", "fin02f_h", "fin02g_h", "fin02h_h", "fin02i_h", "fin03a_h", "fin03b_h", "fin03c_h", "fin03d_h", "fin03e_h", "fin03f_h", "fin03g_h", "fin03h_h", "fin03i_h", "fin04a", "fin04b", "fin04c", "fin04d_v21", "fin04e_v21", "fin04f_v21", "fin04i", "DUM_baf01_h", "DUM_baf05a", "DUM_baf05b", "DUM_baf05c", "DUM_baf05e", "DUM_baf05f_v20", "DUM_baf05g", "DUM_baf05h_v21", "baf05i_v21", "tim02z_c", "tim03z_c", "DUM_job02_h", "DUM_job03a_v21", "DUM_job03b_v21", "DUM_job03g_v21", "DUM_job03n_v21_Jobben", "DUM_job03n_v21_Nachhilfeunterricht", "DUM_job03n_v21_Taetigkeit, die Hochschulabschluss voraussetzt", "DUM_job03n_v21_Taetigkeit, die beruflichen Ausbildungsabschluss voraussetzt", "DUM_job03n_v21_andere Tätigkeit", "DUM_job03n_v21_keine berufliche Tätigkeit", "DUM_job03n_v21_studentische/wissenschaftliche Hilfskraft", "DUM_job03n_v21_nan", "job05a", "job05b", "job05c", "job05d", "job05e", "job05f", "job05g", "job05h", "DUM_abr01_h", "DUM_abr02a_v21", "DUM_abr02b_v21", "DUM_abr02d_v21", "abr03a_v21", "abr03b_v21", "abr03d_v21", "DUM_abr08a_v21", "DUM_abr08b_v21", "DUM_abr08c_v21", "DUM_abr08d_v21", "DUM_abr08e_v21", "DUM_abr08f_v21", "DUM_abr08g_v21", "DUM_abr08h_v21", "DUM_abr08i_v21", "DUM_abr09a_v21", "DUM_abr09b_v21", "DUM_abr09c_v21", "DUM_abr09d_v21", "DUM_abr09e_v21", "DUM_abr09f_v21", "DUM_abr09g_v21", "DUM_abr09h_v21", "DUM_abr09i_v21", "DUM_abr11_h", "DUM_abr12_h_ja", "DUM_abr12_h_nein, kein Interesse", "DUM_abr12_h_nein, sehe keine Realisierungschance", "DUM_abr12_h_weiß ich noch nicht", "DUM_abr12_h_nan", "abr13a_v21", "abr13b_v21", "abr13c_v21", "abr13d_v21", "abr13e_v21", "abr13f_v21", "lan01_v21", "DUM_liv01_v21_Eltern, Verwandte", "DUM_liv01_v21_Untermiete", "DUM_liv01_v21_Wohngemeinschaft", "DUM_liv01_v21_Wohnh.: Einzelappartment", "DUM_liv01_v21_Wohnh.: Einzelzi. in Wohngr.", "DUM_liv01_v21_Wohnh.: Einzelzimmer", "DUM_liv01_v21_Wohnh.: Mehr-Zi.-Wohng.", "DUM_liv01_v21_Wohnh.: Zweibett-Zimmer", "DUM_liv01_v21_Wohnung allein", "DUM_liv01_v21_Wohnung m. Partn./Kind", "DUM_liv01_v21_nan", "DUM_adv01a_v21", "DUM_adv01b_v21", "DUM_adv01c_v21", "DUM_adv01d_v21", "DUM_adv01e_v21", "DUM_adv01g_v21", "DUM_adv01i_v21", "DUM_adv01j_v21", "DUM_adv01k_v21", "DUM_adv01l_v21", "DUM_adv01m_v21", "DUM_adv01n_v21", "DUM_adv01o_v21", "DUM_adv01p_v21", "DUM_adv02_v21"])



# ------- THIS IS WHERE ACTUAL ANALYSIS STARTS ------- #
## "getting to know" the data: revisit research question, then investigate each variable"s statistics and take notes
## RQ: How do first generation and non-first generation students' study conditions differ?

## correlation table to preselect variables correlated to dem99_c for visualization
# se_red_17_corr = round(se_red_17_dum.corr(), 3)
# se_red_17_corr.to_csv(r"C:\Users\marc.feldmann\Dropbox\Dissertation_MFeldmann\se_red_17_corr.csv")

## corr table with asterisks indicating sign:
## rho = se_red_17.corr()
## pval = se_red_17.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*rho.shape)
## p = pval.applymap(lambda x: ''.join(['*' for t in [0.01,0.05,0.1] if x<=t]))
## se_red_17_corr = rho.round(2).astype(str) + p


## PLOTTING
### have preselected variables for plotting based on correlations with dem99_c, my main variable of interest (see se_red_17_corr_ann.csv)
### see notes
### create one dataframe per group FGS/non-FGS
se_red_17_FGS = se_red_17[se_red_17["dem99_c"] == 1]
se_red_17_nFGS = se_red_17[se_red_17["dem99_c"] == 0]


# Fig_SB_ped01: "Art der Studienberechtigung"
# data_temp = se_red_17[se_red_17["par04_v21"] != "anderer Schulabschluss"]
data_temp = se_red_17
pct2 = (data_temp.groupby(["ped01_v21", "dem99_c"]).size() / data_temp[data_temp["ped01_v21"].notna()].groupby(["dem99_c"]).size()).reset_index().rename({0:"percent"}, axis=1)
hue_labels = ["first generation students", "non first generation students"]
p = sns.barplot(data=pct2, x="ped01_v21", y="percent", hue="dem99_c", hue_order=[1, 0], palette=["black", "grey"])
# p.set_xticklabels(["clerk", "worker", "civil servant",  "self-employed"])
p.set_yticklabels(["", "20", "40", "60", "80", "100"])
p.set_xlabel("Type of higher education qualification")
plt.ylim(0, 1)
p.set_ylabel("% of respondents")
h, l = p.get_legend_handles_labels()
p.legend(h, hue_labels, title="", loc="upper right")
plt.show()


# Fig_SC_stu16a_h: "Art der Hochschule"
data_temp = se_red_17
pct2 = (data_temp.groupby(["DUM_stu16a_h", "dem99_c"]).size() / data_temp[data_temp["DUM_stu16a_h"].notna()].groupby(["dem99_c"]).size()).reset_index().rename({0:"percent"}, axis=1)
hue_labels = ["first generation students", "non first generation students"]
p = sns.barplot(data=pct2, x="DUM_stu16a_h", y="percent", order=[1, 0], hue="dem99_c", hue_order=[1, 0], palette=["black", "grey"])
p.set_xticklabels(["university", "university of applied sciences"])
p.set_yticklabels(["", "20", "40", "60", "80", "100"])
p.set_xlabel("Type of higher education instutition")
plt.ylim(0, 1)
p.set_ylabel("% of respondents")
h, l = p.get_legend_handles_labels()
p.legend(h, hue_labels, title="", loc="upper right")
plt.show()


# Fig_SB_ped03: "Berufsausbildung vor Studium"
data_temp = se_red_17
pct2 = (data_temp.groupby(["DUM_ped03", "dem99_c"]).size() / data_temp[data_temp["DUM_ped03"].notna()].groupby(["dem99_c"]).size()).reset_index().rename({0:"percent"}, axis=1)
hue_labels = ["first generation students", "non first generation students"]
p = sns.barplot(data=pct2, x="DUM_ped03", y="percent", hue="dem99_c", order=[1, 0], hue_order=[1, 0], palette=["black", "grey"])
p.set_xticklabels(["yes", "no"])
p.set_yticklabels(["", "20", "40", "60", "80", "100"])
p.set_xlabel("Vocational training completed prior to studies")
plt.ylim(0, 1)
p.set_ylabel("% of respondents")
h, l = p.get_legend_handles_labels()
p.legend(h, hue_labels, title="", loc="upper right")
plt.show()


# Fig_SC_liv01_v21: "Wohnsituation"
data_temp = se_red_17
pct2 = (data_temp.groupby(["liv01_v21", "dem99_c"]).size() / data_temp[data_temp["liv01_v21"].notna()].groupby(["dem99_c"]).size()).reset_index().rename({0:"percent"}, axis=1)
hue_labels = ["first generation students", "non first generation students"]
p = sns.barplot(data=pct2, x="liv01_v21", y="percent", hue="dem99_c", hue_order=[1, 0], palette=["black", "grey"])
# p.set_xticklabels(["clerk", "worker", "civil servant",  "self-employed"])
p.set_yticklabels(["", "20", "40", "60", "80", "100"])
p.set_xlabel("Accommodation")
plt.ylim(0, 1)
p.set_ylabel("% of respondents")
h, l = p.get_legend_handles_labels()
p.legend(h, hue_labels, title="", loc="upper right")
plt.show()


# Fig_SA_abr01_h: "Studienbezogen im Ausland aufgehalten?"
data_temp = se_red_17
pct2 = (data_temp.groupby(["DUM_abr01_h", "dem99_c"]).size() / data_temp[data_temp["DUM_abr01_h"].notna()].groupby(["dem99_c"]).size()).reset_index().rename({0:"percent"}, axis=1)
hue_labels = ["first generation students", "non first generation students"]
p = sns.barplot(data=pct2, x="DUM_abr01_h", y="percent", hue="dem99_c", hue_order=[1, 0], order=[1, 0], palette=["black", "grey"])
p.set_xticklabels(["yes", "no"])
p.set_yticklabels(["", "20", "40", "60", "80", "100"])
p.set_xlabel("Spent time abroad during study programme?")
plt.ylim(0, 1)
p.set_ylabel("% of respondents")
h, l = p.get_legend_handles_labels()
p.legend(h, hue_labels, title="", loc="upper right")
plt.show()


# Fig_SA_abr12_h: "(Weiterer) Auslandsaufenthalt beabsichtigt?"
# data_temp = se_red_17[se_red_17["par04_v21"] != "anderer Schulabschluss"]
data_temp = se_red_17
pct2 = (data_temp.groupby(["abr12_h", "dem99_c"]).size() / data_temp[data_temp["abr12_h"].notna()].groupby(["dem99_c"]).size()).reset_index().rename({0:"percent"}, axis=1)
hue_labels = ["first generation students", "non first generation students"]
p = sns.barplot(data=pct2, x="abr12_h", y="percent", hue="dem99_c", hue_order=[1, 0], palette=["black", "grey"])
# p.set_xticklabels(["clerk", "worker", "civil servant",  "self-employed"])
p.set_yticklabels(["", "20", "40", "60", "80", "100"])
p.set_xlabel("(Another) stay abroad intended?")
plt.ylim(0, 1)
p.set_ylabel("% of respondents")
h, l = p.get_legend_handles_labels()
p.legend(h, hue_labels, title="", loc="upper right")
plt.show()


# Fig_SA_lan01_v21: Englischkenntnisse
data_temp = se_red_17
pct2 = (data_temp.groupby(["lan01_v21", "dem99_c"]).size() / data_temp[data_temp["lan01_v21"].notna()].groupby(["dem99_c"]).size()).reset_index().rename({0:"percent"}, axis=1)
hue_labels = ["first generation students", "non first generation students"]
p = sns.barplot(data=pct2, x="lan01_v21", y="percent", hue="dem99_c", hue_order=[1, 0], palette=["black", "grey"])
plt.ylim(0, 1)
p.set_yticklabels(["", "20", "40", "60", "80", "100"])
p.set_ylabel("% of respondents")
# plt.xlim(0, 5)
# p.set_xticks([1.0, 2.0, 3.0, 4.0, 5.0])
# p.set_xticklabels([1, 2, 3, 4, 5])
p.set_xlabel('English skills')
h, l = p.get_legend_handles_labels()
p.legend(h, hue_labels, title="", loc="upper right")
plt.show()


# Fig_SA_abr13a_v21: "Hindernis Auslandsaufenthalt: Sprachkenntnisse"
data_temp = se_red_17
pct2 = (data_temp.groupby(["abr13a_v21", "dem99_c"]).size() / data_temp[data_temp["abr13a_v21"].notna()].groupby(["dem99_c"]).size()).reset_index().rename({0:"percent"}, axis=1)
hue_labels = ["first generation students", "non first generation students"]
p = sns.barplot(data=pct2, x="abr13a_v21", y="percent", hue="dem99_c", hue_order=[1, 0], palette=["black", "grey"])
plt.ylim(0, 1)
p.set_yticklabels(["", "20", "40", "60", "80", "100"])
p.set_ylabel("% of respondents")
# plt.xlim(0, 5)
# p.set_xticks([1.0, 2.0, 3.0, 4.0, 5.0])
# p.set_xticklabels([1, 2, 3, 4, 5])
p.set_xlabel('"Language skills keep me from going abroad"')
h, l = p.get_legend_handles_labels()
p.legend(h, hue_labels, title="", loc="upper right")
plt.show()


# Fig_SA_abr13b_v21: "Hindernis Auslandsaufenthalt: fehlende Programminfos"
data_temp = se_red_17
pct2 = (data_temp.groupby(["abr13b_v21", "dem99_c"]).size() / data_temp[data_temp["abr13b_v21"].notna()].groupby(["dem99_c"]).size()).reset_index().rename({0:"percent"}, axis=1)
hue_labels = ["first generation students", "non first generation students"]
p = sns.barplot(data=pct2, x="abr13b_v21", y="percent", hue="dem99_c", hue_order=[1, 0], palette=["black", "grey"])
plt.ylim(0, 1)
p.set_yticklabels(["", "20", "40", "60", "80", "100"])
p.set_ylabel("% of respondents")
# plt.xlim(0, 5)
# p.set_xticks([1.0, 2.0, 3.0, 4.0, 5.0])
# p.set_xticklabels([1, 2, 3, 4, 5])
p.set_xlabel('"Lack of information keep me from going abroad"')
h, l = p.get_legend_handles_labels()
p.legend(h, hue_labels, title="", loc="upper right")
plt.show()


# Fig_SA_abr13c_v21: "Hindernis Auslandsaufenthalt: Wohnprobleme im Gastland"
data_temp = se_red_17
pct2 = (data_temp.groupby(["abr13c_v21", "dem99_c"]).size() / data_temp[data_temp["abr13c_v21"].notna()].groupby(["dem99_c"]).size()).reset_index().rename({0:"percent"}, axis=1)
hue_labels = ["first generation students", "non first generation students"]
p = sns.barplot(data=pct2, x="abr13c_v21", y="percent", hue="dem99_c", hue_order=[1, 0], palette=["black", "grey"])
plt.ylim(0, 1)
p.set_yticklabels(["", "20", "40", "60", "80", "100"])
p.set_ylabel("% of respondents")
# plt.xlim(0, 5)
# p.set_xticks([1.0, 2.0, 3.0, 4.0, 5.0])
# p.set_xticklabels([1, 2, 3, 4, 5])
p.set_xlabel('"Accommodation problems keep me from going abroad"')
h, l = p.get_legend_handles_labels()
p.legend(h, hue_labels, title="", loc="upper right")
plt.show()


# Fig_SA_abr13d_v21: "Hindernis Auslandsaufenthalt: Trennung von Partner(in)"
data_temp = se_red_17
pct2 = (data_temp.groupby(["abr13d_v21", "dem99_c"]).size() / data_temp[data_temp["abr13d_v21"].notna()].groupby(["dem99_c"]).size()).reset_index().rename({0:"percent"}, axis=1)
hue_labels = ["first generation students", "non first generation students"]
p = sns.barplot(data=pct2, x="abr13d_v21", y="percent", hue="dem99_c", hue_order=[1, 0], palette=["black", "grey"])
plt.ylim(0, 1)
p.set_yticklabels(["", "20", "40", "60", "80", "100"])
p.set_ylabel("% of respondents")
# plt.xlim(0, 5)
# p.set_xticks([1.0, 2.0, 3.0, 4.0, 5.0])
# p.set_xticklabels([1, 2, 3, 4, 5])
p.set_xlabel('"Being separated from my partner keeps me from going abroad"')
h, l = p.get_legend_handles_labels()
p.legend(h, hue_labels, title="", loc="upper right")
plt.show()


# Fig_SA_abr13e_v21: "Hindernis Auslandsaufenthalt: Wegfall von Leistungen"
data_temp = se_red_17
pct2 = (data_temp.groupby(["abr13e_v21", "dem99_c"]).size() / data_temp[data_temp["abr13e_v21"].notna()].groupby(["dem99_c"]).size()).reset_index().rename({0:"percent"}, axis=1)
hue_labels = ["first generation students", "non first generation students"]
p = sns.barplot(data=pct2, x="abr13e_v21", y="percent", hue="dem99_c", hue_order=[1, 0], palette=["black", "grey"])
plt.ylim(0, 1)
p.set_yticklabels(["", "20", "40", "60", "80", "100"])
p.set_ylabel("% of respondents")
# plt.xlim(0, 5)
# p.set_xticks([1.0, 2.0, 3.0, 4.0, 5.0])
# p.set_xticklabels([1, 2, 3, 4, 5])
p.set_xlabel('"Loss of financial support keeps me from going abroad"')
h, l = p.get_legend_handles_labels()
p.legend(h, hue_labels, title="", loc="upper right")
plt.show()


# Fig_SA_abr13f_v21: "Hindernis Auslandsaufenthalt: finanzielle Mehrbelastung"
data_temp = se_red_17
pct2 = (data_temp.groupby(["abr13f_v21", "dem99_c"]).size() / data_temp[data_temp["abr13f_v21"].notna()].groupby(["dem99_c"]).size()).reset_index().rename({0:"percent"}, axis=1)
hue_labels = ["first generation students", "non first generation students"]
p = sns.barplot(data=pct2, x="abr13f_v21", y="percent", hue="dem99_c", hue_order=[1, 0], palette=["black", "grey"])
plt.ylim(0, 1)
p.set_yticklabels(["", "20", "40", "60", "80", "100"])
p.set_ylabel("% of respondents")
# plt.xlim(0, 5)
# p.set_xticks([1.0, 2.0, 3.0, 4.0, 5.0])
# p.set_xticklabels([1, 2, 3, 4, 5])
p.set_xlabel('"Additional costs keep me from going abroad"')
h, l = p.get_legend_handles_labels()
p.legend(h, hue_labels, title="", loc="upper right")
plt.show()


# Fig_FS_adv01a: "Beratungsbedarf: Studienfinanzierung"
data_temp = se_red_17
pct2 = (data_temp.groupby(["DUM_adv01a_v21", "dem99_c"]).size() / data_temp[data_temp["DUM_adv01a_v21"].notna()].groupby(["dem99_c"]).size()).reset_index().rename({0:"percent"}, axis=1)
hue_labels = ["first generation students", "non first generation students"]
p = sns.barplot(data=pct2, x="DUM_adv01a_v21", y="percent", hue="dem99_c", hue_order=[1, 0], order=[1, 0], palette=["black", "grey"])
p.set_xticklabels(["yes", "no"])
p.set_yticklabels(["", "20", "40", "60", "80", "100"])
p.set_xlabel("Counselling need: financing studies")
plt.ylim(0, 1)
p.set_ylabel("% of respondents")
h, l = p.get_legend_handles_labels()
p.legend(h, hue_labels, title="", loc="upper right")
plt.show()


# Fig_FS_adv01b: "Beratungsbedarf: Auslandsaufenthalt Finanzierung"
data_temp = se_red_17
pct2 = (data_temp.groupby(["DUM_adv01b_v21", "dem99_c"]).size() / data_temp[data_temp["DUM_adv01b_v21"].notna()].groupby(["dem99_c"]).size()).reset_index().rename({0:"percent"}, axis=1)
hue_labels = ["first generation students", "non first generation students"]
p = sns.barplot(data=pct2, x="DUM_adv01b_v21", y="percent", hue="dem99_c", hue_order=[1, 0], order=[1, 0], palette=["black", "grey"])
p.set_xticklabels(["yes", "no"])
p.set_yticklabels(["", "20", "40", "60", "80", "100"])
p.set_xlabel("Counselling need: financing stay abroad")
plt.ylim(0, 1)
p.set_ylabel("% of respondents")
h, l = p.get_legend_handles_labels()
p.legend(h, hue_labels, title="", loc="upper right")
plt.show()


# Fig_FS_fin01a: "Barmittel Eltern"
data_temp = se_red_17[se_red_17["fin01a"] != 0]
p = sns.boxplot(x=data_temp["dem99_c"], y=data_temp["fin01a"], data=data_temp, showfliers=False, palette=["white", "grey"], order=[1, 0])
p.set_xlabel("")
p.set_ylabel("income from parent funding (EUR/month)")
p.set_xticklabels(["first generation students", "other students"])
p.set(ylim=(0,1250))
plt.show()


# Fig_FS_baf01_h: "BAFoeG Förderung im aktuellen Semester"
data_temp = se_red_17
pct2 = (data_temp.groupby(["DUM_baf01_h", "dem99_c"]).size() / data_temp[data_temp["DUM_baf01_h"].notna()].groupby(["dem99_c"]).size()).reset_index().rename({0:"percent"}, axis=1)
hue_labels = ["first generation students", "non first generation students"]
p = sns.barplot(data=pct2, x="DUM_baf01_h", y="percent", hue="dem99_c", hue_order=[1, 0], order=[1, 0], palette=["black", "grey"])
p.set_xticklabels(["yes", "no"])
p.set_yticklabels(["", "20", "40", "60", "80", "100"])
p.set_xlabel("Receiving federal education support in current semester")
plt.ylim(0, 1)
p.set_ylabel("% of respondents")
h, l = p.get_legend_handles_labels()
p.legend(h, hue_labels, title="", loc="upper right")
plt.show()


# Fig_FS_baf05a: "Keine BAFoeG Förderung da Einkommen Eltern/Partner zu hoch"
data_temp = se_red_17
pct2 = (data_temp.groupby(["DUM_baf05a", "dem99_c"]).size() / data_temp[data_temp["DUM_baf05a"].notna()].groupby(["dem99_c"]).size()).reset_index().rename({0:"percent"}, axis=1)
hue_labels = ["first generation students", "non first generation students"]
p = sns.barplot(data=pct2, x="DUM_baf05a", y="percent", hue="dem99_c", hue_order=[1, 0], order=[1, 0], palette=["black", "grey"])
p.set_xticklabels(["yes", "no"])
p.set_yticklabels(["", "20", "40", "60", "80", "100"])
p.set_xlabel("No federal education support due to parent/partner income")
plt.ylim(0, 1)
p.set_ylabel("% of respondents")
h, l = p.get_legend_handles_labels()
p.legend(h, hue_labels, title="", loc="upper right")
plt.show()


# Fig_FS_fin01d: "Barmittel BAFoeG"
data_temp = se_red_17[se_red_17["fin01d"] != 0]
p = sns.boxplot(x=data_temp["dem99_c"], y=data_temp["fin01d"], data=data_temp, showfliers=False, palette=["white", "grey"], order=[1, 0])
p.set_xlabel("")
p.set_ylabel("income from federal education support (EUR/month)")
p.set_xticklabels(["first generation students", "other students"])
p.set(ylim=(0,1250))
plt.show()


# Fig_FS_fin01j: "Barmittel Stipendium"
# Weil es hier mit relativ wenigen Ausnahmen einen uniformen Regelbetrag gibt, scheint Visualisierung von Stipendium "ja" / "nein" aussagekräftiger
data_temp = se_red_17[se_red_17["fin01j"] != 0]
data_temp_FGS = data_temp[data_temp["dem99_c"] == 1]
data_temp_FGS["fin01j"].mean()
data_temp_nFGS = data_temp[data_temp["dem99_c"] == 0]
data_temp_nFGS["fin01j"].mean()

# For plotting relative frequencies of those who get scholarship funding at all:
# data_temp: alle in spalte fin01j größer 0: ersetze mit 1
# data_temp = se_red_17
# data_temp.loc[data_temp["fin01j"] > 0, "fin01j"] = 1
# pct2 = (data_temp.groupby(["fin01j", "dem99_c"]).size() / data_temp[data_temp["fin01j"].notna()].groupby(["dem99_c"]).size()).reset_index().rename({0:"percent"}, axis=1)
# # hue_labels = ["first generation students", "non first generation students"]
# p = sns.barplot(data=pct2, x="fin01j", y="percent", hue="dem99_c", hue_order=[1, 0], palette=["black", "grey"])
# # p.set_xticklabels(["clerk", "worker", "civil servant",  "self-employed"])
# # p.set_xlabel("Mother's Education")
# p.set_yticklabels(["", "20", "40", "60", "80", "100"])
# plt.ylim(0, 1)
# p.set_ylabel("% of respondents")
# h, l = p.get_legend_handles_labels()
# p.legend(h, hue_labels, title="", loc="upper right")
# plt.show()

data_temp = se_red_17[se_red_17["fin01j"] != 0]
# data_temp_FGS = data_temp[data_temp["dem99_c"] == 1]
# data_temp_nFGS = data_temp[data_temp["dem99_c"] == 0]
p = sns.histplot(data=data_temp, x="fin01j", bins=100, hue_order=[1, 0], hue="dem99_c", kde=True, multiple="dodge", shrink=0.8, palette=["black", "grey"])
# the non-logarithmic labels you want
# p.set_yscale("log")
# p.set_yticks([1, 10, 100, 1000, 10000])
# # p.set_yticklabels([1, 10, 100, 1000, 10000])
# p.set_xticklabels(["civil servant", "clerk", "worker", "self-employed"])
# p.set_xlabel("Father's Occupation")
# p.set_ylabel("Counts")
p.set(xlim=(0,4000))
p.set(ylim=(0,100))
plt.legend(title="", loc="upper right", labels=["first generation students", "non first generation students"])
plt.show()

# # data_temp = se_red_17[se_red_17["fin01j"] != 0]
# # p = sns.boxplot(x=data_temp["dem99_c"], y=data_temp["fin01j"], data=data_temp, showfliers=False, palette=["white", "grey"], order=[1, 0])
# # p.set_xlabel("")
# # p.set_ylabel("scholarship funding (EUR/month)")
# # p.set_xticklabels(["first generation students", "other students"])
# # p.set(ylim=(0,800))
# # plt.show()


# Fig_FS_fin01o: "Barmittel eigenes Kindergeld"
data_temp = se_red_17[se_red_17["fin01o"] != 0]
p = sns.boxplot(x=data_temp["dem99_c"], y=data_temp["fin01o"], data=data_temp, showfliers=False, palette=["white", "grey"], order=[1, 0])
p.set_xlabel("")
p.set_ylabel("income from own child benefits (EUR/month)")
p.set_xticklabels(["first generation students", "other students"])
p.set(ylim=(150,250))
plt.show()


# Fig_FS_fin02z_c: "Ausgaben (EUR/Monat)"
data_temp = se_red_17[se_red_17["fin02z_c"] != 0]
p = sns.boxplot(x=data_temp["dem99_c"], y=data_temp["fin02z_c"], data=data_temp, showfliers=False, palette=["white", "grey"], order=[1, 0])
p.set_xlabel("")
p.set_ylabel("living costs (EUR/month)")
p.set_xticklabels(["first generation students", "other students"])
p.set(ylim=(0,1500))
plt.show()


# Fig_SB_par03: "höchster Schulabschluss - Vater"
# data_temp = se_red_17[se_red_17["par03_v21"] != "anderer Schulabschluss"]
data_temp = se_red_17
pct2 = (data_temp.groupby(["par03_v21", "dem99_c"]).size() / data_temp[data_temp["par03_v21"].notna()].groupby(["dem99_c"]).size()).reset_index().rename({0:"percent"}, axis=1)
hue_labels = ["first generation students", "non first generation students"]
p = sns.barplot(data=pct2, x="par03_v21", y="percent", hue="dem99_c", hue_order=[1, 0], palette=["black", "grey"])
# p.set_xticklabels(["clerk", "worker", "civil servant",  "self-employed"])
p.set_yticklabels(["", "20", "40", "60", "80", "100"])
p.set_xlabel("Father's Education")
plt.ylim(0, 1)
p.set_ylabel("% of respondents")
h, l = p.get_legend_handles_labels()
p.legend(h, hue_labels, title="", loc="upper right")
plt.show()


# Fig_SB_par04: "höchster Schulabschluss - Mutter"
# data_temp = se_red_17[se_red_17["par04_v21"] != "anderer Schulabschluss"]
data_temp = se_red_17
pct2 = (data_temp.groupby(["par04_v21", "dem99_c"]).size() / data_temp[data_temp["par04_v21"].notna()].groupby(["dem99_c"]).size()).reset_index().rename({0:"percent"}, axis=1)
hue_labels = ["first generation students", "non first generation students"]
p = sns.barplot(data=pct2, x="par04_v21", y="percent", hue="dem99_c", hue_order=[1, 0], palette=["black", "grey"])
# p.set_xticklabels(["clerk", "worker", "civil servant",  "self-employed"])
p.set_yticklabels(["", "20", "40", "60", "80", "100"])
p.set_xlabel("Mother's Education")
plt.ylim(0, 1)
p.set_ylabel("% of respondents")
h, l = p.get_legend_handles_labels()
p.legend(h, hue_labels, title="", loc="upper right")
plt.show()


# Fig_SB_par07_h: "berufliche Position - Vater"
data_temp = se_red_17[se_red_17["par07_h"] != "nie berufstätig gewesen"]
pct2 = (data_temp.groupby(["par07_h", "dem99_c"]).size() / data_temp[data_temp["par07_h"].notna()].groupby(["dem99_c"]).size()).reset_index().rename({0:"percent"}, axis=1)
hue_labels = ["first generation students", "non first generation students"]
p = sns.barplot(data=pct2, x="par07_h", y="percent", hue="dem99_c", hue_order=[1, 0], palette=["black", "grey"])
p.set_xticklabels(["clerk", "worker", "civil servant",  "self-employed"])
p.set_yticklabels(["", "20", "40", "60", "80", "100"])
p.set_xlabel("Father's Occupation")
plt.ylim(0, 1)
p.set_ylabel("% of respondents")
h, l = p.get_legend_handles_labels()
p.legend(h, hue_labels, title="", loc="upper right")
plt.show()


# Fig_SB_par08_h: "berufliche Position - Mutter"
# wichtig: darauf hinweisen, dass hier nur die studis einfließen, für die beide Werte vorliegen (Anzahl angeben)
data_temp = se_red_17[se_red_17["par08_h"] != "nie berufstätig gewesen"]
pct2 = (data_temp.groupby(["par08_h", "dem99_c"]).size() / data_temp[data_temp["par08_h"].notna()].groupby(["dem99_c"]).size()).reset_index().rename({0:"percent"}, axis=1)
hue_labels = ["first generation students", "non first generation students"]
p = sns.barplot(data=pct2, x="par08_h", y="percent", hue="dem99_c", hue_order=[1, 0], palette=["black", "grey"])
p.set_xticklabels(["clerk", "worker", "civil servant",  "self-employed"])
p.set_yticklabels(["", "20", "40", "60", "80", "100"])
p.set_xlabel("Mother's Occupation")
plt.ylim(0, 1)
p.set_ylabel("% of respondents")
h, l = p.get_legend_handles_labels()
p.legend(h, hue_labels, title="", loc="upper right")
plt.show()


# Fig_FS_abr08a_v21: "Finanzquelle Auslandsstudium Eltern/Partner"
# gerade HIER bspw. wichtig, die Gesamtanzahl zu erwähnen: nur die mit studienbez. Auslandsaufenthalt wurden befragt! die anderen sind Nan und werden von den funktionen ignoriert!
data_temp = se_red_17
pct2 = (data_temp.groupby(["DUM_abr08a_v21", "dem99_c"]).size() / data_temp[data_temp["DUM_abr08a_v21"].notna()].groupby(["dem99_c"]).size()).reset_index().rename({0:"percent"}, axis=1)
hue_labels = ["first generation students", "non first generation students"]
p = sns.barplot(data=pct2, x="DUM_abr08a_v21", y="percent", hue="dem99_c", hue_order=[1, 0], order=[1, 0], palette=["black", "grey"])
p.set_xticklabels(["yes", "no"])
p.set_yticklabels(["", "20", "40", "60", "80", "100"])
p.set_xlabel("Studies abroad financially supported by parents/partner ")
plt.ylim(0, 1)
p.set_ylabel("% of respondents")
h, l = p.get_legend_handles_labels()
p.legend(h, hue_labels, title="", loc="upper right")
plt.show()


# Fig_FS_abr08b_v21: "Finanzquelle Auslandsstudium BAFoeG"
data_temp = se_red_17
pct2 = (data_temp.groupby(["DUM_abr08b_v21", "dem99_c"]).size() / data_temp[data_temp["DUM_abr08b_v21"].notna()].groupby(["dem99_c"]).size()).reset_index().rename({0:"percent"}, axis=1)
hue_labels = ["first generation students", "non first generation students"]
p = sns.barplot(data=pct2, x="DUM_abr08b_v21", y="percent", hue="dem99_c", hue_order=[1, 0], order=[1, 0], palette=["black", "grey"])
p.set_xticklabels(["yes", "no"])
p.set_yticklabels(["", "20", "40", "60", "80", "100"])
p.set_xlabel("Studies abroad financially supported by Federal Education Support Act")
plt.ylim(0, 1)
p.set_ylabel("% of respondents")
h, l = p.get_legend_handles_labels()
p.legend(h, hue_labels, title="", loc="upper right")
plt.show()


# Fig_FS_abr08c_v21: "Finanzquelle Auslandsstudium vorheriger Verdienst"
data_temp = se_red_17
pct2 = (data_temp.groupby(["DUM_abr08c_v21", "dem99_c"]).size() / data_temp[data_temp["DUM_abr08c_v21"].notna()].groupby(["dem99_c"]).size()).reset_index().rename({0:"percent"}, axis=1)
hue_labels = ["first generation students", "non first generation students"]
p = sns.barplot(data=pct2, x="DUM_abr08c_v21", y="percent", hue="dem99_c", hue_order=[1, 0], order=[1, 0], palette=["black", "grey"])
p.set_xticklabels(["yes", "no"])
p.set_yticklabels(["", "20", "40", "60", "80", "100"])
p.set_xlabel("Studies abroad financially supported by own prior income")
plt.ylim(0, 1)
p.set_ylabel("% of respondents")
h, l = p.get_legend_handles_labels()
p.legend(h, hue_labels, title="", loc="upper right")
plt.show()


# Fig_FS_abr09a_v21: "Finanzquelle Auslandspraktikum Eltern/Partner"
data_temp = se_red_17
pct2 = (data_temp.groupby(["DUM_abr09a_v21", "dem99_c"]).size() / data_temp[data_temp["DUM_abr09a_v21"].notna()].groupby(["dem99_c"]).size()).reset_index().rename({0:"percent"}, axis=1)
hue_labels = ["first generation students", "non first generation students"]
p = sns.barplot(data=pct2, x="DUM_abr09a_v21", y="percent", hue="dem99_c", hue_order=[1, 0], order=[1, 0], palette=["black", "grey"])
p.set_xticklabels(["yes", "no"])
p.set_yticklabels(["", "20", "40", "60", "80", "100"])
p.set_xlabel("Internship abroad financially supported by parents/partner")
plt.ylim(0, 1)
p.set_ylabel("% of respondents")
h, l = p.get_legend_handles_labels()
p.legend(h, hue_labels, title="", loc="upper right")
plt.show()


# Fig_FS_abr09b_v21: "Finanzquelle Auslandspraktikum vorheriger Verdienst"
data_temp = se_red_17
pct2 = (data_temp.groupby(["DUM_abr09b_v21", "dem99_c"]).size() / data_temp[data_temp["DUM_abr09b_v21"].notna()].groupby(["dem99_c"]).size()).reset_index().rename({0:"percent"}, axis=1)
hue_labels = ["first generation students", "non first generation students"]
p = sns.barplot(data=pct2, x="DUM_abr09b_v21", y="percent", hue="dem99_c", hue_order=[1, 0], order=[1, 0], palette=["black", "grey"])
p.set_xticklabels(["yes", "no"])
p.set_yticklabels(["", "20", "40", "60", "80", "100"])
p.set_xlabel("Internship abroad financially supported by own prior income")
plt.ylim(0, 1)
p.set_ylabel("% of respondents")
h, l = p.get_legend_handles_labels()
p.legend(h, hue_labels, title="", loc="upper right")
plt.show()


# Fig_FS_fin04a: "Financial situation: parents support as good as they can"
data_temp = se_red_17
pct2 = (data_temp.groupby(["fin04a", "dem99_c"]).size() / data_temp[data_temp["fin04a"].notna()].groupby(["dem99_c"]).size()).reset_index().rename({0:"percent"}, axis=1)
hue_labels = ["first generation students", "non first generation students"]
p = sns.barplot(data=pct2, x="fin04a", y="percent", hue="dem99_c", hue_order=[1, 0], palette=["black", "grey"])
plt.ylim(0, 1)
p.set_yticklabels(["", "20", "40", "60", "80", "100"])
p.set_ylabel("% of respondents")
# plt.xlim(0, 5)
# p.set_xticks([1.0, 2.0, 3.0, 4.0, 5.0])
p.set_xticklabels(["not at all true", "not true", "neutral", "true", "absolutely true"])
p.set_xlabel('"Financial situation: Parents support as good as they can"')
h, l = p.get_legend_handles_labels()
p.legend(h, hue_labels, title="", loc="upper right")
plt.show()


# Fig_FS_fin04b: "Financial situation: Impression of overburdening parents"
data_temp = se_red_17
pct2 = (data_temp.groupby(["fin04b", "dem99_c"]).size() / data_temp[data_temp["fin04b"].notna()].groupby(["dem99_c"]).size()).reset_index().rename({0:"percent"}, axis=1)
hue_labels = ["first generation students", "non first generation students"]
p = sns.barplot(data=pct2, x="fin04b", y="percent", hue="dem99_c", hue_order=[1, 0], palette=["black", "grey"])
plt.ylim(0, 1)
p.set_yticklabels(["", "20", "40", "60", "80", "100"])
p.set_ylabel("% of respondents")
# plt.xlim(0, 5)
# p.set_xticks([1.0, 2.0, 3.0, 4.0, 5.0])
p.set_xticklabels(["not at all true", "not true", "neutral", "true", "absolutely true"])
p.set_xlabel('"Financial situation: Impression of overburdening parents"')
h, l = p.get_legend_handles_labels()
p.legend(h, hue_labels, title="", loc="upper right")
plt.show()


# Fig_FS_fin04c: "Financial situation: want to be independent from parents"
data_temp = se_red_17
pct2 = (data_temp.groupby(["fin04c", "dem99_c"]).size() / data_temp[data_temp["fin04c"].notna()].groupby(["dem99_c"]).size()).reset_index().rename({0:"percent"}, axis=1)
hue_labels = ["first generation students", "non first generation students"]
p = sns.barplot(data=pct2, x="fin04c", y="percent", hue="dem99_c", hue_order=[1, 0], palette=["black", "grey"])
plt.ylim(0, 1)
p.set_yticklabels(["", "20", "40", "60", "80", "100"])
p.set_ylabel("% of respondents")
# plt.xlim(0, 5)
# p.set_xticks([1.0, 2.0, 3.0, 4.0, 5.0])
p.set_xticklabels(["not at all true", "not true", "neutral", "true", "absolutely true"])
p.set_xlabel('"Financial situation: want to be independent from parents"')
h, l = p.get_legend_handles_labels()
p.legend(h, hue_labels, title="", loc="upper right")
plt.show()


# Fig_FS_fin04i: "Financial situation: cost of living secured"
data_temp = se_red_17
pct2 = (data_temp.groupby(["fin04i", "dem99_c"]).size() / data_temp[data_temp["fin04i"].notna()].groupby(["dem99_c"]).size()).reset_index().rename({0:"percent"}, axis=1)
hue_labels = ["first generation students", "non first generation students"]
p = sns.barplot(data=pct2, x="fin04i", y="percent", hue="dem99_c", hue_order=[1, 0], palette=["black", "grey"])
plt.ylim(0, 1)
p.set_yticklabels(["", "20", "40", "60", "80", "100"])
p.set_ylabel("% of respondents")
# plt.xlim(0, 5)
# p.set_xticks([1.0, 2.0, 3.0, 4.0, 5.0])
p.set_xticklabels(["not at all true", "not true", "neutral", "true", "absolutely true"])
p.set_xlabel('"Financial situation: cost of living secured"')
h, l = p.get_legend_handles_labels()
p.legend(h, hue_labels, title="", loc="upper right")
plt.show()


# Fig_SC_fin04f: "Financial situation: could not study without federal education support"
data_temp = se_red_17
pct2 = (data_temp.groupby(["fin04f_v21", "dem99_c"]).size() / data_temp[data_temp["fin04f_v21"].notna()].groupby(["dem99_c"]).size()).reset_index().rename({0:"percent"}, axis=1)
hue_labels = ["first generation students", "non first generation students"]
p = sns.barplot(data=pct2, x="fin04f_v21", y="percent", hue="dem99_c", hue_order=[1, 0], palette=["black", "grey"])
plt.ylim(0, 1)
p.set_yticklabels(["", "20", "40", "60", "80", "100"])
p.set_ylabel("% of respondents")
# plt.xlim(0, 5)
# p.set_xticks([1.0, 2.0, 3.0, 4.0, 5.0])
p.set_xticklabels(["not at all true", "not true", "neutral", "true", "absolutely true"])
p.set_xlabel('"Financial situation: could not study without federal education support"')
h, l = p.get_legend_handles_labels()
p.legend(h, hue_labels, title="", loc="upper right")
plt.show()


# Fig_FS_job02_h: "Erwerbstätigkeit im laufenden Semester"
data_temp = se_red_17
pct2 = (data_temp.groupby(["DUM_job02_h", "dem99_c"]).size() / data_temp[data_temp["DUM_job02_h"].notna()].groupby(["dem99_c"]).size()).reset_index().rename({0:"percent"}, axis=1)
hue_labels = ["first generation students", "non first generation students"]
p = sns.barplot(data=pct2, x="DUM_job02_h", y="percent", hue="dem99_c", hue_order=[1, 0], order=[1, 0], palette=["black", "grey"])
p.set_xticklabels(["yes", "no"])
p.set_yticklabels(["", "20", "40", "60", "80", "100"])
p.set_xlabel("Working alongside studies in current semester")
plt.ylim(0, 1)
p.set_ylabel("% of respondents")
h, l = p.get_legend_handles_labels()
p.legend(h, hue_labels, title="", loc="upper right")
plt.show()


# Fig_SC_job05a: "Erwerbsgrund: Job notwendig für Lebensunterhalt"
data_temp = se_red_17
pct2 = (data_temp.groupby(["job05a", "dem99_c"]).size() / data_temp[data_temp["job05a"].notna()].groupby(["dem99_c"]).size()).reset_index().rename({0:"percent"}, axis=1)
hue_labels = ["first generation students", "non first generation students"]
p = sns.barplot(data=pct2, x="job05a", y="percent", hue="dem99_c", hue_order=[1, 0], palette=["black", "grey"])
plt.ylim(0, 1)
p.set_yticklabels(["", "20", "40", "60", "80", "100"])
p.set_ylabel("% of respondents")
# plt.xlim(0, 5)
# p.set_xticks([1.0, 2.0, 3.0, 4.0, 5.0])
p.set_xticklabels(["not at all true", "not true", "neutral", "true", "absolutely true"])
p.set_xlabel('"Reason for job alongside studies: necessary to secure living"')
h, l = p.get_legend_handles_labels()
p.legend(h, hue_labels, title="", loc="upper right")
plt.show()


# Fig_SC_job05e: "Erwerbsgrund: Unabhängigkeit von Eltern"
data_temp = se_red_17
pct2 = (data_temp.groupby(["job05e", "dem99_c"]).size() / data_temp[data_temp["job05e"].notna()].groupby(["dem99_c"]).size()).reset_index().rename({0:"percent"}, axis=1)
hue_labels = ["first generation students", "non first generation students"]
p = sns.barplot(data=pct2, x="job05e", y="percent", hue="dem99_c", hue_order=[1, 0], palette=["black", "grey"])
plt.ylim(0, 1)
p.set_yticklabels(["", "20", "40", "60", "80", "100"])
p.set_ylabel("% of respondents")
# plt.xlim(0, 5)
# p.set_xticks([1.0, 2.0, 3.0, 4.0, 5.0])
p.set_xticklabels(["not at all true", "not true", "neutral", "true", "absolutely true"])
p.set_xlabel('"Reason for job alongside studies: independence from parents"')
h, l = p.get_legend_handles_labels()
p.legend(h, hue_labels, title="", loc="upper right")
plt.show()


# Fig_SC_job05h: "Erwerbsgrund: Studiengebühren finanzieren"
data_temp = se_red_17
pct2 = (data_temp.groupby(["job05h", "dem99_c"]).size() / data_temp[data_temp["job05h"].notna()].groupby(["dem99_c"]).size()).reset_index().rename({0:"percent"}, axis=1)
hue_labels = ["first generation students", "non first generation students"]
p = sns.barplot(data=pct2, x="job05h", y="percent", hue="dem99_c", hue_order=[1, 0], palette=["black", "grey"])
plt.ylim(0, 1)
p.set_yticklabels(["", "20", "40", "60", "80", "100"])
p.set_ylabel("% of respondents")
# plt.xlim(0, 5)
# p.set_xticks([1.0, 2.0, 3.0, 4.0, 5.0])
p.set_xticklabels(["not at all true", "not true", "neutral", "true", "absolutely true"])
p.set_xlabel('"Reason for job alongside studies: pay tuition"')
h, l = p.get_legend_handles_labels()
p.legend(h, hue_labels, title="", loc="upper right")
plt.show()


# Fig_SC_stu11d: "Unterbrechungsgrund: Finanzielle Probleme"
data_temp = se_red_17
pct2 = (data_temp.groupby(["DUM_stu11d", "dem99_c"]).size() / data_temp[data_temp["DUM_stu11d"].notna()].groupby(["dem99_c"]).size()).reset_index().rename({0:"percent"}, axis=1)
hue_labels = ["first generation students", "non first generation students"]
p = sns.barplot(data=pct2, x="DUM_stu11d", y="percent", hue="dem99_c", hue_order=[1, 0], order=[1, 0], palette=["black", "grey"])
p.set_xticklabels(["yes", "no"])
p.set_yticklabels(["", "20", "40", "60", "80", "100"])
p.set_xlabel("Reason for interrupting studies: financial problems")
plt.ylim(0, 1)
p.set_ylabel("% of respondents")
h, l = p.get_legend_handles_labels()
p.legend(h, hue_labels, title="", loc="upper right")
plt.show()


# Fig_SC_stu11f: "Unterbrechungsgrund: Zweifel am Sinn des Studiums"
data_temp = se_red_17
pct2 = (data_temp.groupby(["DUM_stu11f", "dem99_c"]).size() / data_temp[data_temp["DUM_stu11f"].notna()].groupby(["dem99_c"]).size()).reset_index().rename({0:"percent"}, axis=1)
hue_labels = ["first generation students", "non first generation students"]
p = sns.barplot(data=pct2, x="DUM_stu11f", y="percent", hue="dem99_c", hue_order=[1, 0], order=[1, 0], palette=["black", "grey"])
p.set_xticklabels(["yes", "no"])
p.set_yticklabels(["", "20", "40", "60", "80", "100"])
p.set_xlabel("Reason for interrupting studies: doubts about purpose of studying")
plt.ylim(0, 1)
p.set_ylabel("% of respondents")
h, l = p.get_legend_handles_labels()
p.legend(h, hue_labels, title="", loc="upper right")
plt.show()


# Fig_SC_tim02z: "Studienaufwand pro Woche"
data_temp = se_red_17[se_red_17["tim02z_c"] != 0]
p = sns.boxplot(x=data_temp["dem99_c"], y=data_temp["tim02z_c"], data=data_temp, showfliers=False, palette=["white", "grey"], order=[1, 0])
p.set_xlabel("")
p.set_ylabel("study time outside classroom (h/week)")
p.set_xticklabels(["first generation students", "other students"])
# p.set(ylim=(0,1500))
plt.show()


# Fig_SC_tim03z: "Arbeitszeit pro Woche"
data_temp = se_red_17[se_red_17["tim03z_c"] != 0]
p = sns.boxplot(x=data_temp["dem99_c"], y=data_temp["tim03z_c"], data=data_temp, showfliers=False, palette=["white", "grey"], order=[1, 0])
p.set_xlabel("")
p.set_ylabel("work time alongside studies (h/week)")
p.set_xticklabels(["first generation students", "other students"])
# p.set(ylim=(0,1500))
plt.show()
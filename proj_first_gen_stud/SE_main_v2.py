import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr


# SE_main_v2: Since v1: Here changed how categorical variables are relabelled in the course of dummy coding

# Loading relevant variables from data into dataframe, while replacing missing values with NaN"
# initially was replacing relevant values here with na_values argument, but then realized can be misleading,
# need to investigate for each clm: what do missing value expressions indicate, how can they be replaced

# 26.03.: It seems that I actually have to set this whole thing up differently: Just comparing population means. If
# I do that, and should use two-sample t-tests, I should point out in narrative that this is also used in A/B testing!

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
se_red_17.loc[se_red_17["abr02b_v21"] == "nicht genannt", "abr02b_v21"] = "keine Praktikum im Ausland während studienbezogenem Auslandsaufenthalt"
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
se_red_17["dem99_c"].unique()
se_red_17["dem99_c"].describe


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


# Check right colum types? Nans where needed?
print(se_red_17.info(max_cols=247))
se_red_17_wod = se_red_17 

## Dummy-code nominal categorical variables
##  - create new binary variable for each unique value
##  - assign numerical values to  ordinal categorical variables (e.g., Likert scales), if necessary
##  - clean the original columns for which I now have dummy variables; reset dataframe index values

# create list with variables that require dummies, create loop iterating through list: name dummies after original variable, insert dummy clms,
list = ["dem01_h", "dem11a_h", "par03_v21", "par04_v21", "par05_h", "par06_h", "par07_h", "par08_h", "par09", "par10", "stu01a_h", "stu02_h", "stu03_h", "stu16b_h", "ped01_v21", "job03n_v21", "abr12_h", "liv01_v21"]
for i in list:
    pref = "DUM_" + i
    se_red_17 = pd.get_dummies(se_red_17, prefix=pref, columns=[i], dummy_na=True)

se_red_17.columns.values

# rearrange dataframe so that dummy columns are right after original variable
se_red_17 = se_red_17.reindex(columns=["index", "id", "dem99_c", "DUM_dem01_h_männlich", "DUM_dem01_h_weiblich", "DUM_dem01_h_divers", "DUM_dem01_h_nan", "dem02_h", "DUM_dem03_v21", "DUM_dem05", "DUM_dem09", "DUM_dem11a_h_andere Staatsangehörigkeit", "DUM_dem11a_h_deutsche Staatsangehörigkeit", "DUM_dem11a_h_deutsche u. andere Staatsangeh.", "DUM_dem11a_h_nan", "DUM_par03_v21_Fachhochschulreife", "DUM_par03_v21_Haupt-/Volksschulabschluss", "DUM_par03_v21_Realschulabschluss, mittlere Reife", "DUM_par03_v21_allg./fachg. Hochschulreife", "DUM_par03_v21_anderer Schulabschluss", "DUM_par03_v21_keinen Schulabschluss", "DUM_par03_v21_nan", "DUM_par04_v21_Fachhochschulreife", "DUM_par04_v21_Haupt-/Volksschulabschluss", "DUM_par04_v21_Realschulabschluss, mittlere Reife", "DUM_par04_v21_allg./fachg. Hochschulreife", "DUM_par04_v21_anderer Schulabschluss", "DUM_par04_v21_keinen Schulabschluss", "DUM_par04_v21_nan", "DUM_par05_h_akademischer Abschluss", "DUM_par05_h_keinen beruflichen Abschluss", "DUM_par05_h_nicht-akademischer Berufsabschluss", "DUM_par05_h_nan", "DUM_par06_h_akademischer Abschluss", "DUM_par06_h_keinen beruflichen Abschluss", "DUM_par06_h_nicht-akademischer Berufsabschluss", "DUM_par06_h_nan", "DUM_par07_h_Angestellter", "DUM_par07_h_Arbeiter", "DUM_par07_h_Beamter", "DUM_par07_h_Selbständiger", "DUM_par07_h_nie berufstätig gewesen", "DUM_par07_h_nan", "DUM_par08_h_Angestellte", "DUM_par08_h_Arbeiterin", "DUM_par08_h_Beamtin", "DUM_par08_h_Selbständige", "DUM_par08_h_nie berufstätig gewesen", "DUM_par08_h_nan", "DUM_par09_andere Staatsangehörigkeit", "DUM_par09_deutsche Staatsangehörigkeit", "DUM_par09_deutsche u. andere Staatsangeh.", "DUM_par09_nan", "DUM_par10_andere Staatsangehörigkeit", "DUM_par10_deutsche Staatsangehörigkeit", "DUM_par10_deutsche u. andere Staatsangeh.", "DUM_par10_nan", "DUM_stu01a_h_Agrar-, Forst-, Ernährg.wiss.", "DUM_stu01a_h_Humanmedizin/Gesundheitswiss.", "DUM_stu01a_h_Ingenieurwiss.", "DUM_stu01a_h_Kunst, Kunstwiss.", "DUM_stu01a_h_Mathematik,  Naturwiss.", "DUM_stu01a_h_Rechts-, Wirtsch.-, Sozialwiss.", "DUM_stu01a_h_Sport", "DUM_stu01a_h_Sprach-, Kulturwiss.", "DUM_stu01a_h_andere", "DUM_stu01a_h_nan", "DUM_stu02_h_Bachelor", "DUM_stu02_h_Diplom", "DUM_stu02_h_Magister", "DUM_stu02_h_Master", "DUM_stu02_h_Staatsexamen", "DUM_stu02_h_anderer Abschluss", "DUM_stu02_h_keinen Abschluss", "DUM_stu02_h_nan", "DUM_stu03_h_Bachelor", "DUM_stu03_h_Diplom", "DUM_stu03_h_Magister", "DUM_stu03_h_Master", "DUM_stu03_h_Promotion", "DUM_stu03_h_Staatsexamen", "DUM_stu03_h_kein vorhandener Abschluss", "DUM_stu03_h_sonstiger Abschluss", "DUM_stu03_h_nan", "stu04", "stu05", "DUM_stu10_h", "DUM_stu11b", "DUM_stu11c_v20", "DUM_stu11d", "DUM_stu11e", "DUM_stu11f", "DUM_stu11g", "DUM_stu11h_v21", "stu12_v21", "DUM_stu13_h", "DUM_stu16a_h", "DUM_stu16b_h_Norddeutschland", "DUM_stu16b_h_Ostdeutschland", "DUM_ped01_v21_Fachhochschulreife", "DUM_ped01_v21_allg. Hochschulreife", "DUM_ped01_v21_andere Studienberechtigung", "DUM_ped01_v21_berufl. Qualifikation", "DUM_ped01_v21_fachg. Hochschulreife", "DUM_ped01_v21_nan", "DUM_ped03", "fin01a", "fin01b", "fin01c", "fin01d", "fin01e", "fin01f_h", "fin01g", "fin01h", "fin01j", "fin01o", "fin02a_h", "fin02b_h", "fin02c_h", "fin02d_h", "fin02e_h", "fin02f_h", "fin02g_h", "fin02h_h", "fin02i_h", "fin03a_h", "fin03b_h", "fin03c_h", "fin03d_h", "fin03e_h", "fin03f_h", "fin03g_h", "fin03h_h", "fin03i_h", "fin04a", "fin04b", "fin04c", "fin04d_v21", "fin04e_v21", "fin04f_v21", "fin04i", "DUM_baf01_h", "DUM_baf05a", "DUM_baf05b", "DUM_baf05c", "DUM_baf05e", "DUM_baf05f_v20", "DUM_baf05g", "DUM_baf05h_v21", "baf05i_v21", "tim02z_c", "tim03z_c", "DUM_job02_h", "DUM_job03a_v21", "DUM_job03b_v21", "DUM_job03g_v21", "DUM_job03n_v21_Jobben", "DUM_job03n_v21_Nachhilfeunterricht", "DUM_job03n_v21_Taetigkeit, die Hochschulabschluss voraussetzt", "DUM_job03n_v21_Taetigkeit, die beruflichen Ausbildungsabschluss voraussetzt", "DUM_job03n_v21_andere Tätigkeit", "DUM_job03n_v21_keine berufliche Tätigkeit", "DUM_job03n_v21_studentische/wissenschaftliche Hilfskraft", "DUM_job03n_v21_nan", "job05a", "job05b", "job05c", "job05d", "job05e", "job05f", "job05g", "job05h", "DUM_abr01_h", "DUM_abr02a_v21", "DUM_abr02b_v21", "DUM_abr02d_v21", "abr03a_v21", "abr03b_v21", "abr03d_v21", "DUM_abr08a_v21", "DUM_abr08b_v21", "DUM_abr08c_v21", "DUM_abr08d_v21", "DUM_abr08e_v21", "DUM_abr08f_v21", "DUM_abr08g_v21", "DUM_abr08h_v21", "DUM_abr08i_v21", "DUM_abr09a_v21", "DUM_abr09b_v21", "DUM_abr09c_v21", "DUM_abr09d_v21", "DUM_abr09e_v21", "DUM_abr09f_v21", "DUM_abr09g_v21", "DUM_abr09h_v21", "DUM_abr09i_v21", "DUM_abr11_h", "DUM_abr12_h_ja", "DUM_abr12_h_nein, kein Interesse", "DUM_abr12_h_nein, sehe keine Realisierungschance", "DUM_abr12_h_weiß ich noch nicht", "DUM_abr12_h_nan", "abr13a_v21", "abr13b_v21", "abr13c_v21", "abr13d_v21", "abr13e_v21", "abr13f_v21", "lan01_v21", "DUM_liv01_v21_Eltern, Verwandte", "DUM_liv01_v21_Untermiete", "DUM_liv01_v21_Wohngemeinschaft", "DUM_liv01_v21_Wohnh.: Einzelappartment", "DUM_liv01_v21_Wohnh.: Einzelzi. in Wohngr.", "DUM_liv01_v21_Wohnh.: Einzelzimmer", "DUM_liv01_v21_Wohnh.: Mehr-Zi.-Wohng.", "DUM_liv01_v21_Wohnh.: Zweibett-Zimmer", "DUM_liv01_v21_Wohnung allein", "DUM_liv01_v21_Wohnung m. Partn./Kind", "DUM_liv01_v21_nan", "DUM_adv01a_v21", "DUM_adv01b_v21", "DUM_adv01c_v21", "DUM_adv01d_v21", "DUM_adv01e_v21", "DUM_adv01g_v21", "DUM_adv01i_v21", "DUM_adv01j_v21", "DUM_adv01k_v21", "DUM_adv01l_v21", "DUM_adv01m_v21", "DUM_adv01n_v21", "DUM_adv01o_v21", "DUM_adv01p_v21", "DUM_adv02_v21"])



# ------- THIS IS WHERE ACTUAL ANALYSIS STARTS ------- #
## "getting to know" the data: revisit research question, then investigate each variable"s statistics and take notes
## RQ: How do first generation and non-first generation students' study conditions differ?

## correlation table to identify relationships for visualization
se_red_17_corr = round(se_red_17.corr(min_periods=), 3)
se_red_17_corr.to_csv(r"C:\Users\marc.feldmann\Dropbox\Dissertation_MFeldmann\se_red_17_corr.csv")

## corr table with asterisks indicating sign:
## rho = se_red_17.corr()
## pval = se_red_17.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*rho.shape)
## p = pval.applymap(lambda x: ''.join(['*' for t in [0.01,0.05,0.1] if x<=t]))
## se_red_17_corr = rho.round(2).astype(str) + p

## for ordering x axis values:
## se_red_17.fin04a.value_counts().loc[["trifft gar nicht zu", "Pos. 2", "Pos. 3", "Pos. 4", "trifft völlig zu"]].plot.bar()


## ANALYTICAL NARRATIVE:
# see notes


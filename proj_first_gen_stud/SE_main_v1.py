import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Loading relevant variables from data into dataframe, while replacing missing values with NaN"
# initially was replacing following values here with na_values argument, but then realized can be misleading,
# need to investigate for each clm: what do missing value expressions indicate, how can they be replaced

# hervorheben in Beschreibung: die "breite" (?) des Datensatzes, Feature-Anzahl
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
# NEXT STEPS 20.02.: for each column: 
# a) explore unique values and distribution, based on that:
# b) interpret/replace missing values (see docu file)
# c) aggregate and or/ rename feature values to make better understandable in later output
# d) make sure that existing nominal and ordinal categorical variables are coded numerically (e.g., Likert-scale variables such as "Beratungsbedarf")
# e) declare column data type
## in docu: give rationale why I was not replacing all values with read_csv parameter (as I did originally:)
## because different types of 'missing value', need to inspect individually; also: same type can mean different things
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


## Kinder (dem05)
se_red_17["dem05"].dtype
round(se_red_17["dem05"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs
se_red_17.loc[se_red_17["dem05"] == "keine Angabe", "dem05"] = np.nan
se_red_17.loc[se_red_17["dem05"] == "unbekannter fehlender Wert", "dem05"] = np.nan
se_red_17.loc[se_red_17["dem05"] == "Interviewabbruch", "dem05"] = np.nan


## Geschwister (dem09)
se_red_17["dem09"].dtype
round(se_red_17["dem09"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs
se_red_17.loc[se_red_17["dem09"] == "keine Angabe", "dem09"] = np.nan
se_red_17.loc[se_red_17["dem09"] == "Interviewabbruch", "dem09"] = np.nan


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
se_red_17["stu04"] = pd.to_numeric(se_red_17["stu04"], errors='coerce')

### Declare NaNs
se_red_17.loc[se_red_17["stu04"] == "keine Angabe", "stu04"] = np.nan
se_red_17.loc[se_red_17["stu04"] == "unbekannter fehlender Wert", "stu04"] = np.nan


## Hochschulsemester (stu05)
se_red_17["stu05"].dtype
round(se_red_17["stu05"].value_counts(normalize=True, dropna=False), 3)
se_red_17["stu05"] = pd.to_numeric(se_red_17["stu05"], errors='coerce')

### Declare NaNs
se_red_17.loc[se_red_17["stu05"] == "keine Angabe", "stu05"] = np.nan
se_red_17.loc[se_red_17["stu05"] == "nicht bestimmbar", "stu05"] = np.nan


## Studienunterbrechung (stu10_h)
se_red_17["stu10_h"].unique()
se_red_17["stu10_h"].dtype
round(se_red_17["stu10_h"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs
se_red_17.loc[se_red_17["stu10_h"] == "keine Angabe", "stu10_h"] = np.nan
se_red_17.loc[se_red_17["stu10_h"] == "unbekannter fehlender Wert", "stu10_h"] = np.nan


## Unterbrechungsgrund (stu11X)
### Declare NaNs / rename values

#### for all columns that start with stu11, show unique values
#### to identify values  which need to be replaced with np.nan
list = ["stu11b", "stu11c_v20", "stu11d", "stu11e", "stu11f", "stu11g", "stu11h_v21"]
for i in list:
    round(se_red_17[i].value_counts(normalize=True, dropna=False), 3)

#### create new loop to replace with np.nan
for i in list:
    se_red_17.loc[se_red_17[i] == "filterbedingt fehlend", i] = "keine Unterbrechung"
    se_red_17.loc[se_red_17[i] == "splitbedingt fehlend", i] = "keine Unterbrechung"
    se_red_17.loc[se_red_17[i] == "keine Angabe", i] = np.nan


## Gesamtdauer Studienunterbrechung (Semester) (stu12_v21)
se_red_17["stu12_v21"].dtype
round(se_red_17["stu12_v21"].value_counts(normalize=True, dropna=False), 3)
se_red_17["stu12_v21"] = pd.to_numeric(se_red_17["stu12_v21"], errors='coerce')

### Declare NaNs / rename values
se_red_17.loc[se_red_17["stu12_v21"] == "nicht bestimmbar", "stu12_v21"] = np.nan
se_red_17.loc[se_red_17["stu12_v21"] == "splitbedingt fehlend", "stu12_v21"] = "keine Studienunterbrechung"


## Hochschule gewechselt? (stu13_h)
se_red_17["stu13_h"].dtype
round(se_red_17["stu13_h"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs
se_red_17.loc[se_red_17["stu13_h"] == "keine Angabe", "stu13_h"] = np.nan


## Hochschulart (stu16a_h)
se_red_17["stu16a_h"].dtype
round(se_red_17["stu16a_h"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs
se_red_17.loc[se_red_17["stu16a_h"] == "keine Angabe", "stu16a_h"] = np.nan
se_red_17.loc[se_red_17["stu16a_h"] == "nicht bestimmbar", "stu16a_h"] = np.nan


## Hochschulregion (stu16b_h)
se_red_17["stu16b_h"].dtype
round(se_red_17["stu16b_h"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs / rename values
se_red_17.loc[se_red_17["stu16b_h"] == "Ostdeutschland, inkl. Berlin", "stu16b_h"] = "Ostdeutschland"
se_red_17.loc[se_red_17["stu16b_h"] == "nicht bestimmbar", "stu16b_h"] = np.nan


## Hochschulregion (stu16b_h)
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


## Barmittel (baf05X)
### Declare NaNs / rename values

#### for all columns that start with baf05, show unique values
#### to identify values  which need to be replaced with np.nan
list = ["baf05a", "baf05b", "baf05c", "baf05e", "baf05f_v20", "baf05g", "baf05h_v21", "baf05i_v21"]
for i in list:
    round(se_red_17[i].value_counts(normalize=True, dropna=False), 3)

#### create new loop to replace with np.nan and enforce to_numeric in all columns starting with fin01
for i in list:
    se_red_17.loc[se_red_17[i] == "unbekannter fehlender Wert", i] = np.nan
    se_red_17.loc[se_red_17[i] == "splitbedingt fehlend", i] = "BAfoeG-Bezug/Antrag gestellt"
    se_red_17.loc[se_red_17[i] == "filterbedingt fehlend", i] = "BAfoeG-Bezug/Antrag gestellt"


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

### Declare NaNs
se_red_17.loc[se_red_17["job02_h"] == "keine Angabe", "job02_h"] = np.nan
se_red_17.loc[se_red_17["job02_h"] == "splitbedingt fehlend", "job02_h"] = np.nan


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


## studentische/wiss. Hilfskraft (job03b_v21)
se_red_17["job03b_v21"].dtype
round(se_red_17["job03b_v21"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs
se_red_17.loc[se_red_17["job03b_v21"] == "keine Angabe", "job03b_v21"] = np.nan


## Praktikum/Volontariat (job03g_v21)
se_red_17["job03g_v21"].dtype
round(se_red_17["job03g_v21"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs
se_red_17.loc[se_red_17["job03g_v21"] == "keine Angabe", "job03g_v21"] = np.nan


## berufsbegleitendes Studium (job03n_v21)
se_red_17["job03n_v21"].dtype
round(se_red_17["job03n_v21"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs / rename values
se_red_17.loc[se_red_17["job03n_v21"] == "keine Angabe", "job03n_v21"] = np.nan
se_red_17.loc[se_red_17["job03n_v21"] == "filterbedingt fehlend", "job03n_v21"] = "kein berufsbegleitendes Studium"
se_red_17.loc[se_red_17["job03n_v21"] == "splitbedingt fehlend", "job03n_v21"] = "kein berufsbegleitendes Studium"


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


## Studienbezogen im Ausland aufgehalten? (abr01_h)
se_red_17["abr01_h"].dtype
round(se_red_17["abr01_h"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs
se_red_17.loc[se_red_17["abr01_h"] == "keine Angabe", "abr01_h"] = np.nan
se_red_17.loc[se_red_17["abr01_h"] == "Interviewabbruch", "abr01_h"] = np.nan


## Auslandsaufenthalt Programmbestandteil? (abr11_h)
se_red_17["abr11_h"].dtype
round(se_red_17["abr11_h"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs / rename values
se_red_17.loc[se_red_17["abr11_h"] == "filterbedingt fehlend", "abr11_h"] = "kein studienbezogener Auslandsaufenthalt"
se_red_17.loc[se_red_17["abr11_h"] == "keine Angabe", "abr11_h"] = np.nan
se_red_17.loc[se_red_17["abr11_h"] == "Interviewabbruch", "abr11_h"] = np.nan


## Hindernis Auslandsaufenthalt (abr13X)
## Filter/Split: Studierende, die noch nicht studienbezogen im Ausland waren
## und auch keine Auslandsaufenthalt planen
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
    se_red_17.loc[se_red_17[i] == "splitbedingt fehlend", i] = "Auslandsaufenthalt bereits realisiert oder geplant"
    se_red_17.loc[se_red_17[i] == "filterbedingt fehlend", i] = "Auslandsaufenthalt bereits realisiert oder geplant"


## AUSLANDSSTUDIUM (abr02a_v21)
se_red_17["abr02a_v21"].dtype
round(se_red_17["abr02a_v21"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs / rename values
se_red_17.loc[se_red_17["abr02a_v21"] == "keine Angabe", "abr02a_v21"] = np.nan
se_red_17.loc[se_red_17["abr02a_v21"] == "Interviewabbruch", "abr02a_v21"] = np.nan
se_red_17.loc[se_red_17["abr02a_v21"] == "filterbedingt fehlend", "abr02a_v21"] = "kein studienbezogener Auslandsaufenthalt"
se_red_17.loc[se_red_17["abr02a_v21"] == "nicht genannt", "abr02a_v21"] = "kein Studium an ausländischer Hochschule während studienbezogenem Auslandsaufenthalt"


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
    se_red_17.loc[se_red_17[i] == "splitbedingt fehlend", i] = "kein Auslandsstudium"
    se_red_17.loc[se_red_17[i] == "filterbedingt fehlend", i] = "kein Auslandsstudium"



# Auslandsstudium/-studien: Dauer in Monaten (abr03a_v21)
se_red_17["abr03a_v21"].dtype
round(se_red_17["abr03a_v21"].value_counts(normalize=True, dropna=False), 3)
se_red_17["abr03a_v21"] = pd.to_numeric(se_red_17["abr03a_v21"], errors='coerce')

### Declare NaNs / rename values
se_red_17.loc[se_red_17["abr03a_v21"] == "keine Angabe", "abr03a_v21"] = np.nan
se_red_17.loc[se_red_17["abr03a_v21"] == "unbekannter fehlender Wert", "abr03a_v21"] = np.nan
se_red_17.loc[se_red_17["abr03a_v21"] == "Interviewabbruch", "abr03a_v21"] = np.nan
se_red_17.loc[se_red_17["abr03a_v21"] == "splitbedingt fehlend", "abr03a_v21"] = "kein Studium an ausländischer Hochschule während studienbezogenem Auslandsaufenthalt"


## AUSLANDSPRAKTIKUM (abr02b_v21)
se_red_17["abr02b_v21"].dtype
round(se_red_17["abr02b_v21"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs / rename values
se_red_17.loc[se_red_17["abr02b_v21"] == "keine Angabe", "abr02b_v21"] = np.nan
se_red_17.loc[se_red_17["abr02b_v21"] == "filterbedingt fehlend", "abr02b_v21"] = "kein studienbezogener Auslandsaufenthalt"
se_red_17.loc[se_red_17["abr02b_v21"] == "nicht genannt", "abr02b_v21"] = "keine Praktikum im Ausland während studienbezogenem Auslandsaufenthalt"
se_red_17.loc[se_red_17["abr02b_v21"] == "Interviewabbruch", "abr02b_v21"] = np.nan


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


# Auslandspraktikum/-praktika: Dauer in Monaten (abr03b_v21)
se_red_17["abr03b_v21"].dtype
round(se_red_17["abr03b_v21"].value_counts(normalize=True, dropna=False), 3)
se_red_17["abr03b_v21"] = pd.to_numeric(se_red_17["abr03b_v21"], errors='coerce')

### Declare NaNs / rename values
se_red_17.loc[se_red_17["abr03b_v21"] == "keine Angabe", "abr03b_v21"] = np.nan
se_red_17.loc[se_red_17["abr03b_v21"] == "unbekannter fehlender Wert", "abr03b_v21"] = np.nan
se_red_17.loc[se_red_17["abr03b_v21"] == "Interviewabbruch", "abr03b_v21"] = np.nan
se_red_17.loc[se_red_17["abr03b_v21"] == "splitbedingt fehlend", "abr03b_v21"] = "keine Praktikum im Ausland während studienbezogenem Auslandsaufenthalt"


## SONSTIGE AUSLANDSAUFENTHALTE (abr02d_v21)
se_red_17["abr02d_v21"].dtype
round(se_red_17["abr02d_v21"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs / rename values
se_red_17.loc[se_red_17["abr02d_v21"] == "keine Angabe", "abr02d_v21"] = np.nan
se_red_17.loc[se_red_17["abr02d_v21"] == "filterbedingt fehlend", "abr02d_v21"] = "kein studienbezogener Auslandsaufenthalt"
se_red_17.loc[se_red_17["abr02d_v21"] == "nicht genannt", "abr02d_v21"] = "keine sonstige Aktivitäten (e.g., Summerschool, Exkursion) im Ausland während studienbezogenem Auslandsaufenthalt"
se_red_17.loc[se_red_17["abr02d_v21"] == "Interviewabbruch", "abr02d_v21"] = np.nan


## Sprachkenntnisse: Englisch (lan01_v21)
se_red_17["lan01_v21"].dtype
round(se_red_17["lan01_v21"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs / rename values
se_red_17.loc[se_red_17["lan01_v21"] == "keine Angabe", "lan01_v21"] = np.nan
se_red_17.loc[se_red_17["lan01_v21"] == "trifft nicht zu", "lan01_v21"] = "keine"
se_red_17.loc[se_red_17["lan01_v21"] == "Interviewabbruch", "lan01_v21"] = np.nan


## Wohnform (liv01_v21)
se_red_17["liv01_v21"].dtype
round(se_red_17["liv01_v21"].value_counts(normalize=True, dropna=False), 3)

### Declare NaNs / rename values
se_red_17.loc[se_red_17["liv01_v21"] == "nicht bestimmbar", "lan01_v21"] = np.nan
se_red_17.loc[se_red_17["liv01_v21"] == "trifft nicht zu", "lan01_v21"] = "keine"
se_red_17.loc[se_red_17["liv01_v21"] == "Interviewabbruch", "lan01_v21"] = np.nan


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
    se_red_17.loc[se_red_17[i] == "splitbedingt fehlend", i] = "keine Schwierigkeiten o. Belastungen"


# CREATE FEATURES
## "First Generation Student" (dem99_c): infer from par05_h and par06_h

## create new column and set all cells to "FGS"
se_red_17.insert(8, "dem99_c", "FGS")
se_red_17["dem99_c"].unique()
se_red_17["dem99_c"].describe


## if either father or mother has "akademischer Abschluss": set cell in dem99_c in same row to "Non-FGS"
list = se_red_17[se_red_17["par05_h"]=="akademischer Abschluss"].index.values
for i in list:
    se_red_17.iloc[i, se_red_17.columns.get_loc("dem99_c")] = "Non-FGS"

list = se_red_17[se_red_17["par06_h"]=="akademischer Abschluss"].index.values
for i in list:
    se_red_17.iloc[i, se_red_17.columns.get_loc("dem99_c")] = "Non-FGS"

## if both mother and father columns are NaN: NaN
list = se_red_17[se_red_17["par05_h"].isnull()].index.tolist()
for i in list:
    if pd.isnull(se_red_17.iloc[i, se_red_17.columns.get_loc("par06_h")]):
        se_red_17["dem99_c"].iloc[i] = np.nan

list = se_red_17[se_red_17["par06_h"].isnull()].index.tolist()
for i in list:
    if pd.isnull(se_red_17.iloc[i, se_red_17.columns.get_loc("par05_h")]):
        se_red_17["dem99_c"].iloc[i] = np.nan


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


## CHECK right colum types? Nans where needed?
print(se_red_17.info(max_cols=160))


## Dummy-code nominal categorical variables
##  - create new binary variable for each unique value
##  - assign numerical values to  ordinal categorical variables (e.g., Likert scales), if necessary




# ------- THIS IS WHERE ACTUAL ANALYSIS STARTS ------- #
## "getting to know" the data: revisit research question, then investigate each variable"s statistics and take notes


# for ordering x axis values:
## se_red_17.fin04a.value_counts().loc[["trifft gar nicht zu", "Pos. 2", "Pos. 3", "Pos. 4", "trifft völlig zu"]].plot.bar()

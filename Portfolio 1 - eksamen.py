#Opgave 1. 
#a. Hvilke data typer kan i identificere? (tekst / tal / typer af tal mv.)
#Ved at se på filen i Notepad, har vi kunne se, at de forskellige informationer er listet i rækkefølge, pænt struktureret og adskilt af kommaer, der fungerer som kolonner. Der er 8 kolonner: Om passageren overlevede (angivet boolsk, i 0 eller 1 - vi antager at 1 betyder personen overlevede), passagerklasse, navn, køn, alder, antal søskende/ægtefæller om bord, antal forældre/børn om bord, samt hvor meget de har betalt for overfarten.
#Det er værd at nævne, at gifte kvinders navne er sat i parentes, mens mandens fulde navn også står ved personen. 
#b. Mangler der data?
#Ja, på Titanic var der ca. 1.300 passagerer. Disse passagerer var fordelt med 325 på 1. klasse, 285 på 2. klasse og 706 på 3. klasse, plus ca. 900 besætningsmedlemmer. Dette datasæt indeholder kun 887 passagerer. Derfor mangler der ca. 500 passagerer i dette regnestykke, foruden besætningen.
#Derudover har vi diskuteret hvilke typer af data, der også kunne være angivet for at skabe et mere fyldestgørende datasæt. Der kunne godt have været flere brugbare informationer om passagerne i form af: nationalitet, anvendt type af valuta til betaling, billet nr., kabine nr. og hvilken havn, der blev anvendt til påstigning.
#Samtidig savner vi en bedre separation mellem søskende, ægtefæller, forældre og børn i kolonnerne.
#Derudover kunne data omkring passagernes erhverv og religion også være interessante at undersøge.

#Opgave 2. 
#(følgende opgave 2-5 er Python kode, derfor udeladt at fjerne #)
import pandas as pd
titanic = pd.read_csv("titanic.csv", sep=',')
# print(titanic)
print(titanic.head()) # Viser de første 5 linier af filen for at få en forståelse af dataen
print(len(titanic)) # Antal rows
print(titanic.shape) # Antal rows & columns 
print(titanic.size) # Antal celler i hele filen
print(titanic.columns) # Navnene til columns
print(titanic.dtypes) # Data typer af filen int, str eller float. 
# Kilde: https://datacarpentry.org/python-socialsci/08-Pandas/index.html 

#Opgave 3. 
# Antal af personer der overlevede = Summen af Survived column
print('Antal af personer der overlevede:', sum(titanic.Survived))
# Gennemsnitsalderen på passagererne = mean af Age column
print('Gennemsnitsalderen for alle passagerer på Titanic:', titanic['Age'].mean())
# Median alder på passagerne = median af Age column
print('Median alderen på alle passagerer på Titanic:', titanic['Age'].median())
# Mindste værdi altså yngste passager = min af Age column
print('Alder på yngste passager:', titanic['Age'].min())
# Max værdi ældste passager = max af Age colum
print('Alder på ældste passager:', titanic['Age'].max())
# Antal af passagerer på hver klasse = value_count af Pclass
print(titanic['Pclass'].value_counts())
# Gennemsnitspris pr billet. Dog er billet ikke angivet i valuta = mean af Fare column
print('Gennemsnitspris pr billet alle passagerer på Titanic:', titanic['Fare'].mean())
# Kilde: https://datacarpentry.org/python-socialsci/10-aggregations/index.html 

#Opgave 4. 
#Hvor mange har samme efternavn
new = titanic['Name'].str.split(r'((\w+$))', n = 0, expand = True)
new[1].value_counts()
new = titanic['Name'].str.rsplit(pat = r'((\w+^))', expand = True)
# Column name split str op, r = reverse \w er helt ord og $ er anker
new = (titanic['Name'].str.split().str[-1])
# -1 betyder at der startes med den sidste karakter i string. 
print(new.value_counts())
# https://docs.python.org/3/library/stdtypes.html#str.rsplit 

#Opgave 5. 
# Pivot-tabel over hvor mange passagerer på hver klasse & hvilken klasse havde fleste omkomne? 
pd.pivot_table(titanic, values=['Survived'], columns='Pclass', aggfunc='count')
# Aggregeret funktion med optælling. 
titanic.groupby(['Pclass', 'Survived'])['Survived'].count()
# Gruppering opdelt i Pclass og Survived
df = titanic[['Pclass','Survived']]
print(df.shape)
# Antal personer på inddelt i hver passagerklasse 
print(pd.pivot_table(df, values='Survived', columns='Pclass', aggfunc='count'))
# Antal overlevende inddelt i hver passagerklasse. 
print(pd.pivot_table(df, values='Survived', columns='Pclass', aggfunc='sum'))
# Kilde: https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#named-aggregation  

from nltk.probability import FreqDist
fdist = FreqDist(new.value_counts())
fdist.most_common(10)

# Vi kan visualisere vores data i python blandt andet gennem et søjlediagram og et cirkeldiagram, som ses nedenfor.

# Vi kan ændre, hvilken form vi ønsker at vores visualisering tager ved at ændre kind. For at få et søjlediagram har vi sat kind = bar og for at få et cirkeldiagram har vi sat kind = pie.

# Søjlediagram over antal overlevede pr. klasse
barchart = df['Pclass'].value_counts().plot(kind='bar', title ="Antal overlevede pr. class")

# Cirkeldiagram over antal overlevede pr. klasse
piechart = df['Pclass'].value_counts().plot(kind='pie', title ="Antal overlevede pr. class, som cirkeldiagram")
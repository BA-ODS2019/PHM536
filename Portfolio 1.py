#Opgave 1. 
#a. Hvilke data typer kan i identificere? (tekst / tal / typer af tal mv.)
	#Om passageren overlevede (angivet i 0 eller 1 - vi antager at 1 betyder personen overlevede), passagerklasse, navn, køn, alder, antal søskende/ægtefæller om bord,
	#antal forældre/børn om bord, samt hvor meget de har betalt for overfarten. 
#b. Mangler der data?
    #JA på Titanic var der ca. 1.300 passagerer var fordelt med 325 på 1. klasse, 285 på 2. klasse og 706 på 3. klasse.
    # plus ca 900 besætningsmedlemmer, dette datasæt indeholde kun 887 passagere. derfor mangler der ca. 500 passagere i dette regnestykke, foruden besætningen. 
    #Nationalitet, anvendt type af valuta til betaling, billet nr., kabine nr., hvilken havn der blev anvendt 
    #til påstigning, en bedre separation mellem søskende, ægtefæller, forældre og børn i kolonnerne.  
    #Evt. erhverv og religion. 

#Opgave 2. 
import pandas as pd
titanic = pd.read_csv("titanic.csv", sep=',')
#print(titanic)
print(titanic.head()) #viser de første 5 linier af filen for at få en forståelse af dataen
print(len(titanic)) #Antal rows
print(titanic.shape) #antal rows & columns 
print(titanic.size) #antal celler i hele filen
print(titanic.columns) #Navnene til columns
print(titanic.dtypes) #data typer af filen int, str eller float. 
#Kilde: https://datacarpentry.org/python-socialsci/08-Pandas/index.html

#Opgave 3. 
#Antal af personer der overlevede = Summen af Survived column
print('Antal af personer der overlevede:', sum(titanic.Survived))
#gennemsnits alderen på passagerene = mean af Age column
print('Gennemsnit alderen for alle passagere på Titanic:', titanic['Age'].mean())
#Median alder på passagerne = median af Age column
print('Median alderen på alle passagere på Titanic:', titanic['Age'].median())
#mindste værdi altså yngste passager = min af Age column
print('Alder på yngste passager:', titanic['Age'].min())
#max værdi ældste passager = max af Age colum
print('Alder på ældste passager:', titanic['Age'].max())
#antal af passgere på hver klasse = value_count af Pclass
print(titanic['Pclass'].value_counts())
#Gennemsnitspris pr billet. Dog er billet ikke angivet i valuta = mean af Fare column
print('Gennemsnit pris pr billet alle passagere på Titanic:', titanic['Fare'].mean())
#Kilde: https://datacarpentry.org/python-socialsci/10-aggregations/index.html

#Opgave 4. Hvor mange har samme efternavn
new = titanic['Name'].str.split(r'((\w+$))', n = 0, expand = True)
new[1].value_counts()
new = titanic['Name'].str.rsplit(pat = r'((\w+^))', expand = True)
#column name split str op, r = reverse \w er helt ord og $ er anker
new = (titanic['Name'].str.split().str[-1])
print(new.value_counts())
#https://docs.python.org/3/library/stdtypes.html#str.rsplit
#-1 betyder at der startes med den sidste karakter i string.
# Ved at lave en str.split. opdeles hvert ord seperat. Dette gør at programmet læser hvert ord individuelt og ikke en kombineret string. 

#Opgave 5. Pivot-tabel over hvor mange på hver klasse & hvilken klasse havde fleste omkomne? 
pd.pivot_table(titanic, values=['Survived'], columns='Pclass', aggfunc='count')
#aggregeret funktion med optælling. 
titanic.groupby(['Pclass', 'Survived'])['Survived'].count()
#grupering opdelt i Pclass og Survived
df = titanic[['Pclass','Survived']]
print(df.shape)
#Antal personer på inddelt i hver passagerklasse 
print(pd.pivot_table(df, values='Survived', columns='Pclass', aggfunc='count'))
#Antal overlevne inddelt i hver passagerklasse. 
print(pd.pivot_table(df, values='Survived', columns='Pclass', aggfunc='sum'))
#Kilde: https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#named-aggregation 
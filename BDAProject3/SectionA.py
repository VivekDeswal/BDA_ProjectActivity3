import warnings

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns



print("SECTION-A")

titanic = pd.read_csv(r"F:\Semester 7\titanic dataset.csv")
test_df= pd.read_csv(r"F:\Semester 7\test.csv")
print(" ")

print("1. Find out the overall chance of survival for a Titanic passenger.")
print(" ")

print("Total number of passengers survived are: ", titanic['survived'].value_counts()[1])
print("Overall chance of survival: ", titanic['survived'].value_counts(normalize=True)[1] * 100)

print(" ")
print("Q2. Find out the chance of survival for a Titanic passenger based on their sex and plot it.")
print("Graph plotted ")
print(" ")

sns.barplot(x="sex", y="survived", data=titanic)
plt.show()
print("Percentage of females survived",
      titanic["survived"][titanic["sex"] == 'female'].value_counts(normalize=True)[1] * 100)
print("Percentage of males survived",
      titanic["survived"][titanic["sex"] == 'male'].value_counts(normalize=True)[1] * 100)

print(" ")
print("Q3. Find out the chance of survival for a Titanic passenger by traveling class wise and plot it. ")
print("Graph plotted ")
print(" ")

sns.barplot(x="pclass", y="survived", data=titanic)
plt.show()
print("Percentage of Pclass 1 survived",
      titanic["survived"][titanic["pclass"] == 1].value_counts(normalize=True)[1] * 100)
print("Percentage of Pclass 2 survived",
      titanic["survived"][titanic["pclass"] == 2].value_counts(normalize=True)[1] * 100)
print("Percentage of Pclass 3 survived",
      titanic["survived"][titanic["pclass"] == 3].value_counts(normalize=True)[1] * 100)

print(" ")

print("Q4: Find out the average age for a Titanic passenger who survived by passenger class and sex. ")

print(" ")

fig = plt.figure(figsize=(12, 5))
fig.add_subplot(121)
plt.title('average age for a Titanic passenger who survived by pclass and sex')
sns.barplot(data=titanic, x='age', y='pclass', hue='sex')

meanAgeMale = round(titanic[(titanic['sex'] == "male")]['age'].groupby(titanic['pclass']).mean(), 2)
meanAgeFeMale = round(titanic[(titanic['sex'] == "female")]['age'].groupby(titanic['pclass']).mean(), 2)

print(pd.concat([meanAgeMale, meanAgeFeMale], axis=1, keys=['Male', 'Female']))

print(" ")

print(
    "Q5. Find out the chance of survival for a Titanic passenger based on number of siblings the passenger had on the "
    "ship and plot it.")
print("Graph plotted ")
print(" ")

sns.barplot(x="sibsp", y="survived", data=titanic)
plt.show()
print("Percentage of SibSp 0 who survived is",
      titanic["survived"][titanic["sibsp"] == 0].value_counts(normalize=True)[1] * 100)
print("Percentage of SibSp 1 who survived is",
      titanic["survived"][titanic["sibsp"] == 1].value_counts(normalize=True)[1] * 100)
print("Percentage of SibSp 2 who survived is",
      titanic["survived"][titanic["sibsp"] == 2].value_counts(normalize=True)[1] * 100)

print(" ")

print("Q6. Find out the chance of survival for a Titanic passenger based on number of parents/children the passenger "
      "had on the ship and plot it.")
print("Graph plotted ")
sns.barplot(x="parch", y="survived", data=titanic)
plt.show()
print("Percentage of parch 0 who survived is",
      titanic["survived"][titanic["parch"] == 0].value_counts(normalize=True)[1] * 100)
print("Percentage of parch 1 who survived is",
      titanic["survived"][titanic["parch"] == 1].value_counts(normalize=True)[1] * 100)
print("Percentage of parch 2 who survived is",
      titanic["survived"][titanic["parch"] == 2].value_counts(normalize=True)[1] * 100)
print("Percentage of parch 3 who survived is",
      titanic["survived"][titanic["parch"] == 3].value_counts(normalize=True)[1] * 100)

print(" ")
print("Q7. Plot out the variation of survival and death amongst passengers of different age. ")
print("Graph plotted ")
print(" ")

titanic["age"] = titanic["age"].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
titanic['agegroup'] = pd.cut(titanic['age'], bins, labels=labels)
sns.barplot(x="agegroup", y="survived", data=titanic)
plt.show()

g = sns.FacetGrid(titanic, col='survived')
g.map(plt.hist, 'age', bins=20)

print(" ")

print("Q8. Plot out the variation of survival and death with age amongst passengers of different passenger classes.")
print("Graph plotted ")


grid = sns.FacetGrid(titanic, col='survived', row='pclass', size=3, aspect=2)
grid.map(plt.hist, 'age', alpha=.5, bins=20)
grid.add_legend()
plt.show()


print(" ")
print("Q9. Find out the survival probability for a Titanic passenger based on title from the name of passenger.")
print(" ")

# create a combined group of both datasets
combine = [titanic, test_df]

# extract a title for each Name in the train and test datasets
for dataset in combine:
    dataset['Title'] = dataset.name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(titanic['Title'], titanic['sex'])

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',
                                                 'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

print(titanic[['Title', 'survived']].groupby(['Title'], as_index=False).mean())
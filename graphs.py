import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import csv
import seaborn as sns


def to_int(data):
	res = []
	for val in data:
		if not "age_at_creation" in val:
			res.append(int(val))

	return res

def normalize_weight(weight):
	value = [0,0,0,0,0,0]

	for w in weight:
		if int(w) < 40000:
			value[0] += 1
		elif int(w) >= 40000 and int(w) < 60000:
			value[1] += 1
		elif int(w) >= 60000 and int(w) < 80000:
			value[2] += 1
		elif int(w) >= 80000 and int(w) < 100000:
			value[3] += 1
		elif int(w) >= 100000 and int(w) < 120000:
			value[4] += 1
		else:
			value[5] += 1
	return value

def normalize_height(height):
	value = [0,0,0,0,0,0,0,0]

	for w in height:
		if int(w) < 140:
			value[0] += 1
		elif int(w) >= 140 and int(w) < 160:
			value[1] += 1
		elif int(w) >= 160 and int(w) < 170:
			value[2] += 1
		elif int(w) >= 170 and int(w) < 180:
			value[3] += 1
		elif int(w) >= 180 and int(w) < 190:
			value[4] += 1
		elif int(w) >= 190 and int(w) < 200:
			value[5] += 1
		elif int(w) >= 200 and int(w) < 220:
			value[6] += 1
		else:
			value[7] += 1
	return value

def normalize_body(bodytype):
	value = [0,0,0,0,0,0,0,0]

	for w in bodytype:
		if int(w) == 1:
			value[0] += 1
		elif int(w) == 2:
			value[1] += 1
		elif int(w) == 3:
			value[2] += 1
		elif int(w) == 4:
			value[3] += 1
		elif int(w) == 5:
			value[4] += 1
		elif int(w) == 6:
			value[5] += 1
		elif int(w) == 7:
			value[6] += 1
		else:
			value[7] += 1
	return value

def normalize_account_creation(created_on):
	value = []
	for date in created_on:
		value.append(str(date)[0:4])
	return value

def normalize_drink(drink):
	value = []
	for rel in drink:
		if rel != 'NULL':
			if int(rel) == 1:
				value.append('Nunca')
			elif int(rel) == 2 :
				value.append('Ocasionalmente')
			elif int(rel) == 3:
				value.append('Regularmente')
			else :
				value.append('Não especificado')
		else:
			value.append('Não especificado')
	return value


created_on = []
updated_on = []
city = []
state = []
country = []
gender = []
dob = []
weight = []
height = []
bodytype = []
smoke = []
drink = []
initially_seeking =[]
relationship = []
open_to = []
turns_on = []
looking_for = []
ethnicity = []
age = []

with open("C://Users//jpms2//Desktop//data-science//data-science//ashley_madison.csv",'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
     	created_on.append(row[0])
     	updated_on.append(row[2])
     	city.append(row[7])
     	state.append(row[8])
     	country.append(row[9])
     	gender.append(row[10])
     	dob.append(row[11])
     	weight.append(row[13])
     	height.append(row[14])
     	bodytype.append(row[15])
     	drink.append(row[17])
     	relationship.append(row[19])
     	open_to.append(row[20])
     	turns_on.append(row[21])
     	looking_for.append(row[22])

with open("C://Users//jpms2//Desktop//data-science//data-science//ashley_madison_preprocessed.csv",'r',encoding="utf8") as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        ethnicity.append(row[11])
        smoke.append(row[13])
        initially_seeking.append(row[14])
        age.append(row[16])
# bar charts

# weight chart

objects = ('Abaixo de 40kg', '40kg ~ 60kg', '60kg ~ 80kg', '80kg ~ 100kg', '100kg ~ 120kg', 'Acima de 120kg')
y_pos = np.arange(len(objects))
 
plt.bar(y_pos, normalize_weight(weight[1:-1]), align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Número de usuários')
plt.title('Usuarios por peso')
 
plt.show()


#height chart

objects = ('Abaixo de 1,40m', '1,40m ~ 1,60m', '1,60m ~ 1,70m', '1,70m ~ 1,80m', '1,80m ~ 1,90m', '1,90m ~ 2,00m', '2,00m ~ 2,20m', 'Acima de 2,20m')
y_pos = np.arange(len(objects))
 
plt.bar(y_pos, normalize_height(height[1:-1]), align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Número de usuários')
plt.title('Usuarios por altura')
 
plt.show()


#body type chart

objects = ('Magro', 'Em forma', 'Musculoso', 'Médio', 'Torneado', 'Acima do peso', 'Obeso', 'Voluptuoso')
y_pos = np.arange(len(objects))
 
plt.bar(y_pos, normalize_body(bodytype[1:-1]), align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Número de usuários')
plt.title('Usuarios por tipo de corpo')
 
plt.show()



""" histograms """

# ano de adesão
plt.hist(normalize_account_creation(created_on[1:-1]), rwidth=1, bins='auto')
plt.ylabel('Quantidade');
plt.title('Adesão')

plt.show()

# bebe
plt.hist(normalize_drink(drink[1:-1]), rwidth=0.25, bins=5)
plt.ylabel('Quantidade')
plt.title('Frequencia que bebe')

plt.show()

# fuma
plt.hist(smoke[1:-1], rwidth=1, bins='auto')
plt.ylabel('Quantidade')
plt.title('Frequencia que fuma')

plt.show()

# etnia
plt.hist(ethnicity[1:-1], rwidth=0.25, bins=9)
plt.ylabel('Quantidade')
plt.title('Etnia')

plt.show()

# inicialmente buscando
plt.hist(initially_seeking[1:-1], rwidth=0.25, bins=5)
plt.ylabel('Quantidade')
plt.title('Tipo de relacionamento')

plt.show()

""" density """
data = age[1:-1]
sns.set_style('whitegrid')
sns.kdeplot(np.array(data), bw=0.5)
plt.show()

""" boxplot """
"""
spread = np.random.rand(50) * 100
center = np.ones(25) * 50
flier_high = np.random.rand(10) * 100 + 100
flier_low = np.random.rand(10) * -100
data = np.concatenate((spread, center, flier_high, flier_low), 0)
plt.boxplot(data)
#plt.show()
"""
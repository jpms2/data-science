import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.stats import gaussian_kde


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

def normalize_relationship(initially_seeking):
	value = []
	for rel in initially_seeking:
		if int(rel) == 1:
			value.append("mulher em relação procurando homens")
		elif int(rel) == 2 :
			value.append("Homem em relação procurando mulheres")
		elif int(rel) == 3:
			value.append("Solteiros procurando mulheres")
		elif int(rel) == 4:
			value.append("Solteiras procurando homens")
		elif int(rel) == 5:
			value.append("Homem procurando homens")
		else:
			value.append("Mulheres procurando mulheres")
	return value

def normalize_account_creation(created_on):
	value = []
	for date in created_on:
		value.append(str(date)[0:4])
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

with open(,'r') as csvfile:
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
     	smoke.append(row[16])
     	drink.append(row[17])
     	initially_seeking.append(row[18])
     	relationship.append(row[19])
     	open_to.append(row[20])
     	turns_on.append(row[21])
     	looking_for.append(row[22])

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

# inicialmente buscando
plt.hist(normalize_relationship(initially_seeking[1:-1]), normed=True, bins=6)
plt.ylabel('Probabilidade');
plt.title('Procura por gênero')

plt.show()

# ano de adesão
plt.hist(normalize_account_creation(created_on[1:-1]), normed=True, bins=6)
plt.ylabel('Probabilidade');
plt.title('Adesão')

plt.show()

# status de membro

""" density """
data = [1.5]*7 + [2.5]*2 + [3.5]*8 + [4.5]*3 + [5.5]*1 + [6.5]*8
density = gaussian_kde(data)
xs = np.linspace(0,8,200)
density.covariance_factor = lambda : .25
density._compute_covariance()
plt.plot(xs,density(xs))
#plt.show()

""" boxplot """
spread = np.random.rand(50) * 100
center = np.ones(25) * 50
flier_high = np.random.rand(10) * 100 + 100
flier_low = np.random.rand(10) * -100
data = np.concatenate((spread, center, flier_high, flier_low), 0)
plt.boxplot(data)
#plt.show()
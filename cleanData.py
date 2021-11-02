import pandas as pd
import category_encoders as ce
import seaborn as sn
import eli5
import matplotlib.pyplot as plt
from eli5.sklearn import PermutationImportance
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from IPython.display import display
data = pd.read_excel('Copia de Base de datos prueba tecnica.xlsx',sheet_name='DB')
#Iprimimos la la longitud de los datos originales
print("El número de registros originalmente es " + str(len(data)))
#Quitamos los duplicados de los datos
data = data.drop_duplicates()
print("El número de registros sin duplicados  es " + str(len(data)))
df = data.groupby('cliente_id').first()
print("El número de registros únicos sin ambiguedades  es " + str(len(df)))
print(df.head())

########################################################################################################################
####################################Llenamos los valores nulos con 0################################################
df = df.fillna(0)


##########################################listamos  las columnas categóricas  y las numéricas###########################

num_cols = list(df._get_numeric_data().columns)
cat_cols = list(set(df.columns) - set(num_cols))


############################################################################################################################
###########################################Definimos el problema X--->y#####################################################

y = df['Incumplimiento_pago']
X = df.drop(['Incumplimiento_pago'], axis= 1)

########################################################################################################################
#########################################Grafica del inbalance##########################################################
df.Incumplimiento_pago.value_counts().plot.pie(y='label', title='Proporción de cada clase')
plt.show()

features = ['estrato','CANAL_HOMOLOGADO_MILLICON','productos','REGIONAL','portafolio']

for f in features:
	datos = df[[f, 'Incumplimiento_pago']].groupby([f]).sum()
	labels = datos.index.values.tolist()
	cantidad = list(datos.values.tolist())
	fig1, ax1 = plt.subplots()
	ax1.pie(cantidad, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
	ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
	plt.title('Cantidad de no pagos por ' + f)
	#plt.show()


##########################################################################################################################
###########################################Alistamos los datos que se van a meter al modelo ##############################

encoder = ce.TargetEncoder(cols=cat_cols)
X = encoder.fit_transform(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
########################################################################################################################
###########################################Definimos el modelo que se va a usar#########################################

lr = LogisticRegression()

steps = [('over', SMOTE()), ('model', lr)]
pipeline = Pipeline(steps=steps)
pipeline.fit(X_train, y_train)
y_hat = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X)
y_prob = pd.DataFrame(y_prob)
y_prob.to_csv('prob.csv')
coef = lr.coef_
coef = pd.DataFrame(coef)
coef.columns = X.columns
coef = coef.transpose()
coef = coef.abs()
coef = coef.sort_values(by=[0], ascending= False)
print(coef)
print(classification_report(y_test, y_hat))


########################################################################################################################
#########################################Redes Neuronales ##############################################################


def create_baseline():
	# create model
	model = Sequential()
	model.add(Dense(30, input_dim=20, activation='relu'))
	model.add(Dense(30), activation ='relu')
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)
steps = [('over', SMOTE()), ('model', estimator)]
pipeline = Pipeline(steps=steps)
pipeline.fit(X_train, y_train)
y_hat = pipeline.predict(X_test)
print(classification_report(y_test, y_hat))
con = confusion_matrix(y_test, y_hat)
print(con)
df_cm = pd.DataFrame(con, index = [i for i in "AB"],
                  columns = [i for i in "AB"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
plt.show()
perm = PermutationImportance(estimator, random_state=1).fit(X_train,y_train)
display(eli5.show_weights(perm, feature_names = X_train.columns.tolist()))
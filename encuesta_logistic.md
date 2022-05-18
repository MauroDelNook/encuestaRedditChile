# Conexión a fuente de datos


```python
import numpy as np
import pandas as pd
```


```python
df = pd.read_pickle("df_encuesta.pkl")
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1171 entries, 0 to 1181
    Data columns (total 56 columns):
     #   Column                                            Non-Null Count  Dtype
    ---  ------                                            --------------  -----
     0   edad                                              1171 non-null   int32
     1   religion                                          1171 non-null   int8 
     2   sexo_Hombre                                       1171 non-null   uint8
     3   sexo_Mujer                                        1171 non-null   uint8
     4   sexo_Otro                                         1171 non-null   uint8
     5   orientacion_Bisexual                              1171 non-null   uint8
     6   orientacion_Heterosexual                          1171 non-null   uint8
     7   orientacion_Homosexual                            1171 non-null   uint8
     8   orientacion_NS/NC                                 1171 non-null   uint8
     9   orientacion_Otro                                  1171 non-null   uint8
     10  ocupacion_Estudio                                 1171 non-null   uint8
     11  ocupacion_Otro                                    1171 non-null   uint8
     12  ocupacion_Trabajo                                 1171 non-null   uint8
     13  ocupacion_Trabajo;Estudio                         1171 non-null   uint8
     14  region_Antofagasta                                1171 non-null   uint8
     15  region_Araucanía                                  1171 non-null   uint8
     16  region_Arica y Parinacota                         1171 non-null   uint8
     17  region_Atacama                                    1171 non-null   uint8
     18  region_Aysén del General Carlos Ibañez del Campo  1171 non-null   uint8
     19  region_Biobio                                     1171 non-null   uint8
     20  region_Coquimbo                                   1171 non-null   uint8
     21  region_Fuera del país                             1171 non-null   uint8
     22  region_Libertador General Bernardo O’Higgins      1171 non-null   uint8
     23  region_Los Lagos                                  1171 non-null   uint8
     24  region_Los Ríos                                   1171 non-null   uint8
     25  region_Magallanes y de la Antártica Chilena       1171 non-null   uint8
     26  region_Región Metropolitana de Santiago           1171 non-null   uint8
     27  region_Tarapacá                                   1171 non-null   uint8
     28  region_Valparaíso                                 1171 non-null   uint8
     29  region_del Maule                                  1171 non-null   uint8
     30  region_Ñuble                                      1171 non-null   uint8
     31  educacion_media                                   1171 non-null   uint8
     32  educacion_otro                                    1171 non-null   uint8
     33  educacion_posgrado                                1171 non-null   uint8
     34  educacion_superior                                1171 non-null   uint8
     35  politica_izquierda                                1171 non-null   uint8
     36  musica_Metal                                      1171 non-null   uint8
     37  musica_NS/NC                                      1171 non-null   uint8
     38  musica_Otro                                       1171 non-null   uint8
     39  musica_Pop                                        1171 non-null   uint8
     40  musica_Rap                                        1171 non-null   uint8
     41  musica_Rock                                       1171 non-null   uint8
     42  deporte_1hora                                     1171 non-null   uint8
     43  deporte_2horas                                    1171 non-null   uint8
     44  deporte_4horas                                    1171 non-null   uint8
     45  deporte_No                                        1171 non-null   uint8
     46  perro_Atom y Humber                               1171 non-null   uint8
     47  perro_Chilaquil                                   1171 non-null   uint8
     48  perro_El perro watón                              1171 non-null   uint8
     49  perro_NS/NC                                       1171 non-null   uint8
     50  perro_Perro ladrón de empanadas                   1171 non-null   uint8
     51  perro_Perro lipigas                               1171 non-null   uint8
     52  perro_Perro matapacos                             1171 non-null   uint8
     53  perro_Perro que dice "agua"                       1171 non-null   uint8
     54  perro_Perro que viaja en bus                      1171 non-null   uint8
     55  perro_Washington                                  1171 non-null   uint8
    dtypes: int32(1), int8(1), uint8(54)
    memory usage: 76.6 KB
    


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>edad</th>
      <th>religion</th>
      <th>sexo_Hombre</th>
      <th>sexo_Mujer</th>
      <th>sexo_Otro</th>
      <th>orientacion_Bisexual</th>
      <th>orientacion_Heterosexual</th>
      <th>orientacion_Homosexual</th>
      <th>orientacion_NS/NC</th>
      <th>orientacion_Otro</th>
      <th>...</th>
      <th>perro_Atom y Humber</th>
      <th>perro_Chilaquil</th>
      <th>perro_El perro watón</th>
      <th>perro_NS/NC</th>
      <th>perro_Perro ladrón de empanadas</th>
      <th>perro_Perro lipigas</th>
      <th>perro_Perro matapacos</th>
      <th>perro_Perro que dice "agua"</th>
      <th>perro_Perro que viaja en bus</th>
      <th>perro_Washington</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>27</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>30</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>18</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>19</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 56 columns</p>
</div>




```python
X = df.loc[:, ~df.columns.isin(['politica_izquierda'])]
y = df['politica_izquierda']
```


```python
#creemos ahora muestras de entrenamiento y prueba seleccionadas aleatoriamente:
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, 
                                                    random_state = 42)
```


```python
# balaceo con sobremuestro
y_train_Q = y_train[y_train==1]
y_train_NQ = y_train[y_train==0]
y_train_Q = y_train_Q.sample(len(y_train_NQ), replace=True, random_state= 42)
y_train = pd.concat([y_train_Q,y_train_NQ],axis=0)
y_train.value_counts()
X_train = X_train.loc[y_train.index,:]

del [y_train_Q, y_train_NQ]
```


```python
# Grid search cross validation
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

grid = {'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000]}
logreg = LogisticRegression(max_iter = 10000, random_state = 42, n_jobs = 2)
logreg_cv = GridSearchCV(logreg,grid,cv= 10, n_jobs = 2, scoring = 'accuracy')
logreg_cv.fit(X_train, y_train)

print("tuned hyperparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)
# https://www.kaggle.com/code/enespolat/grid-search-with-logistic-regression/notebook
```

    tuned hyperparameters :(best parameters)  {'C': 1000}
    accuracy : 0.6069187675070028
    


```python
modelo = logreg_cv.best_estimator_
```


```python
modelo.fit(X_train,y_train)
```




    LogisticRegression(C=1000, max_iter=10000, n_jobs=2, random_state=42)




```python
y_pred_train = modelo.predict(X_train)
y_pred_train = pd.DataFrame(y_pred_train, columns=["Y_predicha"],index = y_train.index)

from sklearn.metrics import classification_report
print("Resultado en Muestra de Entrenamiento:")

print(classification_report(y_train, y_pred_train))
```

    Resultado en Muestra de Entrenamiento:
                  precision    recall  f1-score   support
    
               0       0.67      0.66      0.67       421
               1       0.67      0.68      0.67       421
    
        accuracy                           0.67       842
       macro avg       0.67      0.67      0.67       842
    weighted avg       0.67      0.67      0.67       842
    
    


```python
y_pred_test = modelo.predict(X_test)
y_pred_test = pd.DataFrame(y_pred_test, columns=["Y_predicha"],index = y_test.index)

print("Resultado en Muestra de Entrenamiento:")

print(classification_report(y_test, y_pred_test))
```

    Resultado en Muestra de Entrenamiento:
                  precision    recall  f1-score   support
    
               0       0.67      0.72      0.69       116
               1       0.70      0.66      0.68       119
    
        accuracy                           0.69       235
       macro avg       0.69      0.69      0.68       235
    weighted avg       0.69      0.69      0.68       235
    
    


```python
pd.DataFrame(modelo.coef_, columns=X.columns)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>edad</th>
      <th>religion</th>
      <th>sexo_Hombre</th>
      <th>sexo_Mujer</th>
      <th>sexo_Otro</th>
      <th>orientacion_Bisexual</th>
      <th>orientacion_Heterosexual</th>
      <th>orientacion_Homosexual</th>
      <th>orientacion_NS/NC</th>
      <th>orientacion_Otro</th>
      <th>...</th>
      <th>perro_Atom y Humber</th>
      <th>perro_Chilaquil</th>
      <th>perro_El perro watón</th>
      <th>perro_NS/NC</th>
      <th>perro_Perro ladrón de empanadas</th>
      <th>perro_Perro lipigas</th>
      <th>perro_Perro matapacos</th>
      <th>perro_Perro que dice "agua"</th>
      <th>perro_Perro que viaja en bus</th>
      <th>perro_Washington</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.004795</td>
      <td>-0.634486</td>
      <td>-0.511496</td>
      <td>-0.670882</td>
      <td>0.176358</td>
      <td>1.265978</td>
      <td>0.488918</td>
      <td>1.710306</td>
      <td>-5.188261</td>
      <td>0.717038</td>
      <td>...</td>
      <td>-1.625029</td>
      <td>0.0</td>
      <td>0.109279</td>
      <td>-0.469773</td>
      <td>0.065528</td>
      <td>-0.637474</td>
      <td>1.490684</td>
      <td>0.383817</td>
      <td>0.163619</td>
      <td>-0.486671</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 55 columns</p>
</div>



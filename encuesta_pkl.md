# Conexión a fuente de datos


```python
import numpy as np
import pandas as pd
```


```python
df = pd.read_csv('D:\Dropbox\DataScience\Encuesta\Encuesta anual no oficial (todavía) de r\chile 2021.csv')
```

# Limpieza de datos


```python
# seleccionamos variables de interés
df = df.iloc[:, [1, 2, 3, 4, 5, 8, 10, 13, 16, 21, 25]]
df.columns = ['edad', 'sexo', 'orientacion', 'ocupacion', 'region', 'educacion',
              'politica', 'musica', 'deporte', 'religion', 'perro']
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1182 entries, 0 to 1181
    Data columns (total 11 columns):
     #   Column       Non-Null Count  Dtype 
    ---  ------       --------------  ----- 
     0   edad         1182 non-null   object
     1   sexo         1182 non-null   object
     2   orientacion  1180 non-null   object
     3   ocupacion    1182 non-null   object
     4   region       1182 non-null   object
     5   educacion    1182 non-null   object
     6   politica     1148 non-null   object
     7   musica       1051 non-null   object
     8   deporte      1182 non-null   object
     9   religion     1182 non-null   object
     10  perro        1150 non-null   object
    dtypes: object(11)
    memory usage: 101.7+ KB
    

Al tratarse en su mayoría de variables categóricas, iremos campo a campo limpiando formatos de respuesta, tratando nulos y agrupando datos.


```python
df['sexo'].value_counts()
```




    Hombre                      995
    Mujer                       150
    Otro/Prefiero no decirlo     24
    pudú                          1
    Non-Binary                    1
    Helicoptero apache            1
    no binario                    1
    mareaqueta                    1
    Q                             1
    Helicóptero apache            1
    ya                            1
    Mujer Trans                   1
    No binarie                    1
    Helicóptero Cobra             1
    Poring Esponjosito            1
    Género fluido                 1
    Name: sexo, dtype: int64




```python
df.loc[(df['sexo'] != 'Hombre') & (df['sexo'] != 'Mujer'), 'sexo'] = 'Otro'
```


```python
df['orientacion'].value_counts()
```




    Heterosexual                911
    Bisexual                    149
    Homosexual                   65
    Otra/Prefiero no decirlo     55
    Name: orientacion, dtype: int64




```python
df.loc[df['orientacion'] == 'Otra/Prefiero no decirlo', 'orientacion'] = 'Otro'
df['orientacion'].fillna('Otro', inplace = True)
```


```python
df['ocupacion'].value_counts()
```




    Estudio                                                                                                      488
    Trabajo                                                                                                      453
    Trabajo;Estudio                                                                                              151
    Ninguna                                                                                                       50
    Trabajo;Ninguna                                                                                                3
    Estudio;Ninguna                                                                                                3
    busco pega                                                                                                     2
    Trabajo;Acabo de terminar la pdt                                                                               1
    Realizando practica                                                                                            1
    Servicio cliente/soporte remoto                                                                                1
    Jugar lol mientras busco práctica                                                                              1
    Ninguna;Terminé recién el colegio y ahora voy a buscar pega                                                    1
    Cesante                                                                                                        1
    Acabo de egresar                                                                                               1
    Estudio PDT y en 2 semanas trabajo                                                                             1
    Crianza                                                                                                        1
    Independiente                                                                                                  1
    Trabajo;Estudio;Ser padre                                                                                      1
    Egresado en proceso de Tesis                                                                                   1
    Estudio;ayudantías, clases, etc                                                                                1
    Estudio;buy the dip                                                                                            1
    Trabajo;juego video juegos y veo memes                                                                         1
    Egresado, buscando trabajo                                                                                     1
    Maraquear                                                                                                      1
    Trabajo;Estudio;Ninguna                                                                                        1
    Trabajo;Combato el crimen por las noches                                                                       1
    Esto es para la opción de arriba, nada porque recientemente egresé del colegio                                 1
    Estudio de idiomas previo a estudios universitarios                                                            1
    Estudio;Buscando trabajo                                                                                       1
    Estudio;Buscando Trabajo                                                                                       1
    Q                                                                                                              1
    Trabajo;Estudio;Tesista                                                                                        1
    Empiezo un emprendimiento                                                                                      1
    Dueña de casa                                                                                                  1
    A sobrevivir en este mundo cruel                                                                               1
    Trabajo;Disfruto                                                                                               1
    Trabajo;Paja                                                                                                   1
    pudú                                                                                                           1
    Por la pandemia no pude encontrar pega, así que pinto, para ganarme unas luquitas para poner en la casa.       1
    Name: ocupacion, dtype: int64




```python
df.loc[(df['ocupacion'] != 'Estudio') & (df['ocupacion'] != 'Trabajo') &
       (df['ocupacion'] != 'Trabajo;Estudio') & (df['ocupacion'] != 'Ninguna'), 'ocupacion'] = 'Otro'
```


```python
df['region'].value_counts()
```




    Región Metropolitana de Santiago             609
    Valparaíso                                   128
    Biobio                                        82
    Fuera del país                                52
    Libertador General Bernardo O’Higgins         46
    Coquimbo                                      42
    Araucanía                                     39
    Los Lagos                                     38
    del Maule                                     34
    Antofagasta                                   28
    Los Ríos                                      24
    Ñuble                                         19
    Atacama                                       12
    Tarapacá                                      10
    Aysén del General Carlos Ibañez del Campo      9
    Magallanes y de la Antártica Chilena           6
    Arica y Parinacota                             4
    Name: region, dtype: int64




```python
df['region'].replace('Región Metropolitana de Santiago',
                        'RM', inplace=True)
df['region'].replace('Valparaíso', 'V', inplace=True)
df['region'].replace('Biobio', 'VIII', inplace=True)
df['region'].replace('Fuera del país', 'Extranjero', inplace=True)
df['region'].replace('Libertador General Bernardo O’Higgins', 'VI', 
                     inplace=True)
df['region'].replace('Coquimbo', 'IV', inplace=True)
df['region'].replace('Araucanía', 'IX', inplace=True)
df['region'].replace('Los Lagos', 'X', inplace=True)
df['region'].replace('del Maule', 'VII', inplace=True)
df['region'].replace('Antofagasta', 'II', inplace=True)
df['region'].replace('Los Ríos', 'XIV', inplace=True)
df['region'].replace('Ñuble', 'XVI', inplace=True)
df['region'].replace('Atacama', 'III', inplace=True)
df['region'].replace('Tarapacá', 'I', inplace=True)
df['region'].replace('Aysén del General Carlos Ibañez del Campo', 'XI', 
                     inplace=True)
df['region'].replace('Magallanes y de la Antártica Chilena', 'XII', 
                     inplace=True)
df['region'].replace('Arica y Parinacota', 'XV', inplace=True)
```


```python
df['educacion'].value_counts()
```




    Educación superior (Universidad, CFT, IP, etc)                       868
    Educación media                                                      172
    Post título (Magister, PhD, post doc)                                133
    Educación básica                                                       6
    Pudú                                                                   1
    Q                                                                      1
    Educacion "informal"(cursos online). Educación media formalmente.      1
    Name: educacion, dtype: int64




```python
df['educacion'].replace('Educación superior (Universidad, CFT, IP, etc)',
                        'superior', inplace=True)
df['educacion'].replace('Educación media', 'media', inplace=True)
df['educacion'].replace('Post título (Magister, PhD, post doc)', 'posgrado',
                        inplace=True)
df.loc[(df['educacion'] != 'superior') & (df['educacion'] != 'media') & 
       (df['educacion'] != 'posgrado'), 'educacion'] = 'otro'
```


```python
df['politica'].value_counts()
```




    Hacia la izquierda    638
    Centro                318
    Hacia la derecha      192
    Name: politica, dtype: int64




```python
df['politica'].replace('Hacia la izquierda', 'izquierda', inplace=True)
df['politica'].replace('Hacia la derecha', 'derecha', inplace=True)
df['politica'].replace('Centro', 'centro', inplace=True)
df['politica'].fillna('NS_NC', inplace = True)
```


```python
df['musica'].value_counts()
```




    Rock                   215
    Metal                   59
    Pop                     58
    rock                    38
    Rap                     23
                          ... 
    post rock                1
    drum n bass              1
    no tengo                 1
    Música clásica rusa      1
    Música Clásica           1
    Name: musica, Length: 342, dtype: int64




```python
df['musica'].fillna('NS_NC', inplace=True)
df.loc[df['musica'].str.contains('rock', case=False), 'musica'] = 'Rock'
df.loc[df['musica'].str.contains('metal', case=False), 'musica'] = 'Metal'
df.loc[df['musica'].str.contains('rap', case=False), 'musica'] = 'Rap'
df.loc[df['musica'].str.contains('pop', case=False), 'musica'] = 'Pop'
df.loc[(df['musica'] != 'Rock') & (df['musica'] != 'Metal') & 
       (df['musica'] != 'Rap') & (df['musica'] != 'Pop'), 'musica'] = 'Otro'
```


```python
df['deporte'].value_counts()
```




    No                                  466
    Sí, al menos 1 hora a la semana     267
    Sí, al menos 2 horas a la semana    244
    Sí, 4 horas o más                   205
    Name: deporte, dtype: int64




```python
df['deporte'].replace('Sí, al menos 1 hora a la semana', '1hora', inplace=True)
df['deporte'].replace('Sí, al menos 2 horas a la semana', '2horas', 
                      inplace=True)
df['deporte'].replace('Sí, 4 horas o más', '4horas', inplace=True)
```


```python
df['religion'].value_counts()
```




    Ninguna                                                                                                                                              944
    Cristianismo                                                                                                                                         166
    Budismo                                                                                                                                               14
    Pastafarismo                                                                                                                                           8
    Judaísmo                                                                                                                                               4
    Puduismo                                                                                                                                               2
    Ateísmo                                                                                                                                                2
    Agnóstico                                                                                                                                              2
    Jedi                                                                                                                                                   2
    Agnosticismo                                                                                                                                           1
    Deísta                                                                                                                                                 1
    niuna wea                                                                                                                                              1
    Discordianismo                                                                                                                                         1
    no                                                                                                                                                     1
    agnostico                                                                                                                                              1
    Ateo eseptico                                                                                                                                          1
    Rugankomunismo                                                                                                                                         1
    Gnosticismo                                                                                                                                            1
    Católico Ortodoxo                                                                                                                                      1
    pastafari                                                                                                                                              1
    iglesia maradoniana                                                                                                                                    1
    El Templo Satánico                                                                                                                                     1
    Islam                                                                                                                                                  1
    puduismo                                                                                                                                               1
    Satanismo Ateo                                                                                                                                         1
    Luterano                                                                                                                                               1
    wicca                                                                                                                                                  1
    Deismo                                                                                                                                                 1
    Hinduísmo                                                                                                                                              1
    Q                                                                                                                                                      1
    Satanismo (The Satanic Temple)                                                                                                                         1
    Agnostico, criado catolico                                                                                                                             1
    No tengo religión pero me interesa muchísimo el tema espiritual, recojo enseñanzas del budismo y el hinduismo pero ninguna me identifica del todo      1
    Pastafari                                                                                                                                              1
    Puducracia                                                                                                                                             1
    Catolicismo                                                                                                                                            1
    Espiritualidad                                                                                                                                         1
    Soy Agnostico/Ateo, pero no de los pasao a caca que le tiran mierda al cristianismo                                                                    1
    Agnostico                                                                                                                                              1
    Mormonismo (no me gusta el término pero es lo que se me ocurrió)                                                                                       1
    Pucha toy entre ser canuto y agnóstico                                                                                                                 1
    Deísmo                                                                                                                                                 1
    Satanismo                                                                                                                                              1
    Ateismo                                                                                                                                                1
    Flying Spaghetti Monster                                                                                                                               1
    De todo un poco?                                                                                                                                       1
    pagana                                                                                                                                                 1
    Name: religion, dtype: int64




```python
# dado que la mayoría de los encuestados se declara sin religión, 
# mejor enfocar la respuesta en si tiene o no religión.
df.loc[df['religion'] == 'Ninguna', 'religion'] = 0
df.loc[df['religion'] != 0, 'religion'] = 1
df['religion'] = df['religion'].astype('int8')
```


```python
df['perro'].value_counts()
```




    Perro ladrón de empanadas    509
    El perro watón               207
    Perro lipigas                157
    Perro matapacos              133
    Perro que viaja en bus        56
    Perro que dice "agua"         55
    Washington                    23
    Atom y Humber                  8
    Chilaquil                      2
    Name: perro, dtype: int64




```python
df['perro'].fillna('NS_NC', inplace=True)
df['perro'].replace('Perro ladrón de empanadas', 'ladron', inplace=True)
df['perro'].replace('El perro watón', 'waton', inplace=True)
df['perro'].replace('Perro lipigas', 'lipigas', inplace=True)
df['perro'].replace('Perro matapacos', 'matapacos', inplace=True)
df['perro'].replace('Perro que viaja en bus', 'bus', inplace=True)
df['perro'].replace('Perro que dice "agua"', 'agua', inplace=True)
df['perro'].replace('Washington', 'washington', inplace=True)
df['perro'].replace('Atom y Humber', 'atom_humber', inplace=True)
df['perro'].replace('Chilaquil', 'chilaquil', inplace=True)
```


```python
df['edad'] = pd.to_numeric(df['edad'], errors='coerce')
df.dropna(inplace = True)
df['edad'] = df['edad'].astype('int')
```


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
      <th>sexo</th>
      <th>orientacion</th>
      <th>ocupacion</th>
      <th>region</th>
      <th>educacion</th>
      <th>politica</th>
      <th>musica</th>
      <th>deporte</th>
      <th>religion</th>
      <th>perro</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>27</td>
      <td>Hombre</td>
      <td>Bisexual</td>
      <td>Trabajo</td>
      <td>RM</td>
      <td>superior</td>
      <td>centro</td>
      <td>Otro</td>
      <td>1hora</td>
      <td>0</td>
      <td>ladron</td>
    </tr>
    <tr>
      <th>1</th>
      <td>30</td>
      <td>Hombre</td>
      <td>Heterosexual</td>
      <td>Trabajo</td>
      <td>RM</td>
      <td>posgrado</td>
      <td>izquierda</td>
      <td>Rock</td>
      <td>2horas</td>
      <td>0</td>
      <td>ladron</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26</td>
      <td>Hombre</td>
      <td>Heterosexual</td>
      <td>Trabajo</td>
      <td>VII</td>
      <td>superior</td>
      <td>izquierda</td>
      <td>Otro</td>
      <td>No</td>
      <td>0</td>
      <td>ladron</td>
    </tr>
    <tr>
      <th>3</th>
      <td>18</td>
      <td>Hombre</td>
      <td>Heterosexual</td>
      <td>Estudio</td>
      <td>RM</td>
      <td>media</td>
      <td>derecha</td>
      <td>Otro</td>
      <td>4horas</td>
      <td>1</td>
      <td>ladron</td>
    </tr>
    <tr>
      <th>4</th>
      <td>19</td>
      <td>Hombre</td>
      <td>Heterosexual</td>
      <td>Estudio</td>
      <td>VII</td>
      <td>superior</td>
      <td>centro</td>
      <td>Rock</td>
      <td>1hora</td>
      <td>0</td>
      <td>waton</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1171 entries, 0 to 1181
    Data columns (total 11 columns):
     #   Column       Non-Null Count  Dtype 
    ---  ------       --------------  ----- 
     0   edad         1171 non-null   int32 
     1   sexo         1171 non-null   object
     2   orientacion  1171 non-null   object
     3   ocupacion    1171 non-null   object
     4   region       1171 non-null   object
     5   educacion    1171 non-null   object
     6   politica     1171 non-null   object
     7   musica       1171 non-null   object
     8   deporte      1171 non-null   object
     9   religion     1171 non-null   int8  
     10  perro        1171 non-null   object
    dtypes: int32(1), int8(1), object(9)
    memory usage: 97.2+ KB
    


```python
df = pd.get_dummies(df)
```


```python
# Variables X: eliminemos una categoría por clase 
# (para tener categría k-1 dummies)
df = df.loc[:, ~df.columns.isin(['sexo_Otro', 
                                 'orientacion_Otro', 
                                 'ocupacion_Otro', 
                                 'region_Extranjero',
                                 'educacion_otro',
                                 'musica_Otro', 
                                 'deporte_No', 
                                 'perro_NS_NC'
                                ])]
# religión ya estaba ok.
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1171 entries, 0 to 1181
    Data columns (total 50 columns):
     #   Column                     Non-Null Count  Dtype
    ---  ------                     --------------  -----
     0   edad                       1171 non-null   int32
     1   religion                   1171 non-null   int8 
     2   sexo_Hombre                1171 non-null   uint8
     3   sexo_Mujer                 1171 non-null   uint8
     4   orientacion_Bisexual       1171 non-null   uint8
     5   orientacion_Heterosexual   1171 non-null   uint8
     6   orientacion_Homosexual     1171 non-null   uint8
     7   ocupacion_Estudio          1171 non-null   uint8
     8   ocupacion_Ninguna          1171 non-null   uint8
     9   ocupacion_Trabajo          1171 non-null   uint8
     10  ocupacion_Trabajo;Estudio  1171 non-null   uint8
     11  region_I                   1171 non-null   uint8
     12  region_II                  1171 non-null   uint8
     13  region_III                 1171 non-null   uint8
     14  region_IV                  1171 non-null   uint8
     15  region_IX                  1171 non-null   uint8
     16  region_RM                  1171 non-null   uint8
     17  region_V                   1171 non-null   uint8
     18  region_VI                  1171 non-null   uint8
     19  region_VII                 1171 non-null   uint8
     20  region_VIII                1171 non-null   uint8
     21  region_X                   1171 non-null   uint8
     22  region_XI                  1171 non-null   uint8
     23  region_XII                 1171 non-null   uint8
     24  region_XIV                 1171 non-null   uint8
     25  region_XV                  1171 non-null   uint8
     26  region_XVI                 1171 non-null   uint8
     27  educacion_media            1171 non-null   uint8
     28  educacion_posgrado         1171 non-null   uint8
     29  educacion_superior         1171 non-null   uint8
     30  politica_NS_NC             1171 non-null   uint8
     31  politica_centro            1171 non-null   uint8
     32  politica_derecha           1171 non-null   uint8
     33  politica_izquierda         1171 non-null   uint8
     34  musica_Metal               1171 non-null   uint8
     35  musica_Pop                 1171 non-null   uint8
     36  musica_Rap                 1171 non-null   uint8
     37  musica_Rock                1171 non-null   uint8
     38  deporte_1hora              1171 non-null   uint8
     39  deporte_2horas             1171 non-null   uint8
     40  deporte_4horas             1171 non-null   uint8
     41  perro_agua                 1171 non-null   uint8
     42  perro_atom_humber          1171 non-null   uint8
     43  perro_bus                  1171 non-null   uint8
     44  perro_chilaquil            1171 non-null   uint8
     45  perro_ladron               1171 non-null   uint8
     46  perro_lipigas              1171 non-null   uint8
     47  perro_matapacos            1171 non-null   uint8
     48  perro_washington           1171 non-null   uint8
     49  perro_waton                1171 non-null   uint8
    dtypes: int32(1), int8(1), uint8(48)
    memory usage: 69.8 KB
    

En cuanto a política, como nuestra variable de respuesta será binaria en base a si la persona es de izquierda o no, se dejarán fuera las otras variables (centro, derecha, NS/NC).


```python
df = df.loc[:, ~df.columns.isin(['politica_centro', 'politica_derecha', 
                                 'politica_NS_NC'])]
```

# Genera input para posteriores modelos


```python
# generamos salida .pkl para luego presentar a distintos modelos
df.to_pickle("df_encuesta.pkl")
```

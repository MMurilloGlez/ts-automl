import pandas as pd
import numpy as np
import dateparser


### Importar el dataframe tal cual
df = pd.read_csv(filename, sep = ';', decimal = ',')
df = df.iloc[:,:2]
### Parse las dates.
dates = df.iloc[:,0]


dates = pandas.to_datetime(format = "%d/%m/%Y %H%M%S.%f")
print(dates.freq)

if !dates.freq:
dates.freq = dates[1]-dates[0]

print(dates.freq)
freq  = dates.freq

df['DATE'] = pd.date_range(start = dates[0], periods = len(df), freq = freq)

### Detectar formato de fecha. Con dateparser lo hace de uno en uo asi que dificil. Leer 30 dates, ver si alguno se lee con la posicion de mayor de 12 en x sitio y quedarme con ese formato
### Detectar frecuencia correctamente para ello:
        ### Crear el nuevo datetimeindex con formato estandar.
        ### Restar dos fechas cualesquiera de uno del index, con lo que me de saco lo que sea de freq
        ### Build un datetime Range empezando de ahi y con longitud lo que toque. Para evitar gaps.
        ### Si tiene gaps lo quiero organizar?? Casi que no porque nos va a dar las series idbox
        ### Lo mismo con el formato

### Quedarme con los ultimos x valores (quizá poder darselos como tiempo (3h tal))



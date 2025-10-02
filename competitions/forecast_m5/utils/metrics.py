# script con código para calcular métrica - el código se desarrolló en la carpeta EDA

from sklearn.metrics import root_mean_squared_error
import pandas as pd


def calculate_metrics_wrmse_v1(df_true, df_pred):
    """
    Calcular weighted root mean square error. Calcular RMSE para cada serie y ponderarlo por peso. Donde peso se calcula 
    a partir del volumen de los reales de cada serie
    Métrica se calcula haciendo un FOR por serie
    
    Args
        df_true (DataFrame): DataFrame de la forma (S,H) donde S: total de las series // H: total horizonte predicción
        df_pred (DataFrame): DataFrame de la forma (S,H) donde S: total de las series // H: total horizonte predicción
    """

    # generar df con todas las series
    df_metrics = df_true[['id']]
    
    # calcular métrica RMSE para cada serie individual y gurdar en df metrics
    list_all_series = df_metrics['id'].tolist()
    for n_serie in list_all_series:
        # print(n_serie)
    
        # generar serie auxiliar
        aux_true_serie = df_true[df_true['id'] == n_serie]
        aux_pred_serie = df_pred[df_pred['id'] == n_serie]
        
        aux_true_serie = aux_true_serie.drop(columns = 'id').T
        aux_pred_serie = aux_pred_serie.drop(columns = 'id').T
        
        # calcular métrica
        rmse_aux = root_mean_squared_error(y_true = aux_true_serie,
                                           y_pred = aux_pred_serie)
        
        # guardar en df_metrics. columna "rmse"
        df_metrics.loc[df_metrics['id'] == n_serie, 'rmse'] = rmse_aux
    
    
    # calcular peso CADA SERIE en base a todos los datos true (peso = volumen_serie_i / volumen_total_series). 
    df_true_sum_each_serie = df_true.set_index(['id']).sum(axis = 1).to_frame(name='volumen_real_serie').reset_index() 
    df_true_sum_each_serie['weight'] = df_true_sum_each_serie['volumen_real_serie'] / df_true_sum_each_serie['volumen_real_serie'].sum()
    df_true_sum_each_serie.drop(columns = 'volumen_real_serie', inplace = True)
    df_metrics = df_metrics.merge(df_true_sum_each_serie, on='id')
    
    
    # calcular métrica final ponderando rmse * weight (para cada serie)
    df_metrics['weight_rmse'] = df_metrics['rmse'] * df_metrics['weight']
    output_wrmse = df_metrics['weight_rmse'].sum()
    
    return output_wrmse



def calculate_metrics_wrmse_v2(df_true, df_pred):
    """
    Calcular weighted root mean square error. Calcular RMSE para cada serie y ponderarlo por peso. Donde peso se calcula 
    a partir del volumen de los reales de cada serie
    Métrica se calcula haciendo GROUPBY POR SERIE y a la agrupación calcular el RMSE. EVITAR USAR UN FOR.
    
    Args
        df_true (DataFrame): DataFrame de la forma (S,H) donde S: total de las series // H: total horizonte predicción
        df_pred (DataFrame): DataFrame de la forma (S,H) donde S: total de las series // H: total horizonte predicción
    """

    # melt data. dejar en formato "id", "ds", "value"
    df_pred_melt = pd.melt(df_pred, id_vars=['id'], var_name='ds', value_name='pred')
    df_true_melt = pd.melt(df_true, id_vars=['id'], var_name='ds', value_name='true')

    
    # concatenar ambos df true y df pred. # obtener dataframe de la forma "id", "serie", "true", "pred"
    df_true_pred = pd.merge(df_pred_melt, df_true_melt, on = ['id', 'ds'])

    
    # CALCULAR RMSE PARA CADA SERIE
    # definir función para calcular métrica RMSE tras agrupar
    def calculate_rmse(group):
        """
        calcular métrica RMSE individual para cada serie. groupby.
        Es una función se aplica en un .apply(lambda) para la data agrupada por groupby
        """
        y_true = group['true'].values
        y_pred = group['pred'].values
        return root_mean_squared_error(y_true, y_pred)

    # agrupar la data por id. a la agrupación aplicar función que calcula el RMSE de cada serie
    df_metrics = df_true_pred.groupby('id').apply(lambda group: calculate_rmse(group)).reset_index(name='rmse')
    
    
    # calcular peso CADA SERIE en base a todos los datos true (peso = volumen_serie_i / volumen_total_series). 
    df_true_sum_each_serie = df_true.set_index(['id']).sum(axis = 1).to_frame(name='volumen_real_serie').reset_index() 
    df_true_sum_each_serie['weight'] = df_true_sum_each_serie['volumen_real_serie'] / df_true_sum_each_serie['volumen_real_serie'].sum()
    df_true_sum_each_serie.drop(columns = 'volumen_real_serie', inplace = True)
    df_metrics = df_metrics.merge(df_true_sum_each_serie, on='id')
    
    
    # calcular métrica final ponderando rmse * weight (para cada serie)
    df_metrics['weight_rmse'] = df_metrics['rmse'] * df_metrics['weight']
    output_wrmse = df_metrics['weight_rmse'].sum()
    
    return output_wrmse



def save_value_metric_csv(name_model, metric_value):
    """
    Guardar en CSV una fila con el valor de la métrica obtenida. 
    El CSV ya está creado y solo se crea/actualiza una fila con el nombre del modelo y la métrica
    Args
        name_model (string): nombre de modelo a ser guardado
        metric_value (float): valor de la métrica obtenida a ser guardada
    """

    # CODIGO COMENTADO, USADO LA PRIMERA VEZ PARA CREAR EL CSV CON LAS MÉTRICAS
    # df_metrics = pd.DataFrame([[name_model, metric_value]], columns = ['name_model', 'metric'])
    # df_metrics.to_csv(path_csv_metric, index = False)

    ### ACTUALIZAR CSV CON EL VALOR DE LA MÉTRICA OBTENIDA
    # definir nombre del modelo
    path_csv_metric = '2_fcst_puntual/metrics.csv'
    
    # leer df metric
    df_metrics = pd.read_csv(path_csv_metric)
    
    # Verificar si existe una fila con 'name_model' igual a valor "name_model". si NO existe, agregar fila. si existe, actualizar metrica
    if not (df_metrics['name_model'] == name_model).any():
        new_row = {'name_model': name_model, 'metric': metric_value}
        df_metrics = pd.concat([df_metrics, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df_metrics.loc[df_metrics['name_model'] == name_model, 'metric'] = metric_value
    
    # guarar df metrics, TODOS LOS VALORES y más la ACTUALIZACIÓN con la métrica del modelo recién entrenado
    df_metrics.to_csv(path_csv_metric, index = False)





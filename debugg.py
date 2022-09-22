import os
import sys
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Output, Input, State
import dash_bootstrap_components as dbc
import pandas as pd
from datetime import datetime
import plotly.graph_objs as go
from math import log, sqrt, degrees, atan
import collections
from functools import reduce
import site
from sqlalchemy import create_engine
from itertools import groupby
from itertools import chain
import numpy as np


print("hello")

# calcula o retorno perceuntual de um ativo (close)
def calc_return_price(df, lista_return):
    for el in range(len(lista_return)):
        df['return_{}'.format(lista_return[el])] = (df["close"] - df["close"].shift(-lista_return[el])) / abs(
            df["close"].shift(-lista_return[el]))

    return df


def get_avg_time_trade(rep):
    first = rep.sort_values("date")["signal"].iloc[0]
    temp = rep[(rep["signal"] == first) | (pd.isnull(rep["signal"]))]
    temp_signal = temp[temp["signal"] == first]
    tam_all = temp.__len__()
    tam_sigs = temp_signal.__len__()

    avg_time_trade_days = round(tam_all / tam_sigs)

    return avg_time_trade_days


# calcula o retorno perceuntual de um ativo (close)
def calc_return_general(df, col, lista_return, relative_return_general=True):
    for el in range(len(lista_return)):

        if relative_return_general:

            df['return_{}_{}'.format(col, lista_return[el])] = (df[col] - df[col].shift(-lista_return[el])) / abs(
                df[col].shift(-lista_return[el]))

        else:

            df['return_{}_{}'.format(col, lista_return[el])] = (df[col] - df[col].shift(-lista_return[el]))

    return df


# calcula o retorno perceuntual de um ativo (close)
def calc_return_all(df, lista_return, col):
    for el in range(len(lista_return)):
        df['return_{}_{}'.format(col, lista_return[el])] = 100 * (df[col] - df[col].shift(-lista_return[el])) / abs(
            df[col].shift(-lista_return[el]))

    return df


def calc_beta(df, col, col_ref, window_beta):
    df["beta"] = (df[col].rolling(window_beta).std(ddof=1) / df[col_ref].rolling(window_beta).std(ddof=1)) * df[
        col].rolling(window_beta).corr(df[col])

    return df


# @jit(nopython=False,parallel=True)
def calc_angle(angs, el=0, window=10000):
    angs = angs[el:(el + window)]
    angs.reverse()
    y = np.array(angs)
    x = np.linspace(0, len(y) - 1, num=len(y), endpoint=True, retstep=False, dtype=int)
    A = np.array([x, np.ones(len(x))])
    w = np.linalg.lstsq(A.T, y)[0]
    slope = w[0]
    intercept = w[1]

    # definimos o DY como uma porcentagem em relacao ao intercept da reta de regressao
    dy = 100 * (slope * window / abs(intercept))
    angle = degrees(atan(dy / window))
    return angle


'''
 - variavel close como default
 - angulo de uma serie limitada pelos indices passados como parametros

'''


def calc_vol(returns, el=0, el2=10000):
    vol = np.std(returns[el:(el + el2 - 1)], ddof=1)
    return vol


# recebe array
def calc_beta2(s, m, el=0, el2=10000):
    s = s[el:(el + el2)]
    m = m[el:(bb + el2)]
    covariance = np.cov(s, m, ddof=1)  # Calculate covariance between stock and market
    beta = covariance[0, 1] / covariance[1, 1]
    return beta


'''
recebe a lista de angulos e retorna a aceleracao do do angulos (ou qlqr outro serie)
Assume que esta na ordem decrescente
'''


# segunda derivada
def calc_acc(lt2):
    lt2.reverse()
    ac1 = np.diff(lt2, n=1)
    ac2 = np.diff(ac1, n=1)
    ac2 = ac2.tolist()
    ac2.reverse()
    ac2.append(None)
    # df["acc_{}".format(window_angle[el])] = ac2
    return ac2


'''
recebe uma lista de parametros de janela de retornos
assume que o dataframe passado como parametro ja possui a coluna "angle" associada a cada parametro passado
'''


##

def calc_return_angle(df, lista_return_angle, window_angle):
    for el in range(len(lista_return_angle)):
        df['return_angle_{}'.format(lista_return_angle[el])] = 100 * (
                    df["angle_{}".format(window_angle[el])] - df["angle_{}".format(window_angle[el])].shift(
                -lista_return_angle[el])) / abs(df["angle_{}".format(window_angle[el])].shift(-lista_return_angle[el]))

    return df


def calc_return_acc(df, lista_return_acc, window_angle):
    for el in range(len(lista_return_acc)):
        df['return_acc_{}'.format(lista_return_acc[el])] = 100 * (
                    df["acc_{}".format(window_angle[el])] - df["angle_{}".format(window_angle[el])].shift(
                -lista_return_acc[el])) / abs(df["angle_{}".format(window_angle[el])].shift(-lista_return_acc[el]))

    return df


def control_liq(arr, n, k, z):
    k = k + 1
    myset = []
    m = 0
    flag = 0
    for i in range(n):
        # alread passed by k elements. Remove the oldest elemnt from set
        if (i >= k):
            myset.remove(arr[i - k])

        if arr[i] in myset:
            counter = collections.Counter(myset)
            if counter.most_common(1)[0][1] >= z:
                return True
        # else:
        # Add this item to hashset

        myset.append(arr[i])

    return False


''' 

# df: ticker
# w_n: lookBackWindow
# k: rolling window to verify liquididy
# z: number of repetition acceptable

'''


def control_liqTicker(df, w_n=500, k=10, z=10):
    arr = df["close"].tolist()
    lista_res = []
    for i in range(len(arr)):
        n = len(arr[i:i + w_n])
        if (control_liq(arr=arr[i:i + w_n], n=n, k=k, z=z)):
            lista_res.append(True)

        else:
            lista_res.append(False)

    df["flag_liq"] = lista_res
    return df


# daily return for index and assets
def help_return(linha, **kwargs):
    # log_return=True,perc_return=False,index_return=False
    col = 'close'
    col_prev = 'close_d-1'

    if kwargs["index_return"]:
        col = col + '_index'
        col_prev = col_prev + '_index'

    if kwargs["log_return"]:
        return sqrt(252) * (log(linha[col]) - log(linha[col_prev]))

    if kwargs["perc_return"]:
        return 100 * ((linha[col] - linha[col_prev]) / linha[col_prev])


# add data de vencimento a futuros de di
def add_day_to_venciDI(rep):
    dus = pd.read_sql(
        "select date from holidays.brazil_du where isweekday = 'TRUE' and date >='2002-01-01' and date <='2030-01-01' ",
        create_engine("mysql+pymysql://administrator:Enter123@192.168.0.5/{}".format("backtesting")))

    rep["year"] = rep["date"].apply(lambda x: x.year)
    rep["date_venci"] = rep["ticker"].apply(lambda x: datetime(2000 + int(x[-2:]) - 1, 12, 31))
    dt_min = min(rep["date"].min(), rep["date_venci"].min())
    dt_max = max(rep["date"].max(), rep["date_venci"].max())
    du_array = dus[(dus["date"] > dt_min) & (dus["date"] <= dt_max)]
    du_array = du_array.sort_values("date")["date"].tolist()
    dates_rep = rep["date"].tolist()
    dates_venci_rep = rep["date_venci"].tolist()

    lista_venci = []
    for data_el in range(len(dates_rep)):
        days = [du_array[el] for el in range(len(du_array)) if
                du_array[el] > dates_rep[data_el] and du_array[el] <= dates_venci_rep[data_el]].__len__()
        lista_venci.append(days)

    df_venci = pd.DataFrame({"date": dates_rep, "days_venci": lista_venci})
    df_venci = df_venci.drop_duplicates()
    rep = pd.merge(rep, df_venci, left_on=["date"], right_on=["date"], how="inner")

    return rep


'''

df_desc: dataframe dos ticke que se deseja calcular os parametros, passado em ordem descendente

---
vol: Se deseja calcular Vol ou Nao
window_vol: lista de janelas de vol

---
angle: calcular angulo ou Nao
window_angle: lista de angulis
col_angle: coluna usada para se calcular o angulo

---
betas: calcular return_price ou Nao
window_betas: lista de janelas de betas

---
return_price: calcular return_price nos ultimos X dias ou nao
lista_return_price: calcular return_price ou Nao

---
ma: calcular media movel ou nao
col_ma: qual coluna se deseja calcular a media movel ?
ista_mas: lista de janelas de medias moveis

---
lista_return
'''


# window = window_vol
# para o lowbe beta olhanos o angulo de precos
def compute_all(df_desc,
                window_vol=None,
                window_angle=None,
                release_memory=True,
                vol=False,
                short_box=False,
                acc=False,
                angle=False,
                ma=False,
                return_price=False,
                return_angle=False,
                return_acc=False,
                daily_return_perc=False,
                betas=False,
                lista_return_angle=None,
                lista_return_price=None,
                lista_return_acc=None,
                lista_return_PE=None,
                window_betas=None,
                lista_mas=None,
                col_angle='close',
                col_ma_names=['close'],
                col_std='volume',
                flag_asset=True,
                excess_return=False,
                calc_std=False,
                window_stds=None,
                lista_excess_return=None,
                control_liq=False,
                window_liq=None,
                window_fundamentals=None,
                fundamentals=False,
                avg_price=False,
                add_day_to_venci=False,
                spread=False,
                flag_ma=False,
                lista_w=None,
                if_sum=False,
                lista_sums=None,
                turnover=False,
                lista_turnover=None,
                col_sum_names=['volume_shares'],
                size_conver=16,
                col_return_names=None,
                lista_returns=None,
                returns_general=False,
                lista_window_markov=None,
                compute_markov=False,
                periodo_vol=252,
                is_spread_vol=False,
                num_vol_mc=1.5,
                max_con=0.7,
                perc_return=False,
                log_return=True,
                index_return=False,
                # PRECISA TER TODAS AS MA calculadas !
                ma_dif=False,
                window_dif=120,
                lag_returns=False,
                N_max=15,
                nbin=7,
                col_lag_return='return',
                # no caso de taxa, tem q ser false, pois nao da pra fazer retorno de negativo
                want_daily_return=True,
                rollmax=False,
                col_rollmax_names=['close'],
                lista_rollmaxW=None,
                sharpe=False,
                relative_return_general=True,
                k_liq=15,
                prop=3,
                flag_autocorr=False,
                lista_lags_auto=None,
                lista_windows_autocorr=None,
                lista_col_autocorr=None,
                flag_autocorr_spearman=False,
                entropy_features=False,
                windows_entropy=None,
                min_values_features=None,
                cols_entropy=None,
                col_vol_entropy=None,
                mode_entropy=1
                ):

    def add_cum_AutoCorr(df, col, lag):

        pr = df[col].tolist()
        lista_autocorr = []

        for i in range(len(pr)):
            cc = np.corrcoef(pr[i:(len(pr[i + lag:])) + i], pr[i + lag:])
            lista_autocorr.append(cc[0][1])

        df["autocorr_{}_acc_{}".format(col, lag)] = lista_autocorr

        return df

    ##################################### ADDIN SPREAD #############################
    if spread:
        df_desc["spread"] = df_desc["close"] / df_desc["close_index"]

    # retorno usado para calcular vol
    def help_return(linha, **kwargs):

        # log_return=True,perc_return=False,index_return=False
        col = 'close'
        col_prev = 'close_d-1'

        if kwargs["index_return"]:
            col = col + '_index'
            col_prev = col_prev + '_index'

        if kwargs["is_spread_vol"]:
            col = "spread"
            col_prev = "spread_d-1"

        # vol anual, mensal, semanal etc etc etc
        if kwargs["log_return"]:
            return sqrt(kwargs["period_vol"]) * (log(linha[col]) - log(linha[col_prev]))

        if kwargs["perc_return"]:
            return 100 * ((linha[col] - linha[col_prev]) / linha[col_prev])

    if flag_asset:

        _l = df_desc["close"].iloc[1:].tolist()
        _l.append(None)
        df_desc["close_d-1"] = _l

        if is_spread_vol:
            _l = df_desc["spread"].iloc[1:].tolist()
            _l.append(None)
            df_desc["spread_d-1"] = _l

        if vol:
            want_daily_return = True

        if avg_price:
            _l = df_desc["avg_price"].iloc[1:].tolist()
            _l.append(None)
            df_desc["avg_price_d-1"] = _l

        if want_daily_return:
            # print("daily return")
            # print("o spread vol: {}".format(is_spread_vol))
            # calcula log daily return for all assets
            df_desc["daily_return"] = df_desc.apply(help_return, index_return=index_return, perc_return=perc_return,
                                                    log_return=log_return, axis=1, period_vol=periodo_vol,
                                                    is_spread_vol=is_spread_vol)


        else:

            pass

    # retorno de ativos
    if daily_return_perc:
        df_desc["daily_return_perc"] = df_desc.apply(help_return, index_return=index_return, perc_return=perc_return,
                                                     log_return=False, axis=1)

    # retorno de indices
    if betas:

        df_desc["daily_return_index"] = df_desc.apply(help_return, index_return=True, perc_return=perc_return,
                                                      log_return=log_return, axis=1, is_spread_vol=is_spread_vol,
                                                      period_vol=periodo_vol)
        if daily_return_perc:
            df_desc["daily_return_index_perc"] = df_desc.apply(help_return, index_return=index_return,
                                                               perc_return=perc_return, log_return=False, axis=1)

    lista_vols = []
    lista_slope = []
    lista_intercept = []
    lista_return = []
    lista_accelaration = []
    lista_of_listaVol = []
    lista_of_listaAngle = []
    lista_of_listaBetas = []
    lista_angle = []

    ###################################### calculo da vol #######################@@@@

    if sharpe:
        df_desc["daily_return_2"] = df_desc["daily_return"] / sqrt(252)

    # calcula vol para uma data lista de janelas de vol
    if vol:
        df_desc.sort_values("date", inplace=True)

        for el2 in range(len(window_vol)):
            df_desc["vol_{}".format(window_vol[el2])] = df_desc["daily_return"].rolling(window_vol[el2]).std(ddof=1)
            if sharpe:
                df_desc["sharpe_{}".format(window_vol[el2])] = df_desc["daily_return_2"].rolling(
                    window_vol[el2]).mean() / df_desc["daily_return"].rolling(window_vol[el2]).std(ddof=1)

        df_desc.sort_values("date", ascending=False, inplace=True)

    ################################### calculo do angulo #########################@@@

    # calcula angulo para uma dada lista de angulos
    if angle:

        # start_time1 = time.time()
        b = len(window_angle)
        for i in range(b):
            lista_of_listaAngle.append([])

        # esta setado para se calcular o angulo da serie de precos "close". Pode ser qlqr coisa entretanto.
        angs = df_desc[col_angle].tolist()
        for el in range(len(df_desc)):
            for el2 in range(len(window_angle)):
                angle = calc_angle(angs, el, window=window_angle[el2])
                lista_of_listaAngle[el2].append(angle)

        # apenas renomeando
        for _el2 in range(len(window_angle)):
            df_desc["angle_{}".format(window_angle[_el2])] = lista_of_listaAngle[_el2]


    if flag_autocorr_spearman:

        cols_df = df_desc.columns.tolist()

        for col in lista_col_autocorr:

            if ((col not in cols_df) & (col[0:7] == 'return_')):

                col_temp = col.split("return_")[1].split("_")[0]
                w_temp = int(col.split("return_")[1].split("_")[1])
                df_desc = calc_return_general(df_desc.sort_values("date", ascending=False), col_temp, [w_temp],
                                              relative_return_general=False).sort_values("date")

            else:

                pass

            for window in lista_windows_autocorr:

                for lag in lista_lags_auto:
                    df_desc = add_AutoCorr_Sperman(df_desc, col, lag, window=window)

    # evita ficar fazendo sort em dataframes
    if (betas | ma | excess_return | calc_std | if_sum | rollmax | flag_autocorr):

        df_desc.sort_values("date", inplace=True)

        cols_df = df_desc.columns.tolist()

        if flag_autocorr:

            for col in lista_col_autocorr:

                if ((col not in cols_df) & (col[0:7] == 'return_')):

                    col_temp = col.split("return_")[1].split("_")[0]
                    w_temp = int(col.split("return_")[1].split("_")[1])
                    df_desc = calc_return_general(df_desc.sort_values("date", ascending=False), col_temp, [w_temp],
                                                  relative_return_general=False).sort_values("date")

                else:

                    pass

                for window in lista_windows_autocorr:

                    for lag in lista_lags_auto:

                        if window != 'acc':

                            df_desc["autocorr_{}_{}_{}".format(col, window, lag)] = df_desc[col].rolling(window).apply(
                                lambda x: x.autocorr(lag=lag), raw=False)

                        else:

                            df_desc = df_desc.sort_values("date", ascending=False)
                            df_desc = add_cum_AutoCorr(df_desc, col, lag)
                            df_desc = df_desc.sort_values("date", ascending=True)

        if betas:
            for _el2 in range(len(window_betas)):
                df_desc["beta_{}".format(window_betas[_el2])] = (df_desc["daily_return"].rolling(
                    window_betas[_el2]).std(ddof=1) / df_desc["daily_return_index"].rolling(window_betas[_el2]).std(
                    ddof=1)) * df_desc['daily_return'].rolling(window_betas[_el2]).corr(df_desc['daily_return_index'])
                df_desc["beta_{}".format(window_betas[_el2])].fillna(0, inplace=True)

        if if_sum:

            for el in range(len(col_sum_names)):

                for el2 in range(len(lista_sums[el])):
                    df_desc["sum_{}_{}".format(col_sum_names[el], lista_sums[el][el2])] = df_desc[
                        '{}'.format(col_sum_names[el])].rolling(lista_sums[el][el2]).sum()

        if rollmax:

            for el in range(len(col_rollmax_names)):

                for el2 in range(len(lista_rollmaxW[el])):
                    df_desc["rollmax_{}_{}".format(col_rollmax_names[el], lista_rollmaxW[el][el2])] = df_desc[
                        '{}'.format(col_rollmax_names[el])].rolling(lista_rollmaxW[el][el2]).max()
                    df_desc["rollmin_{}_{}".format(col_rollmax_names[el], lista_rollmaxW[el][el2])] = df_desc[
                        '{}'.format(col_rollmax_names[el])].rolling(lista_rollmaxW[el][el2]).min()

        if ma:

            for el in range(len(col_ma_names)):

                for el2 in range(len(lista_mas[el])):

                    df_desc["ma_{}_{}".format(col_ma_names[el], lista_mas[el][el2])] = df_desc[
                        '{}'.format(col_ma_names[el])].rolling(lista_mas[el][el2]).mean()

                    if ma_dif:
                        df_desc["dif_{}_{}".format(col_ma_names[el], lista_mas[el][el2])] = df_desc["close"] - df_desc[
                            "ma_{}_{}".format(col_ma_names[el], lista_mas[el][el2])]

                        # MEDIA MOVEL DAS DISTANCIAS
                        df_desc["ma_{}_{}".format('dif', lista_mas[el][el2])] = df_desc[
                            "dif_{}_{}".format(col_ma_names[el], lista_mas[el][el2])].rolling(window_dif).mean()

                        # STD MOVEL DAS DISTANCIAS
                        df_desc["std_{}_{}".format('dif', lista_mas[el][el2])] = df_desc[
                            "dif_{}_{}".format(col_ma_names[el], lista_mas[el][el2])].rolling(window_dif).std(ddof=1)

                        # df_desc.drop(["dif_{}_{}".format(col_ma_names[el],lista_mas[el][el2])],axis=1,inplace=True)

                    if flag_ma:

                        cc = "ma_{}_{}".format(col_ma_names[el], lista_mas[el][el2])

                        # print("o cc eh: {}".format(cc))

                        new_col = 'flag_{}'.format(cc.split('_')[-1])

                        # df_desc[new_col] = abs((df_desc['close'] > df_desc[cc]).diff())
                        if not short_box:

                            df_desc[new_col] = df_desc[cc] > df_desc["close"]

                        else:

                            df_desc[new_col] = df_desc[cc] < df_desc["close"]

                        # para cada media, cria a flag informando se "TRUE": sempre esteve acima do preco.
                        for w in lista_w:

                            # se tiver 1, quer dizer que a media foi maior que o preco em algum ponto.

                            if not short_box:

                                df_desc['flag_{}_{}'.format(cc.split('_')[-1], w)] = df_desc[new_col].rolling(
                                    w).max().values

                            else:

                                df_desc['flag_{}_{}'.format(cc.split('_')[-1], w)] = df_desc[new_col].rolling(
                                    w).max().values

                        df_desc.drop(new_col, axis=1, inplace=True)

        # if flag_ma:

        #     #df_desc.sort_values("date",ascending=False,inplace=True)
        #     for el in range(len(col_ma_names)):

        #         for el2 in range(len(lista_mas[el])):

        #             cc = "ma_{}_{}".format(col_ma_names[el],lista_mas[el][el2])
        #             new_col = 'flag_{}'.format(cc.split('_')[-1])
        #             df_desc[new_col] = df_desc['close'] < df_desc[cc]

        #             lista_w = [180,90,45]

        #             for w in lista_w:
        #                 #df_desc['flag_{}_{}'.format(cc.split('_')[-1],w)] = df_desc.groupby("ticker")[new_col].rolling(w).min().values
        #                 df_desc['flag_{}_{}'.format(cc.split('_')[-1],w)] = df_desc[new_col].rolling(w).max().values

        #             df_desc.drop(new_col,axis=1,inplace=True)

        # df_desc = df_desc.sort_values("date",ascending=False)
        df_desc.sort_values("date", ascending=False, inplace=True)

        if turnover:

            for el_lista in lista_turnover:

                for el_lista2 in lista_sums[col_sum_names.index("volume_shares")]:
                    # PERIODO DO TURNOVER
                    df_desc["turnover"] = df_desc["sum_volume_shares_{}".format(el_lista2)] / df_desc[
                        "ma_outstanding_{}".format(el_lista2)]

                    df_desc["turnover_d-1"] = df_desc["turnover"].shift(-el_lista)

                    df_desc["change_turnover_{}_{}".format(el_lista, el_lista2)] = 100 * (
                                df_desc["turnover"] / df_desc["turnover_d-1"] - 1)

                    df_desc["turnover_{}_{}".format(el_lista, el_lista2)] = df_desc["turnover"]

                    df_desc.drop(["turnover", "turnover_d-1"], axis=1, inplace=True)

            max_window_turnover = max(lista_turnover) + max(lista_sums[col_sum_names.index("volume_shares")])

        if release_memory:

            # LIBERANDO MEMORA
            lista_64 = df_desc.dtypes.where(lambda x: x == 'float64').dropna().index.tolist()
            for col in lista_64:
                df_desc[col] = df_desc[col].astype('float{}'.format(size_conver), copy=False)

        else:

            pass

        # excesso de retorno com relacao a taxa libre de risco nos ultims  N dias
        if excess_return:
            df_desc["carry_2"] = df_desc["carry"] + 1
            df_desc["carry_acc"] = df_desc["carry_2"].cumprod(axis=0)

        if calc_std:
            for _el2 in range(len(window_stds)):
                df_desc["std_{}_{}".format(col_std, window_stds[_el2])] = df_desc[col_std].rolling(
                    window_stds[_el2]).std(ddof=1)

        # if not flag_ma:

    ########################## FUDAMENTALSSSSSSSSS ##########################

    if returns_general:
        i = 0
        for col in col_return_names:
            df_desc = calc_return_general(df_desc, col, lista_returns[i],
                                          relative_return_general=relative_return_general)
            i = i + 1

    if fundamentals:
        for col in ['ebitev', 'roc', 'pe', 'dvd_yld', 'eps', 'roe', 'g_factor']:
            df_desc = calc_return_all(df_desc, window_fundamentals, col)

    ################## DI

    if add_day_to_venci:
        df_desc = add_day_to_venciDI(df_desc)

    ###################################### calculo de retornos de precos #####

    if return_price:
        df_desc = calc_return_price(df_desc, lista_return_price)

    if control_liq:
        # df_desc = control_liqTicker(df = df_desc,w_n = window_liq[0])
        df_desc = control_liqTicker(df=df_desc, w_n=window_liq[0], k=k_liq, z=int(k_liq / prop))

    if excess_return:
        for el2 in range(len(lista_excess_return)):
            df_desc["return_carry_{}".format(lista_excess_return[el2])] = 100 * (
                        (df_desc["carry_acc"] - df_desc["carry_acc"].shift(-lista_excess_return[el2])) / abs(
                    df_desc["carry_acc"].shift(-lista_excess_return[el2])))
            df_desc["excess_{}".format(lista_excess_return[el2])] = df_desc[
                                                                        "return_{}".format(lista_excess_return[el2])] - \
                                                                    df_desc["return_carry_{}".format(
                                                                        lista_excess_return[el2])]

    ###################################### calculo de retornos ########################
    if flag_asset:
        df_desc = df_desc[~pd.isnull(df_desc["close_d-1"])]

    if return_angle:
        df_desc = calc_return_angle(df_desc, lista_return_angle, window_angle)

    if acc:
        for el in range(len(lista_of_listaAngle)):
            ac2 = calc_acc(lista_of_listaAngle[el])
            df_desc["acc_{}".format(window_angle[el])] = ac2

    if lag_returns:
        for i in range(1, N_max):
            df_desc["{}_{}_d-{}".format(col_lag_return, nbin, nbin * i)] = df_desc[
                "{}_{}".format(col_lag_return, nbin)].shift(-nbin * i)

    if return_acc:
        df_desc = calc_return_acc(df_desc, lista_return_acc, window_angle)

    if vol:
        df_desc = df_desc[~pd.isnull(df_desc["close_d-1"])]


    if not flag_autocorr:
        lista_windows_autocorr = []

    if windows_entropy is None:
        windows_entropy = []

    if not compute_markov:
        lista_window_markov = []

    if not lag_returns:
        window_lag_returns = []


    else:
        window_lag_returns = [nbin * N_max]

    if lista_mas is not None:

        lista_mas = reduce(lambda x, y: x + y, lista_mas)

    else:
        lista_mas = [0]

    if lista_returns is not None:
        lista_returns = reduce(lambda x, y: x + y, lista_returns)

    if lista_rollmaxW is not None:
        lista_rollmaxW = reduce(lambda x, y: x + y, lista_rollmaxW)

    janelas = [windows_entropy, window_lag_returns, lista_window_markov, window_angle, window_vol, window_betas,
               window_stds, lista_windows_autocorr, lista_returns, lista_rollmaxW, lista_mas, lista_return_price,
               lista_excess_return, window_liq]
    janelas = [el if isinstance(el, list) else list() for el in janelas]
    janelas = reduce(lambda x, y: x + y, janelas)
    # print("janelssss")
    # print(janelas)
    # janelas = sum(janelas, [])

    if janelas:
        # max_window = max(janelas)
        max_window = max(janelas) + max(lista_mas)

        if turnover:
            max_window = max(max_window, max_window_turnover)

        df_desc["date"].iloc[(len(df_desc) - (max_window)):len(df_desc)] = None
        df_desc = df_desc[~pd.isnull(df_desc["date"])]

    return df_desc


# Driver Code
# if __name__ == "__main__":
#
#    arr = [3, 3, 3, 3, 3, 3, 4,55,54,67]
#    n = len(arr)
#    start_time1 = time.time()
#    #for i in range(3000*200):
#    if(checkDuplicatesWithinK(arr= arr, n = n,k = 4, z = 4)):
#        print("Tem duplicates")
#
#
#    else:
#        print("Nao tem duplicates")


# retorna precos dos indices. Ja retorna a coluna com preco de d-1 --> Para backtest
def priceIndex(sql_con, start, offset, index='IBX', div_adj=1, flag_intraday=False, flag_signals=False):
    if flag_signals:

        precos = pd.read_sql(
            """SELECT date,close FROM backtesting.eqt_brazil_signals where ticker = 'IBX Index' ORDER BY DATE DESC""",
            sql_con)

        # precos = precos.iloc[(len(precos)-(offset)):]
        precos.sort_values("date", ascending=False, inplace=True)
        _l = precos["close"].iloc[1:].tolist()
        _l.append(None)
        precos["close_d-1"] = _l
        precos = precos.iloc[:-1]
        precos.columns = ["date", "close_index", "close_d-1_index"]

        return precos


    else:

        if index == 'IBX':

            precos = pd.read_sql("""SELECT date,close FROM backtesting.eqt_brazil
                                where ticker = 'IBX Index' and div_adj = {}
                                ORDER BY DATE DESC
                                limit 0,6000""".format(div_adj), sql_con)


        elif index == 'AS51':

            precos = pd.read_sql("""SELECT date,close FROM backtesting.eqt_aus
                        where ticker = 'AS51 index' and div_adj = {}
                        ORDER BY DATE DESC
                        limit 0,6000""".format(div_adj), sql_con)

        elif index == 'IPSA':
            precos = pd.read_sql("""SELECT date,close FROM backtesting.eqt_chile
                            where ticker = 'IPSA Index' and div_adj = {}
                        ORDER BY DATE DESC
                        limit 0,6000""".format(div_adj), sql_con)


        elif index == 'SPTSX':
            precos = pd.read_sql("""SELECT date,close FROM backtesting.eqt_sptsx
                        where ticker = 'SPTSX Index' and div_adj = {}
                        ORDER BY DATE DESC
                        limit 0,6000""".format(div_adj), sql_con)


        elif index == 'MEXBOL':
            precos = pd.read_sql("""SELECT date,close FROM backtesting.eqt_mexbol
                            where ticker = 'MEXBOL Index' and div_adj = {}
                        ORDER BY DATE DESC
                        limit 0,6000""".format(div_adj), sql_con)


        elif index == 'TOP40':
            precos = pd.read_sql("""SELECT date,close FROM backtesting.eqt_top40
                            where ticker = 'TOP40 Index' and div_adj = {}
                        ORDER BY DATE DESC
                        limit 0,6000""".format(div_adj), sql_con)

        elif index == 'SPX':
            precos = pd.read_sql("""SELECT date,close FROM backtesting.eqt_spx where ticker ='SPX Index'
                                and div_adj = {}
                                ORDER BY DATE DESC 
                                limit 0,5000""".format(div_adj), sql_con)

        else:
            raise ValueError('Problem with index prices')

        # precos = precos.iloc[(len(precos)-(offset)):]
        precos.sort_values("date", ascending=False, inplace=True)
        _l = precos["close"].iloc[1:].tolist()
        _l.append(None)
        precos["close_d-1"] = _l
        precos = precos.iloc[:-1]
        precos.columns = ["date", "close_index", "close_d-1_index"]

        return precos


def process_bbg(df_raw, fundamentals=False, intraday=False, index_name='IBX Index', flag_cols=False, _cols=None):
    if flag_cols:
        cols = len(_cols)
        colunas = _cols
        n_cols = len(_cols)
        cols = len(df_raw.columns)

    else:

        if fundamentals:
            colunas = ["date", "close", "ebitev", "roc", "dvd_yld", "eps", "roe", "pe", "ann", "g_factor", "factor_1"]
            n_cols = 11

        elif intraday:
            colunas = ["date", "close", "volume", "qtty"]
            n_cols = 4

        else:
            colunas = ["date", "close"]
            n_cols = 2

        cols = len(df_raw.columns)

    lista_df = []
    dft = df_raw.copy()
    lista_df = []
    for el in range(0, cols, n_cols):

        df_eq = dft.iloc[:, el:(el + n_cols)].copy()

        ticker = df_eq.columns[0]

        try:
            print(ticker)
            df_eq = df_eq.iloc[1:]
            df_eq.columns = colunas
            df_eq["ticker"] = ticker
            lista_df.append(df_eq)

        except:
            pass

    df = pd.concat(lista_df)
    df = df[~pd.isnull(df["date"])]
    df["class"] = df["date"].apply(lambda x: isinstance(x, int))
    df = df[df["class"] == False].drop("class", axis=1)

    if not intraday:

        df["date"] = df["date"].apply(lambda x: x.date())


    else:
        pass

    ticker_ind = ticker.split()[0] + ' Index'

    df["ref_index"] = index_name

    if fundamentals:
        df.drop("ann", axis=1, inplace=True)
        df = df[["date", "ticker", "ref_index", "close", 'ebitev', 'roc', 'dvd_yld', 'eps', 'roe', 'pe', 'g_factor',
                 'factor_1']]


    else:

        pass

    return df


def filter_Nticks(df_params, min_tick, max_ticks):
    df_params2 = df_params.groupby("date").agg({"ticker": "count"}).reset_index().sort_values("ticker")
    df_datas_filtered = df_params2[df_params2["ticker"] > min_tick]
    df_datas_filtered = df_datas_filtered[df_datas_filtered["ticker"] < max_ticks]

    df_datas_filtered["date_str"] = df_datas_filtered["date"].apply(lambda x: str(x))
    df_params["date_str"] = df_params["date"].apply(lambda x: str(x))

    df_params = pd.merge(df_params, df_datas_filtered[["date_str"]], left_on="date_str", right_on="date_str",
                         how="inner")
    df_params["date_short"] = df_params["date"].apply(lambda x: x.date())

    return df_params


class Gen_Statistics():
    '''
    input: lista longs, short de resultados anuais

    output: lista de stats,  e lista de res anuais

    '''

    def duration_drawdown(self, df2, col="pnl_total"):

        df = df2.copy()

        df['acc'] = df[col].cumsum()
        df['acc'] = df['acc'] - df['acc'].cummax()

        df['duration'] = list(chain.from_iterable((np.arange(len(list(j))) + 1).tolist() if i == 1 \
                                                      else [0] * len(list(j)) for i, j in groupby(df['acc'] != 0)))

        return df["duration"].max() + 1

    def gen_stats(self, anual_long, anual_short, pais='brazil'):

        long = anual_long.copy().drop(["param"], axis=1)[["year", "n_trades", "pnl_total"]]
        short = anual_short.copy().drop(["param"], axis=1)[["year", "n_trades", "pnl_total"]]

        long.columns = ['year', 'n_trades_long', 'pnl_total_long']
        short.columns = ['year', 'n_trades_short', 'pnl_total_short']

        long_short = pd.merge(long, short, left_on=["year"], right_on=["year"], how="outer").copy()

        long_short["n_trades_long"] = long_short["n_trades_long"].fillna(0)
        long_short["n_trades_short"] = long_short["n_trades_short"].fillna(0)

        long_short["pnl_total_long"] = long_short["pnl_total_long"].fillna(0)
        long_short["pnl_total_short"] = long_short["pnl_total_short"].fillna(0)

        long_short["n_trades"] = long_short["n_trades_long"] + long_short["n_trades_short"]
        long_short["pnl_total"] = long_short["pnl_total_long"] + long_short["pnl_total_short"]

        long_short = long_short[["year", "n_trades", "pnl_total"]]

        # filtering year control
        long_short = long_short[long_short["year"] != 2030]

        dur = self.duration_drawdown(long_short, col="pnl_total")

        param = anual_long["param"].iloc[0]
        npos = long_short[long_short["pnl_total"] > 0].__len__()

        mean_pos = long_short[long_short["pnl_total"] > 0]["pnl_total"].mean()
        mean_neg = long_short[long_short["pnl_total"] < 0]["pnl_total"].mean()
        pnl_total = long_short["pnl_total"].sum()
        pnl_total_long = long["pnl_total_long"].sum()
        pnl_total_short = short["pnl_total_short"].sum()
        n_trades = long_short["n_trades"].sum()
        pnl_trade = 100 * ((pnl_total / n_trades) / 100000)
        ratio = 100 * (npos / long_short.__len__())

        stats = pd.DataFrame({"pnl_total": [pnl_total],
                              "pnl_total_long": [pnl_total_long],
                              "pnl_total_short": [pnl_total_short],
                              "n_trades": [n_trades],
                              "pnl/trade(%)": [pnl_trade],
                              "pos/total(%)": [ratio],
                              # "mean_pos":[mean_pos],
                              # "mean_neg":[mean_neg],
                              "mean_pos/mean_neg": [mean_pos / abs(mean_neg)]
                                 , "duration_drawdown(Y)": [dur]
                              })

        # stats["pais"] = pais
        stats["param"] = param
        # print("gerando o status:")
        long_short.columns = ["year", "n_trades", "pnl_total_{}".format(pais)]
        long_short = long_short[["year", "pnl_total_{}".format(pais)]]

        return long_short, stats

    def gen_partial(self, rep_temp11):

        rep_temp11["pnl_acc"] = rep_temp11["pnl_total"].cumsum()
        rep_temp11["max_so_far"] = rep_temp11["pnl_acc"].cummax()
        rep_temp11["top_to_current"] = 100 * (rep_temp11["pnl_acc"] / rep_temp11["max_so_far"] - 1)
        rep_temp11["top_to_current"] = rep_temp11["top_to_current"].apply(lambda x: x if x < 0 else 0)

        lista_draws = []
        lista_pos = []

        daily_draw_max = rep_temp11["pnl_total"].min()
        daily_pos_max = rep_temp11["pnl_total"].max()

        lista_draws.append(rep_temp11["top_to_current"].min())
        for w in [30, 250, 750]:
            rep_temp11["DD_{}".format(w)] = rep_temp11["pnl_total"].rolling(w).sum()
            lista_draws.append(rep_temp11["DD_{}".format(w)].min())

        for w in [30, 250, 750]:
            rep_temp11["DD_{}".format(w)] = rep_temp11["pnl_total"].rolling(w).sum()
            lista_pos.append(rep_temp11["DD_{}".format(w)].max())

        rep_temp11["month"] = rep_temp11["date"].apply(lambda x: x.month)
        rep_temp11["year"] = rep_temp11["date"].apply(lambda x: x.year)

        mes = rep_temp11.groupby(["month", "year"]).agg({"pnl_total": "sum"}).reset_index()
        ano = rep_temp11.groupby(["year"]).agg({"pnl_total": "sum"}).reset_index()

        sharpe_mes = mes["pnl_total"].mean() / mes["pnl_total"].std(ddof=1)
        sharpe_ano = ano["pnl_total"].mean() / ano["pnl_total"].std(ddof=1)

        return sharpe_ano, sharpe_mes, lista_draws[0], lista_draws[1], lista_draws[2], lista_draws[3], lista_pos[0], \
               lista_pos[1], lista_pos[2], daily_draw_max, daily_pos_max

    def gen_statistcs_daily(self, rep):

        def get_avg_time(rep1):

            first = rep1.sort_values("date")["signal"].iloc[0]
            temp = rep1[(rep1["signal"] == first) | (pd.isnull(rep1["signal"]))]
            temp_signal = temp[temp["signal"] == first]
            tam_all = temp.__len__()
            tam_sigs = temp_signal.__len__()

            avg_time_trade_days = round(tam_all / tam_sigs)

            if first == 1:

                name = 'long'

            else:

                name = 'short'

            return avg_time_trade_days, name

        lista_avg_time = []
        lista_name = []
        lista_notional_long = []
        lista_notional_short = []

        if ((not isinstance(rep, list)) | (isinstance(rep, list) & (len(rep) == 1))):

            if isinstance(rep, list):
                rep = rep[0]

            else:
                pass

            first = rep.sort_values("date")["signal"].iloc[0]
            rep_temp1 = rep.copy()
            avg_time, name = get_avg_time(rep_temp1)
            lista_avg_time.append(avg_time)
            lista_name.append(name)
            rep_temp1["notional"] = rep_temp1["close"] * rep_temp1["qtty"]

            mean_notional = \
            rep_temp1[(pd.isnull(rep_temp1["signal"])) | (rep_temp1["signal"] == first)].groupby("date").agg(
                {"notional": "sum"}).reset_index()["notional"].mean()
            first = rep_temp1.sort_values("date")["signal"].iloc[0]

            if first == 1:

                lista_notional_long.append(mean_notional)
                lista_notional_short.append(0)
                avg_time_long = avg_time
                avg_time_short = None

            else:

                lista_notional_short.append(0)
                lista_notional_long.append(mean_notional)
                avg_time_short = avg_time
                avg_time_long = None

            rep_temp11 = rep_temp1.groupby("date").agg({"pnl_total": "sum"}).reset_index()


        else:

            i = 0
            for rep_temp in rep:

                rep_temp["notional"] = rep_temp["close"] * rep_temp["qtty"]
                first = rep_temp.sort_values("date")["signal"].iloc[0]
                mean_notional = \
                rep_temp[(pd.isnull(rep_temp["signal"])) | (rep_temp["signal"] == first)].groupby("date").agg(
                    {"notional": "sum"}).reset_index()["notional"].mean()
                avg_time, name = get_avg_time(rep_temp)
                lista_avg_time.append(avg_time)
                lista_name.append(name)

                len_5_pct = round(rep_temp["ticker"].unique().__len__() * 0.03)
                tops = rep_temp.groupby("ticker").agg({"pnl_total": "sum"}).reset_index().sort_values("pnl_total",
                                                                                                      ascending=False).iloc[
                       0:len_5_pct]["pnl_total"].sum()
                total_pnl_temp = rep_temp["pnl_total"].sum()
                mins = rep_temp.groupby("ticker").agg({"pnl_total": "sum"}).reset_index().sort_values("pnl_total",
                                                                                                      ascending=True).iloc[
                       0:len_5_pct]["pnl_total"].sum()

                if abs(total_pnl_temp) > 0:
                    concen_5 = 100 * (tops / (abs(total_pnl_temp)))
                    min_concen_5 = 100 * (mins / (abs(total_pnl_temp)))

                else:

                    concen_5 = 0
                    min_concen_5 = 0

                # se long
                if first == 1:

                    notional_long = mean_notional
                    avg_time_long = avg_time
                    max_long = concen_5
                    min_long = min_concen_5

                # se short
                else:

                    # print("veio no time shoooooooooooooort")
                    notional_short = mean_notional
                    avg_time_short = avg_time
                    max_short = concen_5
                    min_short = min_concen_5

                if i == 0:

                    rep_temp1 = rep_temp[["date", "pnl_total", "notional"]].copy()
                    rep_temp1 = rep_temp1.groupby("date").agg({"pnl_total": "sum", "notional": "sum"}).reset_index()
                    rep_temp1.columns = ["date", "pnl_total_1", "notional_1"]
                    one = rep_temp1.copy()

                else:

                    two = rep_temp[["date", "pnl_total", "notional"]].copy()
                    two = two.groupby("date").agg({"pnl_total": "sum", "notional": "sum"}).reset_index()
                    two.columns = ["date", "pnl_total_2", "notional_2"]

                    self.rep_temp1 = rep_temp1.copy()
                    self.two = two.copy()

                    rep_temp1 = pd.merge(rep_temp1, two, left_on=["date"], right_on=["date"], how="outer")
                    # rep_temp1.columns = ["date","pnl_total_1","pnl_total_2"]
                    rep_temp1.columns = ["date", "pnl_total_1", "notional_1", "pnl_total_2", "notional_2"]
                    rep_temp1["pnl_total"] = rep_temp1["pnl_total_1"] + rep_temp1["pnl_total_2"]

                i += 1

            rep_temp1["pnl_total"] = rep_temp1["pnl_total_1"] + rep_temp1["pnl_total_2"]
            rep_temp1["pnl_total_1"] = rep_temp1["pnl_total_1"].fillna(0)
            rep_temp1["pnl_total_2"] = rep_temp1["pnl_total_2"].fillna(0)

            rep_temp1["notional_1"] = rep_temp1["notional_1"].fillna(0)
            rep_temp1["notional_2"] = rep_temp1["notional_2"].fillna(0)

            rep_temp1["pnl_total"] = rep_temp1["pnl_total_1"] + rep_temp1["pnl_total_2"]
            rep_temp1 = rep_temp1[rep_temp1["date"] != datetime(2030, 1, 1).date()]

            self.cc = rep_temp1.copy()

            self.cc["delta"] = self.cc["notional_1"] + self.cc["notional_2"]

            avg_delta_short = self.cc[self.cc["delta"] < 0]["delta"].mean()
            avg_delta_long = self.cc[self.cc["delta"] > 0]["delta"].mean()
            avg_timPos_long = 100 * (self.cc[self.cc["delta"] > 0].__len__() / (self.cc.__len__()))

            rep_temp11 = rep_temp1.groupby("date").agg({"pnl_total": "sum"}).reset_index()

        return avg_time_long, avg_time_short, lista_name, notional_long, notional_short, rep_temp11, max_long, max_short, min_long, min_short, avg_delta_short, avg_delta_long, avg_timPos_long

    def gen_full_stats(self, anual_long, anual_short, daily_long, daily_short):

        a = [daily_long, daily_short]
        # a = [rep,rep]

        avg_time_long, avg_time_short, lista_name, notional_long, notional_short, rep_temp11, max_long, max_short, min_long, min_short, avg_delta_short, avg_delta_long, avg_timPos_long = self.gen_statistcs_daily(
            a)

        self.rep_temp11 = rep_temp11.copy()

        sharpe_ano, sharpe_mes, top_to_current, d0, d1, d2, p0, p1, p2, daily_draw_max, daily_pos_max = self.gen_partial(
            rep_temp11)

        res_anuais, res_stats = self.gen_stats(anual_long, anual_short)

        # avg_delta_short,avg_delta_long,avg_time_long

        res_stats2 = pd.DataFrame({"sharpe_ano": [sharpe_ano],
                                   "sharpe_mes": [sharpe_mes],
                                   "avg_days_long": [avg_time_long],
                                   "avg_days_short": [avg_time_short],
                                   "avg_notional_long": [notional_long],
                                   "avg_notional_short": [notional_short],
                                   "max_daily_p&L": [daily_pos_max],
                                   "min_daily_p&L": [daily_draw_max],
                                   "Min_Top_to_Crnt(%)": [top_to_current],
                                   "roll_MinP&L_30D": [d0],
                                   "roll_MinP&L_250D": [d1],
                                   "roll_MinP&L_750D": [d2],
                                   "roll_MaxP&L_30D": [p0],
                                   "roll_MaxP&L_250D": [p1],
                                   "roll_MaxP&L_750D": [p2],
                                   "3%Ticker_long_POS": [max_long],
                                   "3%Ticker_long_NEG": [min_long],
                                   "3%Ticker_short_POS": [max_short],
                                   "3%Ticker_short_NEG": [min_short],
                                   "avg_delta_long": [avg_delta_long],
                                   "avg_delta_short": [avg_delta_short],
                                   "avg_Period_long(%)": [avg_timPos_long]
                                   })

        res_stats2 = pd.concat([res_stats, res_stats2], axis=1)
        res_stats2.drop("param", axis=1, inplace=True)
        res_stats2["kelly_C"] = res_stats2.apply(
            lambda x: ((x['pos/total(%)'] / 100) * x['mean_pos/mean_neg'] - (1 - (x['pos/total(%)'] / 100))) / x[
                'mean_pos/mean_neg'], axis=1)
        return res_stats2, res_anuais


# from enterpython.tools import compute_vol
class bm():

    def __init__(self):

        self.sql_con = self.conect_database("backtesting")

    def conect_database(self, schema):

        host = "administrator.cpmflhyn4h6k.us-east-1.rds.amazonaws.com"
        pwd = "enteraws123&"
        user = "administrator"
        mysql_con  = create_engine("mysql+pymysql://{}:{}@{}/{}".format(user, pwd, host, schema))

        return mysql_con

    def aux_live_update(self):

        def format_insert_sql(df, insert=False):

            mysql_con = create_engine("mysql+pymysql://administrator:Enter123@192.168.0.5/{}".format('backtesting'))

            main_cols = ["date", "ticker", "close"]
            # all_cols  = ["date", "ticker","close","open","high","low","volume","source","div_adj","last_update"]
            all_cols = ["date", "ticker", "close"]

            cols_df = df.columns.tolist()
            cols_df = [el.lower() for el in cols_df]
            cols_to_drop = [el for el in cols_df if el not in main_cols]

            if len([el for el in main_cols if el not in cols_df]) != 0:

                print("Alguma coluna essencial faltando. Favor verificar")

            else:

                df = df.drop(cols_to_drop, axis=1)
                # df["open"] = 0
                # df["high"] = 0
                # df["low"] = 0
                # df["volume"] = 0
                # df["source"] = 0
                # df["div_adj"] = 1
                # df["last_update"] = str(datetime.now().date())

                if insert:
                    if df["date"].unique().tolist().__len__() != 1:

                        print("formato incorreto para inserir")


                    else:

                        print("1) apagando a data do mysql")

                        dt = df["date"].iloc[0]

                        mysql_con.execute("SET SQL_SAFE_UPDATES = 0")
                        mysql_con.execute("delete from backtesting.eqt_brazil_signals where date ='{}' ".format(dt))

                        print("2) iniciando uploadn no mysql")

                        tickers = df["ticker"].unique().tolist()

                        i = 1
                        for ticker in tickers:
                            print("i is: {}".format(i))
                            temp = df[df["ticker"] == ticker]
                            temp.to_sql(con=mysql_con, name='eqt_brazil_signals', if_exists='append', index=False)
                            i = i + 1

                        print("3) Fim do upload!")

                return df

        host = "administrator.cpmflhyn4h6k.us-east-1.rds.amazonaws.com"
        pwd = "enteraws123&"
        user = "administrator"
        a_mysql_con = create_engine("mysql+pymysql://{}:{}@{}/{}".format(user, pwd, host, "pnl"))
        mysql_con = create_engine("mysql+pymysql://administrator:Enter123@192.168.0.5/{}".format('backtesting'))

        precos_part = pd.read_sql('''select '{}' as date,ticker,close,close_d1 from (
                                    select date, ticker,close,close_d1 from (
                                    select ticker,date,price as 'close',close_d1
                                    from prices.live_prices
                                    where  (date < (select  max(date) from prices.live_prices))) as tab) as tab2
                                    where tab2.date = 
                                    (
                                    select max(date) from (
                                    select date, ticker,close from (
                                    select ticker,date,price as 'close'
                                    from prices.live_prices
                                    where  (date < (select  max(date) from prices.live_prices))) as tab) as tab2
                                    );'''.format(str(datetime.now().date())), a_mysql_con)

        eqts = pd.read_sql("select ticker from instruments.equities", a_mysql_con)

        # precos_part["ticker"] = precos_part["ticker"].apply(lambda x: x + ' BS Equity')

        df_ibx = precos_part[precos_part["ticker"] == 'IBXX']
        df_eqt = precos_part[precos_part["ticker"] != 'IBXX']

        df_eqt = pd.merge(df_eqt, eqts, left_on=["ticker"], right_on=["ticker"], how="inner")
        df_eqt["ticker"] = df_eqt["ticker"].apply(lambda x: x + ' BS Equity')

        df_ibx["ticker"] = 'IBX Index'

        df = pd.concat([df_eqt, df_ibx])

        df_d0 = df[["date", "ticker", "close"]]
        df_d1 = df[["date", "ticker", "close_d1"]]
        df_d1.columns = ["date", "ticker", "close"]

        date_max_back = str(
            pd.read_sql("select max(date) as date from backtesting.eqt_brazil_signals", mysql_con)["date"].iloc[0])

        date_max_carteira = str(
            pd.read_sql("select max(date) as date from pnl.position_pnl", a_mysql_con)["date"].iloc[0])

        df_d1["date"] = date_max_carteira

        date_max_live = str(df["date"].max())

        if ((date_max_back == date_max_live) | (date_max_back == date_max_carteira)):

            print("INICIANDO UPDATE NA TABELA DE BACKTEST")
            yy = format_insert_sql(df_d1, insert=True)
            yy = format_insert_sql(df_d0, insert=True)

        else:

            print("a data maxima em backtesting está estranha")

    # fill cdi: if days and missing on price data (and vice-versa), it fills the gaps on both
    # se for spread, addicionar o segundo carry
    def fill_carry_inf(self, df_precos, if_spread=False):

        if not if_spread:

            # obtaining CDI and ajusting the price
            df_cdi = pd.read_sql("""select date,value as taxa, (pow(1+(value/100),1/252) -1) as carry from curves.carry 
            where rate ='{}' order by date asc limit 0,{}""".format(self.rate_name, 10000), self.sql_con)

            dd = df_precos.groupby("date").agg({"close": "mean"}).reset_index()
            dd = df_precos.groupby("date").mean().reset_index()[["date", "close"]]

            df_cdi = pd.merge(dd, df_cdi, left_on="date", right_on="date", how="left")[["date", "taxa", "carry"]]
            df_cdi["carry"].fillna(method='ffill', inplace=True)
            df_cdi["taxa"].fillna(method='ffill', inplace=True)

            # self.df_cdi = df_cdi.copy()

            df_precos = pd.merge(df_precos, df_cdi, left_on="date", right_on="date", how="inner")

            return df_precos

        else:

            # CARRY1
            df_cdi = pd.read_sql("""select date,value as taxa1, (pow(1+(value/100),1/252) -1) as carry1 from curves.carry 
            where rate ='{}' order by date asc limit 0,{}""".format(self.rate_name1, 10000), self.sql_con)

            dd = df_precos.groupby("date").agg({"close": "mean"}).reset_index()
            dd = df_precos.groupby("date").mean().reset_index()[["date", "close"]]

            df_cdi = pd.merge(dd, df_cdi, left_on="date", right_on="date", how="left")[["date", "taxa1", "carry1"]]
            df_cdi["carry1"].fillna(method='ffill', inplace=True)
            df_cdi["taxa1"].fillna(method='ffill', inplace=True)

            df_precos = pd.merge(df_precos, df_cdi, left_on="date", right_on="date", how="inner")

            # CARRY2
            df_cdi = pd.read_sql("""select date,value as taxa2, (pow(1+(value/100),1/252) -1) as carry2 from curves.carry 
            where rate ='{}' order by date asc limit 0,{}""".format(self.rate_name2, 10000), self.sql_con)

            dd = df_precos.groupby("date").agg({"close": "mean"}).reset_index()
            dd = df_precos.groupby("date").mean().reset_index()[["date", "close"]]

            df_cdi = pd.merge(dd, df_cdi, left_on="date", right_on="date", how="left")[["date", "taxa2", "carry2"]]
            df_cdi["carry2"].fillna(method='ffill', inplace=True)
            df_cdi["taxa2"].fillna(method='ffill', inplace=True)

            df_precos = pd.merge(df_precos, df_cdi, left_on="date", right_on="date", how="inner")

            return df_precos

    def get_prices(self, start, ticker=None, tipo_price=['close'], date_start=None, date_end=None, flag_control=False,
                   offset_days=1, country='brazil', flag_index=True, flag_div=1, flag_crowd=False, flag_signals=False):

        tipos_2 = ','.join(tipo_price)

        if ticker is not None:
            if len(ticker) == 1:
                ticker.append('_')
            ticker2 = tuple(ticker)

        if flag_signals:
            print("veio nos sinaiiiiiiiiiiiiiiiiiiis")

            prices = pd.read_sql("SELECT * FROM backtesting.eqt_brazil_signals", self.sql_con)

            return prices

        # significa que o back_test está no modo de controle: Verifica apenas se há sinal no dia ! Verfica para quais tickers :????
        if not flag_index:

            if flag_control:
                print("veio no flag div gsgsdsdgs")
                prices = pd.read_sql("""select ticker,date,{} from backtesting.eqt_{}
                            where ticker = '{}'
                            order by date desc LIMIT {},{}""".format(tipos_2, country, ticker2, start, offset_days),
                                     self.sql_con)

            else:

                # back test historico
                if date_end is not None:

                    prices = pd.read_sql("""select ticker,date,{} from backtesting.eqt_{}
                                where date >= '{}' and date <= '{}'
                                and ticker in {}
                                order by date asc""".format(tipos_2, country, date_start, date_end, ticker2),
                                         self.sql_con)

                else:

                    print("veio no flag div")
                    prices = pd.read_sql("""select ticker,date,{} from backtesting.eqt_{}
                                where ticker in {} and div_adj = {}
                                order by date desc LIMIT 0,{}
                                """.format(tipos_2, country, ticker2, flag_div, start), self.sql_con)

                    prices = prices.iloc[(len(prices) - (offset_days)):]

                    # prices = pd.read_sql("""select ticker,date,{} from backtesting.eqt_{}
                    #             #where div_adj = {}
                    #             order by date desc
                    #             """.format(tipos_2,country,flag_div,start),self.sql_con)

            # retorna um dataframe ?
            return prices



        # Só está implementado para brasil !. Retorna os precos de todos os ticker que estao no no indice nos respectivos anos. Utilizad-se a tabela de composicao anual.
        else:

            if not flag_crowd:
                print("gsdsdsd")
                prices = pd.read_sql("""select bk.ticker,bk.date,bk.{} from backtesting.eqt_{} as bk
                            where bk.ticker in {} and div_adj = {}
                            order by bk.date desc LIMIT 0,{}
                            """.format(ticker2, start), self.sql_con)

                prices = prices.iloc[(len(prices) - (offset_days)):]

                return prices


            else:
                mysql_con_crowd = self.conect_database('crowdmonitor')
                prices = pd.read_sql("""select * from crowdmonitor.crowdbase
                            where ticker in {}
                            order by date desc LIMIT 0,{}
                            """.format(ticker2, start), mysql_con_crowd)

                prices = prices.iloc[(len(prices) - (offset_days)):]

                return prices

    # create df null for list
    def create_null_dfs(self, anual_rep, rep, tipos, param):

        anual_rep["param"] = param
        rep["param"] = param

        def help_none_df(df):

            cols = df.columns.tolist()
            df2 = pd.DataFrame(columns=cols)
            a = {}
            for col in cols:
                a[col] = 0
            df2.loc[0] = a
            return df2

        if len(tipos) == 1:

            if 'long' not in tipos:

                anual_long = help_none_df(anual_rep)
                daily_long = help_none_df(rep)

                # fixing
                daily_long["signal"] = 1
                daily_long["date"] = datetime(2030, 1, 1).date()
                anual_long["year"] = 2030
                anual_long["pnl_total"] = 0
                daily_long["pnl_total"] = 0

                return daily_long, anual_long

            else:

                anual_short = help_none_df(anual_rep)
                daily_short = help_none_df(rep)

                daily_short["pnl_total"] = 0
                anual_short["pnl_total"] = 0

                daily_short["signal"] = -1
                daily_short["date"] = datetime(2030, 1, 1).date()
                anual_short["year"] = 2030

                return daily_short, anual_short

        else:

            pass

    def fix_rep(self, anual_long, anual_short, daily_long, daily_short, lista_tipos, param):

        if len(lista_tipos) == 1:

            if 'long' not in lista_tipos:
                print("sdgsdjiogsd")
                daily_long, anual_long = self.create_null_dfs(anual_short, daily_short, lista_tipos, param)

            else:

                daily_short, anual_short = self.create_null_dfs(anual_long, daily_long, lista_tipos, param)

        else:

            pass

        anual_long["param"] = param
        anual_short["param"] = param

        return anual_long, anual_short, daily_long, daily_short

    # apenas itera long e short
    def train_test_backtest(self, df_params_full2, df_params2, param, tipos=['long', 'short']):

        # print("veio no inicio dos testessssss")

        def help_none_df(df):

            cols = df.columns.tolist()
            df2 = pd.DataFrame(columns=cols)
            a = {}
            for col in cols:
                a[col] = 0
            df2.loc[0] = a
            return df2

        for tipo in tipos:

            self.type_trades = [tipo]

            rep, daily_rep, anual_rep = self.backtest_run(df_params2, df_params_full2[
                df_params_full2["date"] >= df_params2["date"].min()], 1000, 1000)
            param = 'test'
            anual_rep["param"] = param

            res_anual = anual_rep[["year", "n_trades", "pnl_total", "param"]]

            if tipo == 'long':

                anual_long = anual_rep.copy()
                daily_long = rep.copy()

            else:

                anual_short = anual_rep.copy()
                daily_short = rep.copy()

                self.aa = anual_rep.copy()
                self.dd = rep.copy()

        if len(tipos) == 1:

            if 'long' not in tipos:

                anual_long = help_none_df(anual_rep)
                daily_long = help_none_df(rep)

                # fixing
                daily_long["signal"] = 1
                daily_long["date"] = datetime(2030, 1, 1).date()
                daily_long["pnl_total"] = 0

                # anuais
                anual_long["year"] = 2030
                anual_long["pnl_total"] = 0

                if daily_short.empty:
                    anual_short = help_none_df(anual_rep)
                    daily_short = help_none_df(rep)

                    # fixing
                    daily_short["signal"] = -1
                    daily_short["date"] = datetime(2030, 1, 1).date()
                    anual_short["year"] = 2030
                    anual_short["pnl_total"] = 0
                    daily_short["pnl_total"] = 0

            else:

                anual_short = help_none_df(anual_rep)
                daily_short = help_none_df(rep)

                daily_short["pnl_total"] = 0
                anual_short["pnl_total"] = 0

                daily_short["signal"] = -1
                daily_short["date"] = datetime(2030, 1, 1).date()
                anual_short["year"] = 2030

                if daily_long.empty:
                    print("veio no long emppppppppppppppppppppppppppty")
                    anual_long = help_none_df(anual_rep)
                    daily_long = help_none_df(rep)

                    # fixing
                    daily_long["signal"] = 1
                    daily_long["date"] = datetime(2030, 1, 1).date()
                    anual_long["year"] = 2030
                    anual_long["pnl_total"] = 0
                    daily_long["pnl_total"] = 0


        else:

            if daily_long.empty:
                print("veio no long emppppppppppppppppppppppppppty")
                anual_long = help_none_df(anual_rep)
                daily_long = help_none_df(rep)

                # fixing
                daily_long["signal"] = 1
                daily_long["date"] = datetime(2030, 1, 1).date()
                anual_long["year"] = 2030
                anual_long["pnl_total"] = 0
                daily_long["pnl_total"] = 0

            if daily_short.empty:
                anual_short = help_none_df(anual_rep)
                daily_short = help_none_df(rep)

                # fixing
                daily_short["signal"] = -1
                daily_short["date"] = datetime(2030, 1, 1).date()
                anual_short["year"] = 2030
                anual_short["pnl_total"] = 0
                daily_short["pnl_total"] = 0

            pass

        anual_long, anual_short, daily_long, daily_short = self.fix_rep(anual_long, anual_short, daily_long,
                                                                        daily_short, tipos, param)
        return anual_long, anual_short, daily_long, daily_short

    def fix_composition(self, df_precos, df=None, n_min=1):

        if df is None:
            df = pd.read_sql("SELECT * FROM backtesting.composition_detailed ", self.sql_con)

        df_precos["month"] = df_precos["date"].apply(lambda x: x.month)
        df_precos["year"] = df_precos["date"].apply(lambda x: x.year)

        # df = pd.read_excel("composition_full.xlsx")
        depara = pd.DataFrame(
            {"month": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "period": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]})

        df = df.sort_values(["period", "year"], ascending=[True, True])

        per = df[["period", "year"]].drop_duplicates()
        years_all = per.sort_values("year")["year"].unique().tolist()
        per["valor"] = 1

        lista_all = []
        for y in years_all:
            lista_all.append(per[per["year"] == y])

        per_ord = pd.concat(lista_all)

        def fix(mm):
            lista_count = []
            last_nan_index = 0
            last_not_nan_index = 0
            for el in range(len(mm)):
                if not (isinstance(mm["ticker"].iloc[el], str)):
                    lista_count.append(0)
                    last_nan_index = el

                else:
                    lista_count.append(mm.iloc[(last_nan_index + 1): (el + 1)].__len__())
                    last_not_nan_index = el

            mm["cout"] = lista_count
            return mm

        tickers = df["ticker"].unique().tolist()

        lista_final = []

        for ticker in tickers:

            vale = df[df["ticker"] == ticker]
            years = vale.sort_values("year")["year"].unique().tolist()

            lista = []
            for y in years:
                lista.append(vale[vale["year"] == y])

            vale2 = pd.concat(lista)
            mm = pd.merge(per_ord, vale2, left_on=["year", "period"], right_on=["year", "period"], how="left")
            mm = fix(mm)
            lista_final.append(mm)

        df_final = pd.concat(lista_final)

        df_precos = pd.merge(df_precos, depara, left_on=["month"], right_on=["month"], how="inner")

        fin = pd.merge(df_precos, df_final[["period", "year", "cout", "ticker"]], left_on=["period", "year", "ticker"],
                       right_on=["period", "year", "ticker"], how="inner")

        fin = fin[fin["cout"] > n_min].drop(['month', 'period', 'cout'], axis=1)

        return fin

    def get_indexCompos(self, index='IBX'):

        # ano anterior
        df = pd.read_sql(
            """SELECT bk.ticker,(YEAR(bk.date) + 0) as year FROM backtesting.composition_anual as bk where bk.index = '{}' """.format(
                index), self.sql_con)

        # ano corrente
        # df = pd.read_sql("""SELECT bk.ticker,(YEAR(bk.date) + 0) as year FROM backtesting.composition_anual as bk where bk.index = '{}' """.format(index),self.sql_con)
        return df

    # index characteristics
    def get_rateName(self):

        if self.index == 'IBX':
            return 'CDI', 100


        elif self.index == 'SPX':
            return 'fedfundS_2', 500


        elif self.index == 'AS51':

            return 'AS51', 200



        elif self.index == 'IPSA':
            return 'IPSA', 40



        elif self.index == 'MEXBOL':

            return 'MEXBOL', 40


        elif self.index == 'TOP40':
            # OLD: RETURN CDI
            return 'TOP40', 40



        elif self.index == 'KOSPI':
            return 'KOSPI', 40



        elif self.index == 'SXXE':

            return 'SXXE', 40


        elif self.index == 'GERMANY':

            return 'EUROPE', 40


        elif self.index == 'SPAIN':

            return 'EUROPE', 40


        elif self.index == 'ENGLAND':

            return 'ENGLAND', 40


        elif self.index == 'JAPAN':

            return 'JAPAN', 40


        elif self.index == 'NIGERIA':

            return 'NIGERIA', 40


        elif self.index == 'RUSSIA':

            return 'RUSSIA', 40


        elif self.index == 'SPTSX':

            return 'sptsx', 200


        elif self.index == 'CHINA':

            return 'CHINA', 200


        elif self.index == 'COLOMBIA':

            return 'COLOMBIA', 200


        elif self.index == 'TURKEY':

            return 'TURKEY', 200

        else:

            print("Rate not found for given country !")

    def joinLongShort(self, lista_types, lista_dailyTicker_rep, lista_daily_rep, lista_anual_rep):

        if len(lista_types) == 2:
            if lista_types[0] == 'long':
                short = lista_daily_rep[1].copy().drop("year", axis=1)
                long = lista_daily_rep[0].copy().drop("year", axis=1)

            else:
                short = lista_daily_rep[0].copy().drop("year", axis=1)
                long = lista_daily_rep[1].copy().drop("year", axis=1)

            short.columns = [col + '_short' if ((col != 'date')) else col for col in short.columns]
            long.columns = [col + '_long' if ((col != 'date')) else col for col in long.columns]
            long_short = pd.merge(long, short, left_on="date", right_on="date", how="outer")

            # long_short["carry_eqt"] = long_short["carry_eqt_long"] + long_short["carry_eqt_short"]
            long_short["carry_eqt"] = 0
            long_short["txBroker"] = long_short["txBroker_long"] + long_short["txBroker_short"]
            long_short["notional"] = long_short["notional_long"] + abs(long_short["notional_short"])

            hedge = False

            if not hedge:
                long_short_daily = long_short.drop(
                    ["carry_eqt_long", "carry_eqt_short", "txBroker_long", "txBroker_short", "daily_return_short",
                     "daily_return_long", "pnl_daily_nominal_Hedge_short", "pnl_daily_nominal_Hedge_long",
                     "carry_hedge_long", "carry_hedge_short",
                     'borrow_cost_long', 'pl_d-1_short', 'pl_d-1_long'], axis=1)

            else:
                long_short_daily = long_short.drop(
                    ["carry_eqt_long", "carry_eqt_short", "txBroker_long", "txBroker_short", "daily_return_short",
                     "daily_return_long", "pnl_daily_nominal_Hedge_short", "pnl_daily_nominal_Hedge_long",
                     'borrow_cost_long', 'pl_d-1_short', 'pl_d-1_long'], axis=1)

            long_short_daily["pnl_daily_nominal_EQT"] = long_short_daily["pnl_daily_nominal_EQT_long"] + \
                                                        long_short_daily["pnl_daily_nominal_EQT_short"]
            long_short_daily["pnl_total"] = long_short_daily["pnl_total_long"] + long_short_daily["pnl_total_short"]
            long_short_daily["borrow_cost"] = long_short_daily["borrow_cost_short"]
            long_short_daily["pl_d-1"] = long_short_daily["notional"].shift(1)
            long_short_daily["daily_return"] = 100 * (
                        (long_short_daily["pnl_total"] + long_short_daily["carry_eqt"]) / abs(
                    long_short_daily["pl_d-1"]))
            long_short_daily["daily_return_temp"] = (long_short_daily["daily_return"] / 100) + 1
            long_short_daily["acc_rateReturn"] = long_short_daily["daily_return_temp"].cumprod()
            long_short_daily["acc_rateReturn"] = 100 * (long_short_daily["acc_rateReturn"] - 1)
            long_short_daily.drop("daily_return_temp", axis=1, inplace=True)

            if lista_types[0] == 'long':
                short = lista_anual_rep[1].copy()
                long = lista_anual_rep[0].copy()

            else:
                short = lista_anual_rep[0].copy()
                long = lista_anual_rep[1].copy()

            short.columns = [col + '_short' if ((col != 'year') & (col != 'date')) else col for col in short.columns]
            long.columns = [col + '_long' if ((col != 'year') & (col != 'date')) else col for col in long.columns]
            print(long.head())
            long_short = pd.merge(long, short, left_on="year", right_on="year", how="outer")

            long_short["carry_eqt"] = long_short["carry_eqt_long"] + long_short["carry_eqt_short"]
            long_short["txBroker"] = long_short["txBroker_long"] + long_short["txBroker_short"]
            long_short["n_trades"] = long_short["n_trades_long"] + long_short["n_trades_short"]

            hedge = False
            if not hedge:
                long_short = long_short.drop(["carry_eqt_long", "carry_eqt_short", "txBroker_long", "txBroker_short",
                                              "pnl_daily_nominal_Hedge_short", "pnl_daily_nominal_Hedge_long",
                                              "carry_hedge_long", "carry_hedge_short", 'pnl/trade(%)_short',
                                              "pnl/trade_long", "pnl/trade(%)_long",
                                              'borrow_cost_long'], axis=1)

            else:
                long_short = long_short.drop(
                    ["carry_eqt_long", "carry_eqt_short", "txBroker_long", "txBroker_short", "pnl/trade_long",
                     "pnl_daily_nominal_Hedge_short", "pnl_daily_nominal_Hedge_long", 'pnl/trade(%)_short',
                     "pnl/trade(%)_long",
                     'pnl/trade_short',
                     'borrow_cost_long'], axis=1)

            long_short_anual = long_short[["year", "pnl_daily_nominal_EQT_long", "pnl_total_long",
                                           "pnl_daily_nominal_EQT_short", "pnl_total_short",
                                           "carry_eqt", "txBroker",
                                           "borrow_cost_short", "n_trades", "n_trades_long", "n_trades_short"]]

            long_short_anual["pnl_total"] = long_short_anual["pnl_total_long"] + long_short_anual["pnl_total_short"]
            long_short_anual["pnl/trade(%)"] = 100 * (
                        (long_short_anual["pnl_total"] / long_short_anual["n_trades"]) / 100000)
            return long_short_daily, long_short_anual


        else:
            print("No types to join")

    # receive daily backtesting data for several parameters: Coming from "generate report"
    # returns the resume of a backtest on a monthly
    def monthly_resume(self, days, notional=100000):

        if isinstance(days, list):
            days = pd.concat(days)

        days_long = days[days["type_trade"] == 'long']
        days_short = days[days["type_trade"] == 'short']

        days_long["month"] = days_long["date"].apply(lambda x: x.month)
        days_long["year"] = days_long["date"].apply(lambda x: x.year)

        days_short["month"] = days_short["date"].apply(lambda x: x.month)
        days_short["year"] = days_short["date"].apply(lambda x: x.year)

        days_long = days_long.groupby(["param", "year", "month"]).sum().reset_index().sort_values(
            ["param", "year", "month"], ascending=[True, True, True])
        days_short = days_long.groupby(["param", "year", "month"]).sum().reset_index().sort_values(
            ["param", "year", "month"], ascending=[True, True, True])

        days_long_pos = days_long[days_long["pnl_total"] >= 0].groupby(["param"]).agg({"pnl_daily_nominal_EQT": "count",
                                                                                       "pnl_total": ["mean", "sum"]})

        days_long_neg = days_long[days_long["pnl_total"] <= 0].groupby(["param"]).agg({"pnl_daily_nominal_EQT": "count",
                                                                                       "pnl_total": ["mean", "sum"]})

        days_long_neg = days_long_neg.reset_index()
        days_long_neg.columns = ["param", "neg_months", "avg_pnl_neg", "total_pnl_neg"]
        # days_long_neg["pnl_neg/trade(%)"] = 100*((days_long_neg["total_pnl_neg"]/days_long_neg["neg_months"])/notional)

        days_long_pos = days_long_pos.reset_index()
        days_long_pos.columns = ["param", "pos_months", "avg_pnl_pos", "total_pnl_pos"]
        # days_long_pos["pnl_pos/trade(%)"] = 100*((days_long_pos["total_pnl_pos"]/days_long_pos["pos_months"])/notional)

        month_res = pd.merge(days_long_pos, days_long_neg, left_on=["param"], right_on=["param"], how="outer")

        return month_res

    def conver_n_contracts_df(self, linha, **kwargs):

        notional_risco = kwargs["notional_risco"]
        variacao_chosen = kwargs["variacao_chosen"]
        notional = kwargs["notional"]

        # variação (%) escolhida: N bps. Nesse caso, 1bps
        var_taxa = (variacao_chosen) / (100)

        # VARIACAO EM CIMA DA TAXA REAL
        var_efetiva = var_taxa * (linha["px_entry"] / 100)

        # NOVA TAXA: APOS A VARIACAO
        nova_taxa = (linha["px_entry"] / 100) + var_efetiva

        # PU EM D0 (SE PRAZO = 1Y) ---> SEM VARIACAO
        pu_d0 = notional / (1 + (linha["px_entry"] / 100))

        # PU EM D0 (SE PRAZO = 1Y) ---> APOS APLICADO VARIACAO EFETIVA
        pu_d0_var = notional / (1 + ((linha["px_entry"] / 100) + var_efetiva))

        # diferenca em PU em DO (SE PRAZO = 1Y) (di 1Y exatamente) --> PNL para 0.01% de variaçã na taxa do DI 1Y
        # nao considera o cdi em cima ?
        pnl_contrato_d0 = -(pu_d0 - pu_d0_var)

        ''' 
        PNL EM TAXA --> PNL ANTERIOR ERA EM FINANCEIRO (CONVERTIDO PARA PU) ---> EM D0

        ********** INDEPENDE DA TAXA SPOT DE REFERENCIA. É APENAS UM PERCENTUAL DO PU (100K)
        '''

        # sempre sera o dv-01 !!!
        pnl_taxa = (nova_taxa / (linha["px_entry"] / 100)) - 1

        '''
        PNL em financeiro a partir do PNL em taxa --> O SINAL NEGATIVO SE DEVE AO FATO DE SE CONSIDERAR O PADRAO = COMPRADO EM TAXA
        Esse é meio que o "pnl target" desejado para aquela variação na taxa
        '''

        # pnl_financeiro_fromtaxa = -notional*pnl_taxa
        pnl_financeiro_fromtaxa = -notional * pnl_taxa

        '''
        multiplier = pnl_target (pnl obtido em porcentagem do PU) / pnl_contrato(pnl efetivo obtido por contrato para 1dv = 0.01%)
        Em outras palavras, quantos contratos eu preciso, dado o nivel de taxa, para que, um variacao de 0.01% produza o mesmo Pnl
        '''

        multiplier = pnl_financeiro_fromtaxa / pnl_contrato_d0

        '''
        numero do contratos necessário. 1unidade do multiplier = 100k. 
        N_unidades = notional_risco/1000k. 
        numero contratos = N_unidades*multiplier
        '''

        numero_contratos_final = np.sign(linha["signal_2"]) * abs(round((notional_risco / notional) * (multiplier)))

        return numero_contratos_final

    def gen_report_fast2(self, df_final, df_params_full, type_trade='long', notional=100000, borrow_cost=2,
                         postive_borrow_factor=0, flag_hedge=True, multi_tx=1, vol_adjust=False, vol_number=250):

        # print("porra do relatorio 2")
        if type_trade == 'long':

            module_signal = 1


        else:

            module_signal = -1

        df_entradas = df_final[df_final["signal"] == module_signal * 1]
        df_vendas = df_final[df_final["signal"] == module_signal * -1]
        df_resto = df_final[df_final["signal"] != module_signal * 1]

        if flag_hedge:

            multi_hedge = 1

        else:

            multi_hedge = 0

        daily_cost = ((1 + (borrow_cost / 100)) ** (1 / 252)) - 1

        if not vol_adjust:

            df_entradas = pd.merge(df_entradas, df_params_full[["date", "ticker", "close", "close_index"]],
                                   left_on=["date", "ticker"], right_on=["date", "ticker"], how="inner")
            df_entradas.columns = ["signal", "date", "ticker", "px_entry", "px_entry_hedge"]
            df_entradas["qtty"] = round(notional / (module_signal * df_entradas["px_entry"]))
            # df_entradas["qtty"] = notional/(module_signal*df_entradas["px_entry"])

        else:

            # df_entradas = pd.merge(df_entradas,df_params_full[["date","ticker","close","close_index","vol_{}".format(vol_number),"vol_mean","flag_liq"]],left_on=["date","ticker"],right_on=["date","ticker"],how="inner")
            # df_entradas.columns = ["signal","date","ticker","px_entry","px_entry_hedge","vol_entry","vol_mean","flag_liq"]

            df_entradas = pd.merge(df_entradas, df_params_full[
                ["date", "ticker", "close", "close_index", "vol_{}".format(vol_number), "flag_liq"]],
                                   left_on=["date", "ticker"], right_on=["date", "ticker"], how="inner")
            df_entradas.columns = ["signal", "date", "ticker", "px_entry", "px_entry_hedge", "vol_entry", "flag_liq"]

            self.df_entradas = df_entradas.copy()

            def help_notional(linha):

                if linha["flag_liq"] == True:

                    return 100000

                else:

                    return 100000 / (linha["vol_entry"])

            df_entradas["notional"] = df_entradas.apply(help_notional, axis=1)

            # df_entradas["notional"] = 100000*((1)/(0.2/df_entradas["vol_mean"]))
            # df_entradas["qtty"] = round(df_entradas["notional"]/(module_signal*df_entradas["px_entry"]))
            df_entradas["qtty"] = df_entradas["notional"] / (module_signal * df_entradas["px_entry"])

        if flag_hedge:

            if not vol_adjust:

                df_entradas["qtty_hedge"] = round(notional / (-module_signal * df_entradas["px_entry_hedge"]))

            else:

                df_entradas["qtty_hedge"] = round(
                    df_entradas["notional"] / (-module_signal * df_entradas["px_entry_hedge"]))
                # df_entradas["qtty"] = round(df_entradas["notional"]/(module_signal*df_entradas["px_entry"]))#
        else:

            df_entradas["qtty_hedge"] = 0
            df_entradas["qtty_hedge"] = None

        df_all = pd.concat([df_entradas, df_resto]).sort_values(["ticker", "date"], ascending=[True, True])
        df_all = df_all.replace((np.nan, ''), (None, None))

        df_all["px_entry"] = df_all["px_entry"].fillna(method='ffill')
        df_all["qtty"] = df_all["qtty"].fillna(method='ffill')

        if flag_hedge:
            df_all["px_entry_hedge"] = df_all["px_entry_hedge"].fillna(method='ffill')
            df_all["qtty_hedge"] = df_all["qtty_hedge"].fillna(method='ffill')

        else:
            df_all["px_entry_hedge"] = 0
            df_all["qtty_hedge"] = 0

        df_all2 = pd.merge(df_all, df_params_full[
            ["date", "ticker", "close", "close_d-1", "close_index", "close_d-1_index", "carry"]],
                           left_on=["date", "ticker"], right_on=["date", "ticker"], how="inner")

        entradas_saidas = df_all2[~pd.isnull(df_all2["signal"])]
        positions = df_all2[pd.isnull(df_all2["signal"])]

        entradas_saidas["txBrokerEqt"] = (-1) * abs(
            entradas_saidas["close"] * abs(entradas_saidas["qtty"]) * 0.001 * multi_tx)

        if flag_hedge:

            entradas_saidas["txBrokerHedge"] = (-1) * abs(
                entradas_saidas["close_index"] * abs(entradas_saidas["qtty_hedge"]) * 0.0005 * multi_tx)

        else:

            entradas_saidas["txBrokerHedge"] = 0

        positions["txBrokerEqt"] = 0
        positions["txBrokerHedge"] = 0
        positions["txBroker"] = 0
        entradas_saidas["txBroker"] = entradas_saidas["txBrokerEqt"] + entradas_saidas["txBrokerHedge"]
        entradas_saidas["pnl_daily_nominal_Hedge"] = 0
        entradas_saidas["pnl_daily_nominal_EQT"] = 0
        entradas_saidas["carry_eqt"] = 0
        entradas_saidas["carry_hedge"] = 0
        entradas_saidas["borrow_cost_hedge"] = 0
        # borrow cost eqt
        if type_trade == 'long':

            entradas_saidas["borrow_cost"] = 0

        else:

            entradas_saidas_p1 = entradas_saidas[entradas_saidas["signal"] == module_signal]
            entradas_saidas_p2 = entradas_saidas[entradas_saidas["signal"] == -module_signal]

            entradas_saidas_p1["borrow_cost"] = 1 * module_signal * 1 * abs(
                entradas_saidas_p1["qtty"] * entradas_saidas_p1["px_entry"] * daily_cost)
            entradas_saidas_p2["borrow_cost"] = 0
            entradas_saidas = pd.concat([entradas_saidas_p1, entradas_saidas_p2])

        # start = time.time()
        positions["pnl_daily_nominal_EQT"] = (positions["close"] - positions["close_d-1"]) * positions["qtty"]
        positions["pnl_daily_nominal_Hedge"] = (positions["close_index"] - positions["close_d-1_index"]) * positions[
            "qtty_hedge"]
        positions["carry_hedge"] = (-1) * positions["close_index"] * positions["qtty_hedge"] * positions["carry"]
        positions["carry_eqt"] = (-1) * positions["close"] * positions["qtty"] * positions["carry"]

        # hedge borrow
        if flag_hedge:

            if type_trade == 'long':

                positions["borrow_cost_hedge"] = -1 * module_signal * abs(
                    positions["qtty_hedge"] * positions["px_entry_hedge"] * daily_cost)

            else:

                positions["borrow_cost_hedge"] = 0

        else:

            positions["borrow_cost_hedge"] = 0

        if type_trade == 'long':

            positions["borrow_cost"] = 1 * module_signal * postive_borrow_factor * abs(
                positions["qtty"] * positions["px_entry"] * daily_cost)

        else:

            positions["borrow_cost"] = 1 * module_signal * 1 * abs(
                positions["qtty"] * positions["px_entry"] * daily_cost)

        df_all = pd.concat([entradas_saidas, positions]).sort_values(["ticker", "date"], ascending=[True, True])
        df_all["pnl_cdi_eqt"] = df_all["pnl_daily_nominal_EQT"] + df_all["carry_eqt"] + df_all["borrow_cost"]
        df_all["pnl_cdi_hedge"] = df_all["pnl_daily_nominal_Hedge"] + df_all["carry_hedge"] + df_all[
            "borrow_cost_hedge"]
        df_all["pnl_total"] = df_all["pnl_cdi_hedge"] + df_all["pnl_cdi_eqt"] + df_all["txBroker"]

        daily_rep = df_all.groupby(["date"]).agg({"pnl_daily_nominal_EQT": "sum", "pnl_daily_nominal_Hedge": "sum",
                                                  "txBroker": "sum", "borrow_cost": "sum", "borrow_cost_hedge": "sum",
                                                  "carry_eqt": "sum", "carry_hedge": "sum",
                                                  "pnl_total": "sum"}).reset_index()

        daily_rep["year"] = daily_rep["date"].apply(lambda x: x.year)
        anual_rep = daily_rep.groupby("year").agg({"pnl_daily_nominal_EQT": "sum",
                                                   "pnl_daily_nominal_Hedge": "sum",
                                                   "txBroker": "sum",
                                                   "borrow_cost": "sum",
                                                   "borrow_cost_hedge": "sum",
                                                   "carry_eqt": "sum",
                                                   "carry_hedge": "sum",
                                                   "pnl_total": "sum"}).reset_index()

        df_all["year"] = df_all["date"].apply(lambda x: x.year)
        trades = df_all[(df_all["signal"] == (-module_signal * 1))]
        trades2 = trades.groupby("year").agg({"ticker": "count"}).reset_index()
        trades2.rename(columns={'ticker': 'n_trades'}, inplace=True)

        anual_rep = pd.merge(anual_rep, trades2, left_on="year", right_on="year", how="left")
        anual_rep["pnl/trade"] = anual_rep["pnl_total"] / anual_rep['n_trades']
        anual_rep["pnl/trade(%)"] = 100 * (anual_rep["pnl/trade"] / 100000)

        return df_all, daily_rep, anual_rep

    # type 1, with control variable and rebalance(dias corridos).
    # No external control variables
    def generate_signal_fast(self, type_trade, rank_in, rank_out, lista_lista_buydn_cond, lista_lista_selldn_cond,
                             lista_lista_buyd0_cond, lista_lista_selld0_cond, lista_tickers, df_params_full, datas,
                             rebalance_frequence=20):

        lista_df = []

        if type_trade == 'long':

            module_signal = 1

        else:
            module_signal = -1

        count_days = 0
        data_max = max(datas)

        for el in range(len(lista_tickers)):

            flag_init = True
            lista_signal = []
            datas_ticker = []
            pos = False
            ticker = lista_tickers[el]

            count_days = 0

            # itera nas datas
            for el2 in range(len(lista_lista_buyd0_cond[el])):

                if not flag_init:

                    # nao tem posicoa, verifica se compra indepenen do sinal de rebalance
                    if not pos:

                        count_days += 1

                        if (((count_days - 1) % rebalance_frequence) == 0):

                            # nao tem posicao e esta no rank_in: compra. Nao pode ter sinal de compra na ultima data pois nao tera como zerar (e nem como identificar para zerar virtualmente depois)
                            # print("testando o tamanho do rank: {}".format(lista_lista_buydn_cond[el][el2]))
                            if (ticker in rank_in[el2]) & (not pos) & lista_lista_buydn_cond[el][el2] & (
                            not (lista_lista_selldn_cond[el][el2])):

                                if datas[el2] != data_max:

                                    pos = True
                                    # count_days += 1
                                    lista_signal.append(1 * module_signal)
                                    datas_ticker.append(datas[el2])

                                else:

                                    pass

                            else:

                                pass

                        else:

                            if (pos):

                                lista_signal.append(None)
                                datas_ticker.append(datas[el2])

                            else:

                                lista_signal.append(2)
                                datas_ticker.append(datas[el2])

                    else:

                        # tem posicao, incrementa a contagem
                        count_days += 1
                        # if rebalnace
                        if (((count_days - 1) % rebalance_frequence) == 0):

                            if ((ticker not in rank_out[el2]) | (lista_lista_selldn_cond[el][el2])) & (pos):

                                pos = False

                                lista_signal.append(-1 * module_signal)
                                lista_signal.append(None)
                                datas_ticker.append(datas[el2])
                                datas_ticker.append(datas[el2])


                            # esta no rank_in e ja tem posicao: hold
                            elif (ticker in rank_in[el2]) & (pos):

                                datas_ticker.append(datas[el2])
                                lista_signal.append(None)


                            # ja tem posicao e esta no rank_out: hold
                            elif (ticker in rank_out[el2]) & (pos):

                                lista_signal.append(None)
                                datas_ticker.append(datas[el2])

                            else:

                                pass


                        # if not rebalance, hold
                        else:
                            if (pos):
                                lista_signal.append(None)
                                datas_ticker.append(datas[el2])

                            else:

                                pass


                else:

                    flag_init = False

                    count_days += 1
                    if (((count_days - 1) % rebalance_frequence) == 0):

                        # nao tem posicao e esta no rank_in: compra
                        if (ticker in rank_in[el2]) & (not pos) & lista_lista_buyd0_cond[el][el2] & (
                        not (lista_lista_selld0_cond[el][el2])):

                            # count_days += 1
                            pos = True
                            lista_signal.append(1 * module_signal)
                            datas_ticker.append(datas[el2])

                        else:

                            pass

                    else:

                        if (pos):

                            lista_signal.append(None)
                            datas_ticker.append(datas[el2])

                        else:

                            pass

            if pos == True:
                lista_signal.append(-1 * module_signal)
                datas_ticker.append(datas[el2])

            df = pd.DataFrame({"signal": lista_signal, "date": datas_ticker})
            df["ticker"] = ticker
            lista_df.append(df)
            df_final = pd.concat(lista_df)

        return lista_df, df_final

    def gen_report_fast(self, df_final, df_params_full, type_trade='long', notional=100000, borrow_cost=2,
                        postive_borrow_factor=0, flag_hedge=True, multi_taxa=1):

        if type_trade == 'long':

            module_signal = 1

        else:

            module_signal = -1

        df_entradas = df_final[df_final["signal"] == module_signal * 1]
        df_vendas = df_final[df_final["signal"] == module_signal * -1]
        df_resto = df_final[df_final["signal"] != module_signal * 1]

        if flag_hedge:

            multi_hedge = 1

        else:

            multi_hedge = 0

        daily_cost = ((1 + (borrow_cost / 100)) ** (1 / 252)) - 1

        df_entradas = pd.merge(df_entradas, df_params_full[["date", "ticker", "close", "close_index"]],
                               left_on=["date", "ticker"], right_on=["date", "ticker"], how="inner")
        df_entradas.columns = ["signal", "date", "ticker", "px_entry", "px_entry_hedge"]

        df_entradas["qtty"] = round(notional / (module_signal * df_entradas["px_entry"]))

        if flag_hedge:

            df_entradas["qtty_hedge"] = round(notional / (-module_signal * df_entradas["px_entry_hedge"]))

        else:

            df_entradas["qtty_hedge"] = 0
            df_entradas["qtty_hedge"] = None

        df_all = pd.concat([df_entradas, df_resto]).sort_values(["ticker", "date"], ascending=[True, True])
        df_all = df_all.replace((np.nan, ''), (None, None))

        df_all["px_entry"] = df_all["px_entry"].fillna(method='ffill')
        df_all["qtty"] = df_all["qtty"].fillna(method='ffill')

        if flag_hedge:
            df_all["px_entry_hedge"] = df_all["px_entry_hedge"].fillna(method='ffill')
            df_all["qtty_hedge"] = df_all["qtty_hedge"].fillna(method='ffill')

        else:
            pass

        df_all2 = pd.merge(df_all, df_params_full[
            ["date", "ticker", "close", "close_d-1", "close_index", "close_d-1_index", "carry"]],
                           left_on=["date", "ticker"], right_on=["date", "ticker"], how="inner")

        entradas_saidas = df_all2[~pd.isnull(df_all2["signal"])]
        positions = df_all2[pd.isnull(df_all2["signal"])]

        entradas_saidas["txBrokerEqt"] = (-1) * abs(
            entradas_saidas["close"] * abs(entradas_saidas["qtty"]) * 0.001 * multi_tx)

        if flag_hedge:

            entradas_saidas["txBrokerHedge"] = (-1) * abs(
                entradas_saidas["close_index"] * abs(entradas_saidas["qtty_hedge"]) * 0.0005 * multi_tx)


        else:

            entradas_saidas["txBrokerHedge"] = 0

        positions["txBrokerEqt"] = 0
        positions["txBrokerHedge"] = 0
        positions["txBroker"] = 0
        entradas_saidas["txBroker"] = entradas_saidas["txBrokerEqt"] + entradas_saidas["txBrokerHedge"]
        entradas_saidas["pnl_daily_nominal_Hedge"] = 0
        entradas_saidas["pnl_daily_nominal_EQT"] = 0
        entradas_saidas["carry_eqt"] = 0
        entradas_saidas["carry_hedge"] = 0
        entradas_saidas["borrow_cost_hedge"] = 0
        # borrow cost eqt
        if type_trade == 'long':

            entradas_saidas["borrow_cost"] = 0

        else:

            entradas_saidas_p1 = entradas_saidas[entradas_saidas["signal"] == module_signal]
            entradas_saidas_p2 = entradas_saidas[entradas_saidas["signal"] == -module_signal]

            entradas_saidas_p1["borrow_cost"] = 1 * module_signal * 1 * abs(
                entradas_saidas_p1["qtty"] * entradas_saidas_p1["px_entry"] * daily_cost)
            entradas_saidas_p2["borrow_cost"] = 0
            entradas_saidas = pd.concat([entradas_saidas_p1, entradas_saidas_p2])

        # start = time.time()
        positions["pnl_daily_nominal_EQT"] = (positions["close"] - positions["close_d-1"]) * positions["qtty"]
        positions["pnl_daily_nominal_Hedge"] = (positions["close_index"] - positions["close_d-1_index"]) * positions[
            "qtty_hedge"]
        positions["carry_hedge"] = (-1) * positions["close_index"] * positions["qtty_hedge"] * positions["carry"]
        positions["carry_eqt"] = (-1) * positions["close"] * positions["qtty"] * positions["carry"]

        # hedge borrow
        if flag_hedge:

            if type_trade == 'long':

                positions["borrow_cost_hedge"] = -1 * module_signal * abs(
                    positions["qtty_hedge"] * positions["px_entry_hedge"] * daily_cost)

            else:

                positions["borrow_cost_hedge"] = 0

        else:
            positions["borrow_cost_hedge"] = 0

        if type_trade == 'long':

            positions["borrow_cost"] = 1 * module_signal * postive_borrow_factor * abs(
                positions["qtty"] * positions["px_entry"] * daily_cost)

        else:

            positions["borrow_cost"] = 1 * module_signal * 1 * abs(
                positions["qtty"] * positions["px_entry"] * daily_cost)

        df_all = pd.concat([entradas_saidas, positions]).sort_values(["ticker", "date"], ascending=[True, True])

        df_all["pnl_cdi_eqt"] = df_all["pnl_daily_nominal_EQT"] + df_all["carry_eqt"] + df_all["borrow_cost"]
        df_all["pnl_cdi_hedge"] = df_all["pnl_daily_nominal_Hedge"] + df_all["carry_hedge"] + df_all[
            "borrow_cost_hedge"]
        df_all["pnl_total"] = df_all["pnl_cdi_hedge"] + df_all["pnl_cdi_eqt"] + df_all["txBroker"]

        daily_rep = df_all.groupby(["date"]).agg({"pnl_daily_nominal_EQT": "sum", "pnl_daily_nominal_Hedge": "sum",
                                                  "txBroker": "sum", "borrow_cost": "sum", "borrow_cost_hedge": "sum",
                                                  "carry_eqt": "sum", "carry_hedge": "sum",
                                                  "pnl_total": "sum"}).reset_index()

        daily_rep["year"] = daily_rep["date"].apply(lambda x: x.year)
        anual_rep = daily_rep.groupby("year").agg({"pnl_daily_nominal_EQT": "sum",
                                                   "pnl_daily_nominal_Hedge": "sum",
                                                   "txBroker": "sum",
                                                   "borrow_cost": "sum",
                                                   "borrow_cost_hedge": "sum",
                                                   "carry_eqt": "sum",
                                                   "carry_hedge": "sum",
                                                   "pnl_total": "sum"}).reset_index()

        return df_all, daily_rep, anual_rep

    def gen_report(self, lista_df, df_params_full, type_trade='long', notional=100000, borrow_cost=3, hedge=True,
                   di=False, multi_tx=1, if_pu_adjust=False):

        df_params_full = df_params_full.sort_values("date")
        notional = notional
        type_trade = type_trade
        daily_cost = ((1 + (borrow_cost / 100)) ** (1 / 252)) - 1

        if type_trade == 'long':
            module_signal = 1


        else:
            module_signal = -1

        ###################################### defiining helpers
        ### long
        def help_qtty_long(linha):
            if linha["signal"] == 1:
                return linha["signal"] * round(notional / linha["close"])

            else:
                return None

        # long
        def help_px_long(linha):
            if linha["signal"] == 1:
                return linha["close"]

            else:
                return None

        # px_entry_long - Hedge
        def help_px_long_hedge(linha):
            if linha["signal"] == 1:
                return linha["close_index"]

            else:
                return None

        #### short
        def help_qtty_long(linha):
            if linha["signal"] == 1:
                return linha["signal"] * round(notional / linha["close"])

            else:
                return None

        def help_px_short(linha):
            if linha["signal"] == -1:
                return linha["close"]

            else:
                return None

        def help_px_short_hedge(linha):
            if linha["signal"] == -1:
                return linha["close_index"]

            else:
                return None

        def help_txBroker_eqt(linha):
            if (linha["signal"] == -1) | (linha["signal"] == 1):
                return (-1) * abs(linha["close"] * abs(linha["qtty"]) * 0.001 * multi_tx)

            else:
                return 0

        def help_txBroker_di(linha):

            if (linha["signal"] == -1) | (linha["signal"] == 1):
                # 2*-0.00016499999999999848
                # return (-1)*abs(linha["close"]*(0.01/100))
                return abs(linha["notional"]) * (2 * -0.00016499999999999848)

            else:
                return 0

        def help_txBroker_hedge(linha):
            if (linha["signal"] == -1) | (linha["signal"] == 1):
                return (-1) * abs(linha["close_index"] * abs(linha["qtty_hedge"]) * 0.0005 * multi_tx)

            else:
                return 0

        ################################### generating Report
        lista_report = []
        if type_trade == 'long':

            for df in lista_df:

                df = df[df["signal"] != 2]
                df = df.sort_values(["date", "signal"], ascending=["True", "True"])
                if not df.empty:
                    if np.isnan(df["signal"].iloc[-1]):
                        df['signal'] = np.where(df['date'] >= df["date"].max(), -1 * module_signal, df["signal"])

                if (not df.empty) & (df.__len__() > 1):

                    df_report = pd.merge(df, df_params_full[
                        ["date", "ticker", "close", "close_d-1", "close_index", "close_d-1_index", "carry"]],
                                         left_on=["date", "ticker"], right_on=["date", "ticker"], how="inner")

                    # start of filling NAs
                    df_report["px_entry"] = df_report.apply(help_px_long, axis=1)
                    df_report["px_entry"].fillna(method='ffill', inplace=True)
                    df_report["px_entry_hedge"] = df_report.apply(help_px_long_hedge, axis=1)
                    df_report["px_entry_hedge"].fillna(method='ffill', inplace=True)
                    df_report["signal_2"] = df_report["signal"].fillna(method='ffill')

                    # end of filling NAs
                    if not if_pu_adjust:

                        df_report["qtty"] = df_report["signal_2"] * round(notional / df_report["px_entry"])

                    else:

                        df_report["qtty"] = df_report.apply(self.conver_n_contracts_df, notional_risco=1000000,
                                                            variacao_chosen=0.01, notional=100000, axis=1)

                    df_report["qtty_hedge"] = (-1) * df_report["signal_2"] * round(
                        notional / df_report["px_entry_hedge"])
                    df_report["txBrokerHedge"] = df_report.apply(help_txBroker_hedge, axis=1)
                    df_report["txBrokerEqt"] = df_report.apply(help_txBroker_eqt, axis=1)

                    if hedge:

                        df_report["txBroker"] = df_report["txBrokerEqt"] + df_report["txBrokerHedge"]
                        df_report["notional"] = (df_report["qtty"] * df_report["close"]) + abs(
                            (df_report["qtty_hedge"] * df_report["close_index"]))


                    else:
                        df_report["txBrokerHedge"] = 0
                        df_report["txBroker"] = df_report["txBrokerEqt"]
                        df_report["notional"] = (df_report["qtty"] * df_report["close"])

                    lista_report.append(df_report)


        else:
            for df in lista_df:
                df = df[df["signal"] != 2]
                df = df.sort_values(["date", "signal"], ascending=["True", "True"])
                if not df.empty:
                    if np.isnan(df["signal"].iloc[-1]):
                        df['signal'] = np.where(df['date'] >= df["date"].max(), -1 * module_signal, df["signal"])

                if (not df.empty) & (df.__len__() > 1):
                    df_report = pd.merge(df_params_full[
                                             ["date", "ticker", "close", "close_d-1", "close_index", "close_d-1_index",
                                              "carry"]], df, left_on=["date", "ticker"], right_on=["date", "ticker"],
                                         how="inner")
                    df_report["px_entry"] = df_report.apply(help_px_short, axis=1)
                    df_report["px_entry"].fillna(method='ffill', inplace=True)
                    df_report["px_entry_hedge"] = df_report.apply(help_px_short_hedge, axis=1)
                    df_report["px_entry_hedge"].fillna(method='ffill', inplace=True)
                    df_report["signal_2"] = df_report["signal"].fillna(method='ffill')

                    # end of filling NAs
                    if not if_pu_adjust:

                        df_report["qtty"] = df_report["signal_2"] * round(notional / df_report["px_entry"])

                    else:

                        df_report["qtty"] = df_report.apply(self.conver_n_contracts_df, notional_risco=1000000,
                                                            variacao_chosen=0.01, notional=100000, axis=1)

                    df_report["qtty_hedge"] = (-1) * df_report["signal_2"] * round(
                        notional / df_report["px_entry_hedge"])
                    df_report["txBrokerHedge"] = df_report.apply(help_txBroker_hedge, axis=1)
                    df_report["txBrokerEqt"] = df_report.apply(help_txBroker_eqt, axis=1)

                    if hedge:

                        df_report["txBroker"] = df_report["txBrokerEqt"] + df_report["txBrokerHedge"]
                        df_report["notional"] = (df_report["qtty"] * df_report["close"]) + abs(
                            (df_report["qtty_hedge"] * df_report["close_index"]))


                    else:
                        df_report["txBrokerHedge"] = 0
                        df_report["txBroker"] = df_report["txBrokerEqt"]
                        df_report["notional"] = (df_report["qtty"] * df_report["close"])
                    lista_report.append(df_report)

        rep = pd.concat(lista_report)
        rep["pnl_daily_nominal_EQT"] = (rep["close"] - rep["close_d-1"]) * rep["qtty"]
        rep["pnl_daily_nominal_Hedge"] = (rep["close_index"] - rep["close_d-1_index"]) * rep["qtty_hedge"]
        rep["carry_eqt"] = (-1) * rep["close"] * rep["qtty"] * rep["carry"]
        rep["carry_hedge"] = (-1) * rep["close_index"] * rep["qtty_hedge"] * rep["carry"]

        # if long, doesnt pay borrow coast
        if type_trade == 'long':

            if hedge:
                rep["pnl_daily_nominal_Hedge"] = rep.apply(
                    lambda x: 0 if x["signal"] == 1 else x['pnl_daily_nominal_Hedge'], axis=1)
                rep["carry_hedge"] = rep.apply(lambda x: 0 if x["signal"] == 1 else x['carry_hedge'], axis=1)
                rep["borrow_cost_hedge"] = rep.apply(
                    lambda x: -abs(x["qtty_hedge"] * x["px_entry_hedge"] * daily_cost if x["signal"] == 1 else 0),
                    axis=1)
                # rep["borrow_cost"] = rep.apply(lambda x: 0 if x["borrow_cost"] == 1 else x['carry_eqt'],axis=1)
            else:
                rep["pnl_daily_nominal_Hedge"] = 0
                rep["carry_hedge"] = 0
                rep["borrow_cost_hedge"] = 0

            rep["pnl_daily_nominal_EQT"] = rep.apply(lambda x: 0 if x["signal"] == 1 else x['pnl_daily_nominal_EQT'],
                                                     axis=1)
            rep["carry_eqt"] = rep.apply(lambda x: 0 if x["signal"] == 1 else x['carry_eqt'], axis=1)
            rep["borrow_cost"] = 0

        # if short, does pay borrow coast
        else:

            if hedge:
                rep["pnl_daily_nominal_Hedge"] = rep.apply(
                    lambda x: 0 if x["signal"] == -1 else x['pnl_daily_nominal_Hedge'], axis=1)
                rep["carry_hedge"] = rep.apply(lambda x: 0 if x["signal"] == -1 else x['carry_hedge'], axis=1)
                rep["borrow_cost_hedge"] = 0

            else:

                rep["pnl_daily_nominal_Hedge"] = 0
                rep["carry_hedge"] = 0
                rep["borrow_cost_hedge"] = 0

            rep["pnl_daily_nominal_EQT"] = rep.apply(lambda x: 0 if x["signal"] == -1 else x['pnl_daily_nominal_EQT'],
                                                     axis=1)
            rep["carry_eqt"] = rep.apply(lambda x: 0 if x["signal"] == -1 else x['carry_eqt'], axis=1)
            rep["borrow_cost"] = rep.apply(lambda x: -abs(x["qtty"] * x["px_entry"] * daily_cost), axis=1)

        if di:
            print("dididididid")
            # rep["pnl_daily_nominal_EQT"] = rep["pnl_daily_nominal_EQT"]/(-1*module_signal*rep["qtty"])

            # rep["pnl_daily_nominal_EQT"] = rep["pnl_daily_nominal_EQT"]/(1*module_signal*rep["qtty"])
            rep["txBrokerEqt"] = rep.apply(help_txBroker_di, axis=1)
            rep["txBroker"] = rep["txBrokerEqt"]
            rep["pnl_cdi_eqt"] = rep["pnl_daily_nominal_EQT"] + rep["carry_eqt"] + rep["txBrokerEqt"]
            # rep["pnl_cdi_eqt"] = rep["pnl_daily_nominal_EQT"] + rep["txBrokerEqt"]


        else:
            print("oi")
            rep["pnl_cdi_eqt"] = rep["pnl_daily_nominal_EQT"] + rep["carry_eqt"] + rep["txBrokerEqt"] + rep[
                "borrow_cost"]

        rep["pnl_cdi_hedge"] = rep["pnl_daily_nominal_Hedge"] + rep["carry_hedge"] + rep["txBrokerHedge"] + rep[
            "borrow_cost_hedge"]

        if hedge:
            rep["pnl_total"] = rep["pnl_cdi_hedge"] + rep["pnl_cdi_eqt"]


        # sem hedge pnl_total é carry + pos + custos
        else:
            rep["pnl_total"] = rep["pnl_cdi_eqt"]

        rep["type_boleta"] = 'normal'

        # hold
        rep_hold = rep[pd.isnull(rep["signal"])]

        # apenas compra
        rep_compra = rep[rep["signal"] == module_signal]

        # venda
        rep_venda = rep[rep["signal"] == -1 * module_signal]

        # new --> sera a boleta de zeragem de sinal igual
        new = rep_venda.copy()

        new2 = rep_venda.copy()

        ##############
        new2["pnl_daily_nominal_EQT"] = 0
        new2["type_boleta"] = 'zeragem'
        new2["pnl_cdi_eqt"] = new2["txBrokerEqt"]
        new2["pnl_cdi_hedge"] = new2["txBrokerHedge"]
        # new2["pnl_cdi_hedge"] = 0
        new2["borrow_cost"] = 0
        new2["pnl_daily_nominal_Hedge"] = 0
        new2["carry_eqt"] = 0
        new2["carry_hedge"] = 0
        new2["pnl_total"] = new2["txBroker"]

        ############## LAST HOLD DAY POSITION
        new["qtty"] = (-1) * new["qtty"]
        new["notional"] = (-1) * (new["notional"])
        new["signal"] = (-1) * (new["signal"])
        new["carry_eqt"] = (-1) * (new["carry_eqt"])
        new["carry_hedge"] = (-1) * (new["carry_hedge"])
        new["txBrokerHedge"] = 0
        new["txBrokerEqt"] = 0
        new["txBroker"] = 0

        if not di:
            new["pnl_daily_nominal_EQT"] = (-1) * (new["pnl_daily_nominal_EQT"])
            new["pnl_daily_nominal_Hedge"] = (-1) * (new["pnl_daily_nominal_Hedge"])

        else:
            new["pnl_daily_nominal_EQT"] = (new["pnl_daily_nominal_EQT"])
            new["pnl_daily_nominal_Hedge"] = (new["pnl_daily_nominal_Hedge"])

        if not di:

            new["pnl_cdi_eqt"] = new["pnl_daily_nominal_EQT"] + new["carry_eqt"] + new["borrow_cost"]


        else:

            new["pnl_cdi_eqt"] = new["pnl_daily_nominal_EQT"] + new["carry_eqt"] + new["borrow_cost"]
            # new["pnl_cdi_eqt"]= new["pnl_daily_nominal_EQT"]

        if hedge:

            new["pnl_cdi_hedge"] = new["pnl_daily_nominal_Hedge"] + new["carry_hedge"] + new["borrow_cost_hedge"]
            new["qtty_hedge"] = (-1) * new["qtty_hedge"]

        else:

            new["pnl_cdi_hedge"] = 0

        # nao cobra aluguel nesse ?
        new["pnl_total"] = new["pnl_cdi_hedge"] + new["pnl_cdi_eqt"]
        new["type_boleta"] = 'normal'

        rep = pd.concat([rep_hold, rep_compra, new2, new]).sort_values("date")

        daily_rep = rep.groupby(["date"]).agg(
            {"pnl_daily_nominal_EQT": "sum", "pnl_daily_nominal_Hedge": "sum", "txBroker": "sum", "borrow_cost": "sum",
             "carry_eqt": "sum", "carry_hedge": "sum", "notional": "sum", "pnl_total": "sum"}).reset_index()
        daily_rep["pl_d-1"] = daily_rep["notional"].shift(1)
        daily_rep["daily_return"] = 100 * ((daily_rep["pnl_total"]) / abs(daily_rep["pl_d-1"]))
        daily_rep["year"] = daily_rep["date"].apply(lambda x: x.year)

        anual_rep = daily_rep.groupby("year").agg({"pnl_daily_nominal_EQT": "sum",
                                                   "pnl_daily_nominal_Hedge": "sum",
                                                   "txBroker": "sum",
                                                   "borrow_cost": "sum",
                                                   "borrow_cost_hedge": "sum",
                                                   "carry_eqt": "sum",
                                                   "carry_hedge": "sum",
                                                   "pnl_total": "sum"}).reset_index()

        rep["year"] = rep["date"].apply(lambda x: x.year)
        trades = rep[(rep["signal"] == (-module_signal * 1)) & (rep["type_boleta"] == 'zeragem')]
        trades2 = trades.groupby("year").agg({"ticker": "count"}).reset_index()
        trades2.rename(columns={'ticker': 'n_trades'}, inplace=True)

        anual_rep = pd.merge(anual_rep, trades2, left_on="year", right_on="year", how="left")
        anual_rep["pnl/trade"] = anual_rep["pnl_total"] / anual_rep['n_trades']
        anual_rep["pnl/trade(%)"] = 100 * (anual_rep["pnl/trade"] / 100000)

        anual_rep["pnl_acc"] = anual_rep["pnl_total"].cumsum()
        daily_rep["pnl_acc"] = daily_rep["pnl_total"].cumsum()
        daily_rep["daily_return_temp"] = (daily_rep["daily_return"] / 100) + 1
        daily_rep["acc_rateReturn"] = daily_rep["daily_return_temp"].cumprod()
        daily_rep["acc_rateReturn"] = 100 * (daily_rep["acc_rateReturn"] - 1)
        daily_rep.drop("daily_return_temp", axis=1, inplace=True)

        if di:
            # anual_rep = anual_rep.drop(["pnl_daily_nominal_Hedge","borrow_cost","carry_eqt","carry_hedge","pnl_acc"],axis=1)
            anual_rep = anual_rep.drop(["pnl_daily_nominal_Hedge", "borrow_cost", "carry_hedge", "pnl_acc"], axis=1)

        return rep, daily_rep, anual_rep

    def generate_signal_type_3(self, datas, rank_in, rank_out, lista_lista_buy_cond, lista_lista_sell_cond,
                               lista_tickers, tam_rank_in, type_trade='long', rebalance_frequence=20):
        import time
        max_date = max(datas)
        lista_df = []

        var_control = []
        var_control.extend([1] * len(lista_lista_buy_cond[0]))

        # var_control[65] = 0
        # var_control[50] = 0

        # if long trade
        if type_trade == 'long':
            module_signal = 1
        else:
            module_signal = -1

        tam_rank_in = tam_rank_in

        # lista de listas para posicoes (hold + buy) E lista de lista de sinais (hold + buy)
        lista_lista_pos = []
        lista_lista_signals = []

        # lista de listas para posicoes (sell) E lista de lista de sinais (sell)
        lista_lista_sells = []
        lista_lista_signals_sell = []
        lista_datas = []
        flag_init = True
        count_days = 0

        # itera nas datas: gera a posicao da estrategia a cada data
        for el in range(len(lista_lista_buy_cond[0])):

            lista_signal_daily = []
            lista_signal_sell_daily = []

            # posicoes diarias
            pos_d1 = []
            pos_d1_sell = []

            if len(lista_lista_pos) == 0:
                pos = []


            else:
                pos = lista_lista_pos[-1]

            # 1) Só vai entrar aqui no momento em 0 do backtest. Testa apenas condicao de ligagem: var_control[el]
            # Como não tem posicao, nao precisa tratar vendas, apenas compras. Por isso essa condicao especial

            if (len(pos) == 0) & (var_control[el]):

                # se não é o primeiro trade(provavelmente zerou apos atingir v.c OU nao tinha posicao mesmo).
                # Nesse caso, compra todas a posicoes disponíveis. Nao verifica vendas pois nao há posicoes.
                # **** variavel de controle nao reseta a contagem, incrementa a variavel cada vez q vem aqui
                if not flag_init:
                    count_days += 1
                    # Só rebalanceia se atigiub contagem
                    if ((count_days % rebalance_frequence) == 0):
                        for el_ in range(len(rank_in[el])):
                            # print("nao comecou el: {}".format(el))
                            buy = lista_lista_buy_cond[lista_tickers.index(rank_in[el][el_])][el]
                            sell = lista_lista_sell_cond[lista_tickers.index(rank_in[el][el_])][el]
                            if (buy & (not sell)):
                                # print("deu comprar !!")
                                pos_d1.append(rank_in[el][el_])
                                lista_signal_daily.append(1 * module_signal)

                            else:
                                # nao comproou. Fica zerado ate  o proximo rebalance
                                pass

                        # reset e incremente para nao entrar de novo
                        count_days = 0

                    else:
                        # nao atingiu a contagem !
                        pass


                # primeiro trade ! ignora a contagem
                else:
                    flag_init = False
                    for el_ in range(len(rank_in[el])):
                        # print("nao comecou el: {}".format(el))
                        buy = lista_lista_buy_cond[lista_tickers.index(rank_in[el][el_])][el]
                        sell = lista_lista_sell_cond[lista_tickers.index(rank_in[el][el_])][el]
                        if (buy & (not sell)):
                            pos_d1.append(rank_in[el][el_])
                            lista_signal_daily.append(1 * module_signal)

                        else:
                            pass

                    # mesmo que nao tenha comprado nenhuma papel, considera-se que foi iniciado o bacotest. Proxima iteracao cai no loop acima
                    count_days += 1

                # só adiciona possicoa caso haja algum papel
                if pos_d1 != []:
                    lista_datas.append(datas[el])
                    lista_lista_signals.append(lista_signal_daily)
                    lista_lista_pos.append(pos_d1)

                    # append NoNe lists to sincronization purpose !
                    lista_lista_signals_sell.append([])
                    lista_lista_sells.append([])


            # 2) Esta desligada. Nao conta a variavel de frequencia
            elif (len(pos) == 0) & (not var_control[el]):

                pass


            # 3) ligada e verifica os rankings. Casos 1) len(pos) !=0 & var[i] ==1 e 2)  len(pos) !=0 & var[i] == 0 (zerada de tudo)
            else:

                count_days += 1
                qtty_sells = 0
                ticker_sell = []
                # Rebalanceia Normalmente !
                if ((count_days % rebalance_frequence) == 0):

                    for el_pos in range(len(pos)):

                        sell = lista_lista_sell_cond[lista_tickers.index(pos[el_pos])][el]  # the ticker's sell status ?

                        in_rank_out = pos[el_pos] in rank_out[el]  # is the ticker in rank_out ?

                        # 3.1: Verifica  se houve vendas.
                        if (sell | (not in_rank_out) | (not var_control[el])):
                            # print(" vendeu na iteracao: {} o ticer {}".format(el,pos[el_pos]))
                            pos_d1_sell.append(pos[el_pos])
                            lista_signal_sell_daily.append(-1 * module_signal)
                            qtty_sells += 1

                        # 3.2:Hold posicao do ativo
                        else:
                            pos_d1.append(pos[el_pos])
                            lista_signal_daily.append(None)

                    # se permanece ligada
                    if var_control[el]:
                        # obtaining tickers to buy: we buy  tam_rank_in (desired position) - len(curren position) - len(sold tickers)
                        # the elegible tickers are the ones that are not on the position and cond_buy = True and cond_sell = False
                        tickers_buy = [rank_in[el][el_buy] for el_buy in range(len(rank_in[el])) if (
                                    (rank_in[el][el_buy] not in pos) & (
                            lista_lista_buy_cond[lista_tickers.index(rank_in[el][el_buy])][el]) & (
                                        not lista_lista_sell_cond[lista_tickers.index(rank_in[el][el_buy])][el]))]
                        m = tam_rank_in - len(pos)

                        n_buys = len(tickers_buy[0:(qtty_sells + m)])

                        # print(datas[el])
                        pos_d1.extend(tickers_buy[0:(qtty_sells + m)])

                        sigs_buy = [module_signal * 1] * len(tickers_buy[0:n_buys])
                        lista_signal_daily.extend(sigs_buy)


                    # se deu zerada da variavel de controle, nem verifica se comprar alguma acoo, apenas realiza as vendas
                    # Os sinais de zerada ja foram executados na etapa anterior.
                    # pos se tornara []
                    else:
                        pass

                    count_days = 0

                # Nao rebalanceia ! Mantêm as posicoes do dia anterior.
                else:
                    # verifico apenas variavel de controle em dias foras da data de rebalance.
                    for el_pos in range(len(pos)):
                        # print("veio onde não a rebalnce")
                        # 3.1: Verifica  se houve vendas.
                        if ((not var_control[el])):
                            # print(" vendeu na iteracao: {} o ticer {}".format(el,pos[el_pos]))
                            pos_d1_sell.append(pos[el_pos])
                            lista_signal_sell_daily.append(-1 * module_signal)
                            qtty_sells += 1

                        # 3.2:Hold posicao do ativo
                        else:

                            pos_d1.append(pos[el_pos])
                            lista_signal_daily.append(None)

                # apenda os sinais do dia a lista_lista_sinais
                lista_lista_signals.append(lista_signal_daily)
                lista_lista_pos.append(pos_d1)

                lista_lista_signals_sell.append(lista_signal_sell_daily)
                lista_lista_sells.append(pos_d1_sell)
                lista_datas.append(datas[el])
                # print(pos_d1_sell)

        # joining stuff to generate a whole dataframe !
        a = [lista_lista_pos[el].extend(lista_lista_sells[el]) for el in range(len(lista_lista_pos))]
        a = [lista_lista_signals[el].extend(lista_lista_signals_sell[el]) for el in range(len(lista_lista_signals))]
        lista_of_listaDatas = [len(lista_lista_signals[el]) * [lista_datas[el]] for el in
                               range(len(lista_lista_signals))]

        datas_full = sum(lista_of_listaDatas, [])
        signals_full = sum(lista_lista_signals, [])
        lista_lista_pos = sum(lista_lista_pos, [])

        df_final = pd.DataFrame({"date": datas_full, "ticker": lista_lista_pos, "signals": signals_full})

        # vende na zerada da estratégia(data maxima). Zera caso haja posicoa. Pode nao haver caso a estratégia tivesse
        # sido desligada antes por conta de outros motivos
        _temp = df_final[df_final["date"] != max_date]
        _temp2 = df_final[df_final["date"] == max_date]
        if not _temp2.empty:
            _temp2["signals"] = -1 * module_signal
            df_final = pd.concat([_temp, _temp2])

        # agrup por ticker para se calcular o report
        df_final.columns = ["date", "ticker", "signal"]
        lista_df = []
        for ticker, df in df_final.groupby("ticker"):
            lista_df.append(df)

        return lista_df, df_final

    # generate_signal_type_4_daily é um q zera todo dia !!!!!
    def generate_signal_type_4(self, datas, rank_in, rank_out, lista_lista_buy_cond, lista_lista_sell_cond,
                               lista_tickers, tam_rank_in, type_trade='long', rebalance_frequence=1, var_control=None):
        import time
        max_date = max(datas)
        lista_df = []

        if var_control is None:
            var_control = []
            var_control.extend([1] * len(lista_lista_buy_cond[0]))

        # var_control[65] = 0
        # var_control[50] = 0

        # if long trade
        if type_trade == 'long':
            module_signal = 1
        else:
            module_signal = -1

        tam_rank_in = tam_rank_in

        # lista de listas para posicoes (hold + buy) E lista de lista de sinais (hold + buy)
        lista_lista_pos = []
        lista_lista_signals = []

        # lista de listas para posicoes (sell) E lista de lista de sinais (sell)
        lista_lista_sells = []
        lista_lista_signals_sell = []
        lista_datas = []
        flag_init = True
        count_days = 0

        # itera nas datas: gera a posicao da estrategia a cada data
        for el in range(len(lista_lista_buy_cond[0])):

            lista_signal_daily = []
            lista_signal_sell_daily = []

            # posicoes diarias
            pos_d1 = []
            pos_d1_sell = []

            if len(lista_lista_pos) == 0:
                pos = []


            else:
                pos = lista_lista_pos[-1]

            # 1) Só vai entrar aqui no momento em 0 do backtest. Testa apenas condicao de ligagem: var_control[el]
            # Como não tem posicao, nao precisa tratar vendas, apenas compras. Por isso essa condicao especial

            if (len(pos) == 0) & (var_control[el]):

                # se não é o primeiro trade(provavelmente zerou apos atingir v.c OU nao tinha posicao mesmo).
                # Nesse caso, compra todas a posicoes disponíveis. Nao verifica vendas pois nao há posicoes.
                # **** variavel de controle nao reseta a contagem, incrementa a variavel cada vez q vem aqui
                if not flag_init:
                    count_days += 1
                    # Só rebalanceia se atigiub contagem
                    if ((count_days % rebalance_frequence) == 0):
                        for el_ in range(len(rank_in[el])):
                            # print("nao comecou el: {}".format(el))
                            buy = lista_lista_buy_cond[lista_tickers.index(rank_in[el][el_])][el]
                            sell = lista_lista_sell_cond[lista_tickers.index(rank_in[el][el_])][el]
                            if (buy & (not sell)):
                                # print("deu comprar !!")
                                pos_d1.append(rank_in[el][el_])
                                lista_signal_daily.append(1 * module_signal)

                            else:
                                # nao comproou. Fica zerado ate  o proximo rebalance
                                pass

                        # reset e incremente para nao entrar de novo
                        count_days = 0

                    else:
                        # nao atingiu a contagem !
                        pass

                # primeiro trade ! ignora a contagem
                else:
                    flag_init = False
                    for el_ in range(len(rank_in[el])):
                        # print("nao comecou el: {}".format(el))
                        buy = lista_lista_buy_cond[lista_tickers.index(rank_in[el][el_])][el]
                        sell = lista_lista_sell_cond[lista_tickers.index(rank_in[el][el_])][el]
                        if (buy & (not sell)):
                            pos_d1.append(rank_in[el][el_])
                            lista_signal_daily.append(1 * module_signal)

                        else:
                            pass

                    # mesmo que nao tenha comprado nenhuma papel, considera-se que foi iniciado o bacotest. Proxima iteracao cai no loop acima
                    count_days += 1

                # só adiciona possicoa caso haja algum papel
                if pos_d1 != []:
                    lista_datas.append(datas[el])
                    lista_lista_signals.append(lista_signal_daily)
                    lista_lista_pos.append(pos_d1)

                    # append NoNe lists to sincronization purpose !
                    lista_lista_signals_sell.append([])
                    lista_lista_sells.append([])


            # 2) Esta desligada. Nao conta a variavel de frequencia
            elif (len(pos) == 0) & (not var_control[el]):

                pass

            # 3) ligada e verifica os rankings. Casos 1) len(pos) !=0 & var[i] ==1 e 2)  len(pos) !=0 & var[i] == 0 (zerada de tudo)
            else:

                count_days += 1
                qtty_sells = 0
                ticker_sell = []
                # Rebalanceia Normalmente !
                if ((count_days % rebalance_frequence) == 0):

                    for el_pos in range(len(pos)):

                        sell = lista_lista_sell_cond[lista_tickers.index(pos[el_pos])][el]  # the ticker's sell status ?

                        in_rank_out = pos[el_pos] in rank_out[el]  # is the ticker in rank_out ?

                        # 3.1: Verifica  se houve vendas.
                        if (sell | (not in_rank_out) | (not var_control[el])):
                            # print(" vendeu na iteracao: {} o ticer {}".format(el,pos[el_pos]))
                            pos_d1_sell.append(pos[el_pos])
                            lista_signal_sell_daily.append(-1 * module_signal)
                            qtty_sells += 1

                        # 3.2:Hold posicao do ativo
                        else:
                            pos_d1.append(pos[el_pos])
                            lista_signal_daily.append(None)

                    # se permanece ligada
                    if var_control[el]:
                        # obtaining tickers to buy: we buy  tam_rank_in (desired position) - len(curren position) - len(sold tickers)
                        # the elegible tickers are the ones that are not on the position and cond_buy = True and cond_sell = False
                        tickers_buy = [rank_in[el][el_buy] for el_buy in range(len(rank_in[el])) if (
                                    (rank_in[el][el_buy] not in pos) & (
                            lista_lista_buy_cond[lista_tickers.index(rank_in[el][el_buy])][el]) & (
                                        not lista_lista_sell_cond[lista_tickers.index(rank_in[el][el_buy])][el]))]
                        m = tam_rank_in - len(pos)

                        n_buys = len(tickers_buy[0:(qtty_sells + m)])

                        # print(datas[el])
                        pos_d1.extend(tickers_buy[0:(qtty_sells + m)])

                        sigs_buy = [module_signal * 1] * len(tickers_buy[0:n_buys])
                        lista_signal_daily.extend(sigs_buy)


                    # se deu zerada da variavel de controle, nem verifica se comprar alguma acoo, apenas realiza as vendas
                    # Os sinais de zerada ja foram executados na etapa anterior.
                    # pos se tornara []
                    else:
                        pass

                    count_days = 0

                # Nao rebalanceia ! Mantêm as posicoes do dia anterior.
                else:
                    # verifico apenas variavel de controle em dias foras da data de rebalance.
                    for el_pos in range(len(pos)):
                        # print("veio onde não a rebalnce")
                        # 3.1: Verifica  se houve vendas.
                        if ((not var_control[el])):
                            # print(" vendeu na iteracao: {} o ticer {}".format(el,pos[el_pos]))
                            pos_d1_sell.append(pos[el_pos])
                            lista_signal_sell_daily.append(-1 * module_signal)
                            qtty_sells += 1

                        # 3.2:Hold posicao do ativo
                        else:

                            pos_d1.append(pos[el_pos])
                            lista_signal_daily.append(None)

                # apenda os sinais do dia a lista_lista_sinais
                lista_lista_signals.append(lista_signal_daily)
                lista_lista_pos.append(pos_d1)

                lista_lista_signals_sell.append(lista_signal_sell_daily)
                lista_lista_sells.append(pos_d1_sell)
                lista_datas.append(datas[el])
                # print(pos_d1_sell)

        # joining stuff to generate a whole dataframe !
        a = [lista_lista_pos[el].extend(lista_lista_sells[el]) for el in range(len(lista_lista_pos))]
        a = [lista_lista_signals[el].extend(lista_lista_signals_sell[el]) for el in range(len(lista_lista_signals))]
        lista_of_listaDatas = [len(lista_lista_signals[el]) * [lista_datas[el]] for el in
                               range(len(lista_lista_signals))]

        datas_full = sum(lista_of_listaDatas, [])
        signals_full = sum(lista_lista_signals, [])
        lista_lista_pos = sum(lista_lista_pos, [])

        df_final = pd.DataFrame({"date": datas_full, "ticker": lista_lista_pos, "signals": signals_full})

        # vende na zerada da estratégia(data maxima). Zera caso haja posicoa. Pode nao haver caso a estratégia tivesse
        # sido desligada antes por conta de outros motivos
        _temp = df_final[df_final["date"] != max_date]
        _temp2 = df_final[df_final["date"] == max_date]
        if not _temp2.empty:
            _temp2["signals"] = -1 * module_signal
            df_final = pd.concat([_temp, _temp2])

        # agrup por ticker para se calcular o report
        df_final.columns = ["date", "ticker", "signal"]
        lista_df = []
        for ticker, df in df_final.groupby("ticker"):
            lista_df.append(df)

        return lista_df, df_final

    def generate_signal_type_4_fixed(self, datas, rank_in, rank_out, lista_lista_buy_cond, lista_lista_sell_cond,
                                     lista_tickers, tam_rank_in, type_trade='long', rebalance_frequence=1,
                                     var_control=None):
        import time
        max_date = max(datas)
        lista_df = []

        if var_control is None:
            var_control = []
            var_control.extend([1] * len(lista_lista_buy_cond[0]))

        # var_control[65] = 0
        # var_control[50] = 0

        # if long trade
        if type_trade == 'long':
            module_signal = 1
        else:
            module_signal = -1

        tam_rank_in = tam_rank_in

        # lista de listas para posicoes (hold + buy) E lista de lista de sinais (hold + buy)
        lista_lista_pos = []
        lista_lista_signals = []

        # lista de listas para posicoes (sell) E lista de lista de sinais (sell)
        lista_lista_sells = []
        lista_lista_signals_sell = []
        lista_datas = []
        count_first = True
        flag_init = True
        count_days = 0

        # itera nas datas: gera a posicao da estrategia a cada data
        for el in range(len(lista_lista_buy_cond[0])):

            lista_signal_daily = []
            lista_signal_sell_daily = []

            # posicoes diarias
            pos_d1 = []
            pos_d1_sell = []

            if len(lista_lista_pos) == 0:
                pos = []


            else:
                pos = lista_lista_pos[-1]

            if (len(pos) == 0) & (var_control[el]):

                # se não é o primeiro trade(provavelmente zerou apos atingir v.c OU nao tinha posicao mesmo).
                # Nesse caso, compra todas a posicoes disponíveis. Nao verifica vendas pois nao há posicoes.
                # **** variavel de controle nao reseta a contagem, incrementa a variavel cada vez q vem aqui

                if not flag_init:
                    count_days += 1
                    # Só rebalanceia se atigiub contagem
                    if ((count_days % rebalance_frequence) == 0) & (not count_first):
                        for el_ in range(len(rank_in[el])):
                            # print("TENTOU rebalanceia na data: {}, nao sei se conseguiu".format(datas[el]))
                            try:
                                # rint("oaoaoaoaoaoaoaooaaoa")
                                buy = lista_lista_buy_cond[lista_tickers.index(rank_in[el][el_])][el]
                                sell = lista_lista_sell_cond[lista_tickers.index(rank_in[el][el_])][el]

                            except:
                                # rint("o len rank_in is: {}, rank_in[el] is: {}".format(len(rank_in),len(rank_in[el])))
                                buy = False
                                sell = True

                            if (buy & (not sell)):
                                # print("deu comprar !!")
                                pos_d1.append(rank_in[el][el_])
                                lista_signal_daily.append(1 * module_signal)

                            else:
                                # nao comproou. Fica zerado ate  o proximo rebalance
                                pass

                        # reset e incremente para nao entrar de novo
                        count_days = 0

                    else:
                        # nao atingiu a contagem !
                        pass

                # primeiro trade ! ignora a contagem
                else:
                    # print("primeiro trade ever na data: {}".format(datas[el]))
                    flag_init = False
                    for el_ in range(len(rank_in[el])):

                        # print("teste o ticker: {}".format(rank_in[el][el_]))

                        # print("nao comecou el: {}".format(el))
                        try:
                            buy = lista_lista_buy_cond[lista_tickers.index(rank_in[el][el_])][el]
                            sell = lista_lista_sell_cond[lista_tickers.index(rank_in[el][el_])][el]

                        except:
                            buy = False
                            sell = True

                        if (buy & (not sell)):
                            pos_d1.append(rank_in[el][el_])
                            lista_signal_daily.append(1 * module_signal)

                        else:
                            pass

                    # mesmo que nao tenha comprado nenhuma papel, considera-se que foi iniciado o bacotest. Proxima iteracao cai no loop acima
                    # count_days +=1
                    count_first = False

                # só adiciona possicoa caso haja algum papel
                if pos_d1 != []:
                    lista_datas.append(datas[el])
                    lista_lista_signals.append(lista_signal_daily)
                    lista_lista_pos.append(pos_d1)

                    # append NoNe lists to sincronization purpose !
                    lista_lista_signals_sell.append([])
                    lista_lista_sells.append([])

            # 2) Esta desligada. Nao conta a variavel de frequencia
            elif (len(pos) == 0) & (not var_control[el]):

                pass


            # 3) ligada e verifica os rankings. Casos 1) len(pos) !=0 & var[i] ==1 e 2)  len(pos) !=0 & var[i] == 0 (zerada de tudo)
            else:

                count_days += 1
                qtty_sells = 0
                ticker_sell = []

                # Rebalanceia Normalmente !
                if ((count_days % rebalance_frequence) == 0):
                    # print("rebalanceia na data: {}".format(datas[el]))
                    for el_pos in range(len(pos)):

                        sell = lista_lista_sell_cond[lista_tickers.index(pos[el_pos])][el]  # the ticker's sell status ?

                        in_rank_out = pos[el_pos] in rank_out[el]  # is the ticker in rank_out ?

                        # 3.1: Verifica  se houve vendas.
                        if (sell | (not in_rank_out) | (not var_control[el])):
                            # print(" vendeu na iteracao: {} o ticer {}".format(el,pos[el_pos]))
                            pos_d1_sell.append(pos[el_pos])
                            lista_signal_sell_daily.append(-1 * module_signal)
                            qtty_sells += 1

                        # 3.2:Hold posicao do ativo
                        else:
                            pos_d1.append(pos[el_pos])
                            lista_signal_daily.append(None)

                    # se permanece ligada
                    if var_control[el]:
                        # obtaining tickers to buy: we buy  tam_rank_in (desired position) - len(curren position) - len(sold tickers)
                        # the elegible tickers are the ones that are not on the position and cond_buy = True and cond_sell = False
                        tickers_buy = [rank_in[el][el_buy] for el_buy in range(len(rank_in[el])) if (
                                    (rank_in[el][el_buy] not in pos) & (
                            lista_lista_buy_cond[lista_tickers.index(rank_in[el][el_buy])][el]) & (
                                        not lista_lista_sell_cond[lista_tickers.index(rank_in[el][el_buy])][el]))]
                        m = tam_rank_in - len(pos)

                        n_buys = len(tickers_buy[0:(qtty_sells + m)])

                        # print(datas[el])
                        pos_d1.extend(tickers_buy[0:(qtty_sells + m)])

                        sigs_buy = [module_signal * 1] * len(tickers_buy[0:n_buys])
                        lista_signal_daily.extend(sigs_buy)


                    # se deu zerada da variavel de controle, nem verifica se comprar alguma acoo, apenas realiza as vendas
                    # Os sinais de zerada ja foram executados na etapa anterior.
                    # pos se tornara []
                    else:
                        pass

                    count_days = 0

                # Nao rebalanceia ! Mantêm as posicoes do dia anterior.
                else:
                    # verifico apenas variavel de controle em dias foras da data de rebalance.
                    for el_pos in range(len(pos)):
                        # print("veio onde não a rebalnce")
                        # 3.1: Verifica  se houve vendas.
                        if ((not var_control[el])):
                            # print(" vendeu na iteracao: {} o ticer {}".format(el,pos[el_pos]))
                            pos_d1_sell.append(pos[el_pos])
                            lista_signal_sell_daily.append(-1 * module_signal)
                            qtty_sells += 1

                        # 3.2:Hold posicao do ativo
                        else:

                            pos_d1.append(pos[el_pos])
                            lista_signal_daily.append(None)

                # apenda os sinais do dia a lista_lista_sinais
                lista_lista_signals.append(lista_signal_daily)
                lista_lista_pos.append(pos_d1)

                lista_lista_signals_sell.append(lista_signal_sell_daily)
                lista_lista_sells.append(pos_d1_sell)
                lista_datas.append(datas[el])
                # print(pos_d1_sell)

        # joining stuff to generate a whole dataframe !
        a = [lista_lista_pos[el].extend(lista_lista_sells[el]) for el in range(len(lista_lista_pos))]
        a = [lista_lista_signals[el].extend(lista_lista_signals_sell[el]) for el in range(len(lista_lista_signals))]
        lista_of_listaDatas = [len(lista_lista_signals[el]) * [lista_datas[el]] for el in
                               range(len(lista_lista_signals))]

        datas_full = sum(lista_of_listaDatas, [])
        signals_full = sum(lista_lista_signals, [])
        lista_lista_pos = sum(lista_lista_pos, [])

        df_final = pd.DataFrame({"date": datas_full, "ticker": lista_lista_pos, "signals": signals_full})

        # vende na zerada da estratégia(data maxima). Zera caso haja posicoa. Pode nao haver caso a estratégia tivesse
        # sido desligada antes por conta de outros motivos
        _temp = df_final[df_final["date"] != max_date]
        _temp2 = df_final[df_final["date"] == max_date]
        if not _temp2.empty:
            _temp2["signals"] = -1 * module_signal
            df_final = pd.concat([_temp, _temp2])

        # agrup por ticker para se calcular o report
        df_final.columns = ["date", "ticker", "signal"]
        lista_df = []
        for ticker, df in df_final.groupby("ticker"):
            lista_df.append(df)

        return lista_df, df_final

    # type 1, with control variable and rebalance(dias corridos).
    # No external control variables
    def generate_signal_type_1_2_fixed(self, type_trade, rank_in, rank_out, lista_lista_buydn_cond,
                                       lista_lista_selldn_cond, lista_lista_buyd0_cond, lista_lista_selld0_cond,
                                       lista_tickers, df_params_full, datas, rebalance_frequence=20):

        # rebalance_frequence +=1

        lista_df = []

        if type_trade == 'long':

            module_signal = 1


        else:
            module_signal = -1

        count_days = 0
        data_max = max(datas)
        # ebalance_frequence=20
        # itera nos tickers
        # if ((count_days%rebalance_frequence)==0):
        for el in range(len(lista_tickers)):

            flag_init = True
            lista_signal = []
            pos = False
            ticker = lista_tickers[el]

            count_days = 0

            # itera nas datas
            for el2 in range(len(lista_lista_buyd0_cond[el])):

                if not flag_init:

                    # nao tem posicoa, verifica se compra indepenen do sinal de rebalance
                    if not pos:

                        # nao tem posicao e esta no rank_in: compra. Nao pode ter sinal de compra na ultima data pois nao tera como zerar (e nem como identificar para zerar virtualmente depois)
                        if (ticker in rank_in[el2]) & (not pos) & lista_lista_buydn_cond[el][el2] & (
                        not (lista_lista_selldn_cond[el][el2])):
                            if datas[el2] != data_max:
                                pos = True
                                count_days += 1
                                lista_signal.append(1 * module_signal)

                            else:

                                lista_signal.append(2)


                        else:

                            lista_signal.append(2)

                    else:

                        # tem posicao, incrementa a contagem
                        count_days += 1

                        # if rebalnace
                        if (((count_days - 1) % rebalance_frequence) == 0):

                            # se tem posicoes, verifica-se a possivel venda
                            if ((ticker not in rank_out[el2]) | (lista_lista_selldn_cond[el][el2])) & (pos):
                                pos = False
                                lista_signal.append(-1 * module_signal)


                            # esta no rank_in e ja tem posicao: hold
                            elif (ticker in rank_in[el2]) & (pos):
                                lista_signal.append(None)


                            # ja tem posicao e esta no rank_out: hold
                            elif (ticker in rank_out[el2]) & (pos):

                                lista_signal.append(None)

                            else:

                                lista_signal.append(2)

                        # if not rebalance, hold
                        else:
                            if (pos):
                                lista_signal.append(None)

                            else:
                                lista_signal.append(2)

                else:

                    flag_init = False

                    # nao tem posicao e esta no rank_in: compra
                    if (ticker in rank_in[el2]) & (not pos) & lista_lista_buyd0_cond[el][el2] & (
                    not (lista_lista_selld0_cond[el][el2])):

                        count_days += 1
                        pos = True
                        lista_signal.append(1 * module_signal)

                    else:
                        lista_signal.append(2)

            df = pd.DataFrame({"signal": lista_signal, "date": datas})
            df["ticker"] = ticker
            lista_df.append(df)
            df_final = pd.concat(lista_df)

        return lista_df, df_final

    # type 1, with control variable and rebalance(dias corridos).
    # No external control variables
    def generate_signal_type_1_2(self, type_trade, rank_in, rank_out, lista_lista_buydn_cond, lista_lista_selldn_cond,
                                 lista_lista_buyd0_cond, lista_lista_selld0_cond, lista_tickers, df_params_full, datas,
                                 rebalance_frequence=20):

        lista_df = []
        if type_trade == 'long':
            module_signal = 1

        else:
            module_signal = -1

        count_days = 0
        data_max = max(datas)
        # ebalance_frequence=20
        # itera nos tickers
        # if ((count_days%rebalance_frequence)==0):
        for el in range(len(lista_tickers)):

            flag_init = True
            lista_signal = []
            pos = False
            ticker = lista_tickers[el]
            count_days = 0
            # itera nas datas
            for el2 in range(len(lista_lista_buyd0_cond[el])):
                if not flag_init:
                    count_days += 1
                    if ((count_days % rebalance_frequence) == 0):
                        # nao tem posicao e esta no rank_in: compra. Nao pode ter sinal de compra na ultima data pois nao tera como zerar (e nem como identificar para zerar virtualmente depois)
                        if (ticker in rank_in[el2]) & (not pos) & lista_lista_buydn_cond[el][el2] & (
                        not (lista_lista_selldn_cond[el][el2])):
                            if datas[el2] != data_max:
                                pos = True
                                lista_signal.append(1 * module_signal)

                            else:
                                lista_signal.append(2)

                        # tem posicao e nao estao no rank_out: venda
                        elif ((ticker not in rank_out[el2]) | (lista_lista_selldn_cond[el][el2])) & (pos):
                            pos = False
                            lista_signal.append(-1 * module_signal)


                        # esta no rank_in e ja tem posicao: hold
                        elif (ticker in rank_in[el2]) & (pos):
                            lista_signal.append(None)


                        # ja tem posicao e esta no rank_out: hold
                        elif (ticker in rank_out[el2]) & (pos):

                            lista_signal.append(None)

                        else:

                            lista_signal.append(2)

                    else:
                        if (pos):
                            lista_signal.append(None)

                        else:
                            lista_signal.append(2)

                else:
                    count_days += 1
                    flag_init = False
                    # nao tem posicao e esta no rank_in: compra
                    if (ticker in rank_in[el2]) & (not pos) & lista_lista_buyd0_cond[el][el2] & (
                    not (lista_lista_selld0_cond[el][el2])):
                        pos = True
                        lista_signal.append(1 * module_signal)

                    else:
                        lista_signal.append(2)

            df = pd.DataFrame({"signal": lista_signal, "date": datas})
            df["ticker"] = ticker
            lista_df.append(df)
            df_final = pd.concat(lista_df)

        return lista_df, df_final

    # type1  with  control variables and not rebalance
    # No external control variables
    def generate_signal_type_1_1(self, rank_in, rank_out, lista_lista_buydn_cond, lista_lista_selldn_cond,
                                 lista_lista_buyd0_cond, lista_lista_selld0_cond, lista_tickers, df_params_full, datas):
        lista_df = []
        type_trade = 'long'

        if type_trade == 'long':
            module_signal = 1

        else:
            module_signal = -1

        # itera nos tickers
        for el in range(len(lista_tickers)):
            flag_init = True
            lista_signal = []
            pos = False
            ticker = lista_tickers[el]
            count_days = 0
            # itera nas datas
            for el2 in range(len(lista_lista_buyd0_cond[el])):
                if not flag_init:
                    # nao tem posicao e esta no rank_in: compra
                    if (ticker in rank_in[el2]) & (not pos) & lista_lista_buydn_cond[el][el2] & (
                    not (lista_lista_selldn_cond[el][el2])):
                        pos = True
                        lista_signal.append(1 * module_signal)


                    # tem posicao e nao estao no rank_out: venda
                    elif ((ticker not in rank_out[el2]) | (lista_lista_selldn_cond[el][el2])) & (pos):
                        pos = False
                        lista_signal.append(-1 * module_signal)


                    # esta no rank_in e ja tem posicao: hold
                    elif (ticker in rank_in[el2]) & (pos):
                        lista_signal.append(None)


                    # ja tem posicao e esta no rank_out: hold
                    elif (ticker in rank_out[el2]) & (pos):

                        lista_signal.append(None)

                    else:

                        lista_signal.append(2)

                else:
                    flag_init = False
                    # nao tem posicao e esta no rank_in: compra
                    if (ticker in rank_in[el2]) & (not pos) & lista_lista_buyd0_cond[el][el2] & (
                    not (lista_lista_selld0_cond[el][el2])):
                        pos = True
                        lista_signal.append(1 * module_signal)

                    else:
                        lista_signal.append(2)

            df = pd.DataFrame({"signal": lista_signal, "date": datas})
            df["ticker"] = ticker
            lista_df.append(df)
            df_final = pd.concat(lista_df)

        return lista_df, df_final

    def generate_signal_type_1(self, datas, rank_in, rank_out, lista_lista_buy_cond, lista_lista_sell_cond,
                               lista_tickers, tam_rank_in, type_trade='long'):

        lista_df = []

        if type_trade == 'long':
            module_signal = 1

        else:
            module_signal = -1

        # itera nos tickers
        for el in range(len(lista_tickers)):

            lista_signal = []
            pos = False
            ticker = lista_tickers[el]

            # itera nas datas
            for el2 in range(len(lista_lista_buy_cond[el])):

                # nao tem posicao e esta no rank_in: compra
                if (ticker in rank_in[el2]) & (not pos) & lista_lista_buy_cond[el][el2] & (
                not (lista_lista_sell_cond[el][el2])):
                    pos = True
                    lista_signal.append(1 * module_signal)


                # tem posicao e nao estao no rank_out: venda
                elif (ticker not in rank_out[el2]) & (pos):
                    pos = False
                    lista_signal.append(-1 * module_signal)

                # esta no rank_in e ja tem posicao: hold
                elif (ticker in rank_in[el2]) & (pos):
                    lista_signal.append(None)


                # ja tem posicao e esta no rank_out: hold
                elif (ticker in rank_out[el2]) & (pos):

                    lista_signal.append(None)


                else:

                    lista_signal.append(2)

            df = pd.DataFrame({"signal": lista_signal, "date": datas})
            df["ticker"] = ticker
            lista_df.append(df)
            df_final = pd.concat(lista_df)

        return lista_df, df_final

    def gen_report_intraday(self, df_final, df_new, type_trade='long', notional=100000):

        def gen_df_pnl(df_pnl_final):
            pnl_total = df_pnl_final["pnl_total"].sum()
            pnl_eqt = df_pnl_final["pnl_eqt"].sum()
            pnl_hedge = df_pnl_final["pnl_hedge"].sum()
            tx = df_pnl_final["tx"].sum()
            pnl_nominal = df_pnl_final["pnl_eqt"].sum() + df_pnl_final["pnl_hedge"].sum()
            return pd.DataFrame(
                {"pnl_total": [pnl_total], "pnl_eqt": [pnl_eqt], "pnl_hedge": [pnl_hedge], "pnl_nominal": pnl_nominal,
                 "tx_total": [tx], "type_trade": [type_trade]}).T

        df_signals = df_final[~pd.isnull(df_final["signal"])]
        df_prices = df_new[
            ["date", "date_short", "ticker", "close", "close_d-1", "close_index", "close_d-1_index", "px_last",
             "px_last_index"]]
        df_pnl = pd.merge(df_signals, df_prices, left_on=["date", "ticker"], right_on=["date", "ticker"], how="inner")

        if type_trade == 'long':
            module = 1


        else:
            module = -1

        lista_pnl = []
        tickers = df_pnl["ticker"].unique().tolist()
        for ticker in tickers:
            df1 = df_pnl[df_pnl["ticker"] == ticker].sort_values("date")
            df1["qtty"] = np.nan
            df1["qtty_hedge"] = np.nan

            df1_buy = df1[df1["signal"] == module * 1]
            df1_sell = df1[df1["signal"] == module * -1]

            df1_buy["qtty"] = df1_buy["close"].apply(lambda x: np.round((100000 / x)))
            df1_buy["qtty_hedge"] = df1_buy["close_index"].apply(lambda x: np.round((100000 / x)))

            df1 = pd.concat([df1_buy, df1_sell]).sort_values("date")
            df1["qtty"].fillna(method='ffill', inplace=True)
            df1["qtty_hedge"].fillna(method='ffill', inplace=True)
            df1_buy["qtty"] = df1_buy["close"].apply(lambda x: np.round((100000 / x)))
            # df1["qtty_hedge"] = df1["qtty_hedge"]*df1["signal"]*-1
            df1["qtty_hedge"] = df1["qtty_hedge"] * df1["signal"] * -1

            df1["pnl_eqt"] = df1["qtty"] * (df1["px_last"] - df1["close"])
            df1["pnl_hedge"] = df1["qtty_hedge"] * (df1["px_last_index"] - df1["close_index"])

            df1["tx"] = -1 * abs(df1["qtty"] * df1["close"]) * 0.0012

            df1["pnl_total"] = df1["pnl_hedge"] + df1["pnl_eqt"] + df1["tx"]

            lista_pnl.append(df1)

        df_pnl_final = pd.concat(lista_pnl)

        df_res = gen_df_pnl(df_pnl_final)

        return df_res, df_pnl_final

    #
    def compose_res(self, df_res_all,
                    dict_weight={"pnl_total": 1, "sharp": 1, "positives_years(%)": 1, "pnl_pos/pnl_neg(%)": 1},
                    lista_orders=[False, False, False, False]):

        def add_rankings(df, lista_cols, lista_orders):
            def ranker(df, col):
                df['rank_{}'.format(col)] = np.arange(len(df)) + 1
                return df

            # for col in lista_cols:
            #    df.sort('value', ascending = True, inplace=True)
            #    df = df.groupby(['group']).apply(ranker)
            el = 0
            for col in lista_cols:
                df.sort_values(col, ascending=lista_orders[el], inplace=True)
                df = ranker(df, col)
                el += 1

            return df

        pnl_pos = df_res_all[df_res_all["pnl_total"] > 0].groupby(["param_name"]).agg(
            {"year": "count", "pnl_total": "mean"}).reset_index()
        pnl_pos.columns = ["param_name", "count_pos_years", "mean_pos_pnl"]

        pnl_neg = df_res_all[df_res_all["pnl_total"] < 0].groupby(["param_name"]).agg(
            {"year": "count", "pnl_total": "mean"}).reset_index()
        pnl_neg.columns = ["param_name", "count_neg_years", "mean_neg_pnl"]

        shp = df_res_all.groupby(["param_name"]).agg({"pnl_total": ["mean", "std", "sum"]}).reset_index()
        shp.columns = ["param_name", "mean", "std", "pnl_total"]
        shp["sharp"] = shp["mean"] / shp["std"]
        shp = shp.drop(["std", "mean"], axis=1)

        pnls = pd.merge(pnl_pos, pnl_neg, left_on=["param_name"], right_on=["param_name"], how="outer")
        pnls["positives_years(%)"] = pnls["count_pos_years"] / (pnls["count_pos_years"] + pnls["count_neg_years"])
        pnls["pnl_pos/pnl_neg(%)"] = 100 * (pnls["mean_pos_pnl"] / abs(pnls["mean_neg_pnl"]))
        pnls.drop(["mean_pos_pnl", "mean_neg_pnl", "count_pos_years", "count_neg_years"], axis=1, inplace=True)

        # pnls["score"] = pnls["score"]

        pnls = pd.merge(pnls, shp, left_on=["param_name"], right_on=["param_name"], how="outer")
        pnls = add_rankings(pnls, ["pnl_total", "sharp", "positives_years(%)", "pnl_pos/pnl_neg(%)"], lista_orders)

        pnls["score"] = pnls.apply(lambda x: 100 - (((x["rank_pnl_total"] * dict_weight.get("pnl_total")) + (
                    x["rank_sharp"] * dict_weight.get("sharp")) + (x["rank_positives_years(%)"] * dict_weight.get(
            "positives_years(%)")) + (x["rank_pnl_pos/pnl_neg(%)"] * dict_weight.get("pnl_pos/pnl_neg(%)"))) / sum(
            dict_weight.values())), axis=1)

        return pnls

    def generate_signal_type_3_intraday(self, datas, rank_in, rank_out, lista_lista_buy_cond, lista_lista_sell_cond,
                                        lista_tickers, tam_rank_in, type_trade='long', rebalance_frequence=20,
                                        var_control=None):

        import time
        max_date = max(datas)
        lista_df = []

        if var_control is None:
            var_control = []
            var_control.extend([1] * len(lista_lista_buy_cond[0]))

        # if long trade
        if type_trade == 'long':

            module_signal = 1


        else:
            module_signal = -1

        tam_rank_in = tam_rank_in

        # lista de listas para posicoes (hold + buy) E lista de lista de sinais (hold + buy)
        lista_lista_pos = []
        lista_lista_signals = []

        # lista de listas para posicoes (sell) E lista de lista de sinais (sell)
        lista_lista_sells = []
        lista_lista_signals_sell = []
        lista_datas = []
        flag_init = True
        count_days = 0

        # itera nas datas: gera a posicao da estrategia a cada data
        for el in range(len(lista_lista_buy_cond[0])):

            lista_signal_daily = []
            lista_signal_sell_daily = []

            # posicoes diarias
            pos_d1 = []
            pos_d1_sell = []

            if len(lista_lista_pos) == 0:
                pos = []


            else:
                pos = lista_lista_pos[-1]

            # 1) Só vai entrar aqui no momento em 0 do backtest. Testa apenas condicao de ligagem: var_control[el]
            # Como não tem posicao, nao precisa tratar vendas, apenas compras. Por isso essa condicao especial

            if (len(pos) == 0) & (var_control[el]):

                # se não é o primeiro trade(provavelmente zerou apos atingir v.c OU nao tinha posicao mesmo).
                # Nesse caso, compra todas a posicoes disponíveis. Nao verifica vendas pois nao há posicoes.
                # **** variavel de controle nao reseta a contagem, incrementa a variavel cada vez q vem aqui
                if not flag_init:

                    count_days += 1
                    # Só rebalanceia se atigiub contagem -->
                    if ((count_days % rebalance_frequence) == 0):
                        for el_ in range(len(rank_in[el])):
                            buy = lista_lista_buy_cond[lista_tickers.index(rank_in[el][el_])][el]
                            sell = lista_lista_sell_cond[lista_tickers.index(rank_in[el][el_])][el]
                            if (buy & (not sell)):
                                pos_d1.append(rank_in[el][el_])
                                lista_signal_daily.append(1 * module_signal)

                            else:
                                # nao comproou. Fica zerado ate  o proximo rebalance
                                pass

                        # reset e incremente para nao entrar de novo
                        count_days = 0

                    else:
                        # nao atingiu a contagem !
                        pass

                # primeiro trade ! ignora a contagem
                else:
                    flag_init = False
                    for el_ in range(len(rank_in[el])):

                        # print("nao comecou el: {}".format(el))
                        buy = lista_lista_buy_cond[lista_tickers.index(rank_in[el][el_])][el]
                        sell = lista_lista_sell_cond[lista_tickers.index(rank_in[el][el_])][el]
                        if (buy & (not sell)):
                            pos_d1.append(rank_in[el][el_])
                            lista_signal_daily.append(1 * module_signal)

                        else:
                            pass

                    # mesmo que nao tenha comprado nenhuma papel, considera-se que foi iniciado o bacotest. Proxima iteracao cai no loop acima
                    count_days += 1

                # só adiciona possicoa caso haja algum papel
                if pos_d1 != []:
                    lista_datas.append(datas[el])
                    lista_lista_signals.append(lista_signal_daily)
                    lista_lista_pos.append(pos_d1)

                    # append NoNe lists to sincronization purpose !
                    lista_lista_signals_sell.append([])
                    lista_lista_sells.append([])


            # 2) Esta desligada. Nao conta a variavel de frequencia
            elif (len(pos) == 0) & (not var_control[el]):

                pass


            # 3) ligada e verifica os rankings. Casos 1) len(pos) !=0 & var[i] ==1 e 2)  len(pos) !=0 & var[i] == 0 (zerada de tudo)
            else:
                count_days += 1
                qtty_sells = 0
                ticker_sell = []
                # Rebalanceia Normalmente !
                if ((count_days % rebalance_frequence) == 0):

                    for el_pos in range(len(pos)):

                        # esta venda ?
                        sell = lista_lista_sell_cond[lista_tickers.index(pos[el_pos])][el]

                        # esta no rank out ?
                        in_rank_out = pos[el_pos] in rank_out[el]

                        # 3.1: Verifica  se houve vendas.
                        if (sell | (not in_rank_out) | (not var_control[el])):
                            # print(" vendeu na iteracao: {} o ticer {}".format(el,pos[el_pos]))
                            pos_d1_sell.append(pos[el_pos])
                            lista_signal_sell_daily.append(-1 * module_signal)
                            qtty_sells += 1

                        # 3.2:Hold posicao do ativo
                        else:
                            pos_d1.append(pos[el_pos])
                            lista_signal_daily.append(None)

                    # se permanece ligada
                    if var_control[el]:
                        # obtaining tickers to buy: we buy  tam_rank_in (desired position) - len(curren position) - len(sold tickers)
                        # the elegible tickers are the ones that are not on the position and cond_buy = True and cond_sell = False
                        tickers_buy = [rank_in[el][el_buy] for el_buy in range(len(rank_in[el])) if (
                                    (rank_in[el][el_buy] not in pos) & (
                            lista_lista_buy_cond[lista_tickers.index(rank_in[el][el_buy])][el]) & (
                                        not lista_lista_sell_cond[lista_tickers.index(rank_in[el][el_buy])][el]))]
                        m = tam_rank_in - len(pos)

                        n_buys = len(tickers_buy[0:(qtty_sells + m)])

                        # print(datas[el])
                        pos_d1.extend(tickers_buy[0:(qtty_sells + m)])

                        sigs_buy = [module_signal * 1] * len(tickers_buy[0:n_buys])
                        lista_signal_daily.extend(sigs_buy)


                    # se deu zerada da variavel de controle, nem verifica se comprar alguma acoo, apenas realiza as vendas
                    # Os sinais de zerada ja foram executados na etapa anterior.
                    # pos se tornara []
                    else:
                        pass

                    count_days = 0

                # Nao rebalanceia ! Mantêm as posicoes do dia anterior.
                else:
                    # verifico apenas variavel de controle em dias foras da data de rebalance.
                    for el_pos in range(len(pos)):
                        # print("veio onde não a rebalnce")
                        # 3.1: Verifica  se houve vendas.
                        if ((not var_control[el])):
                            # print(" vendeu na iteracao: {} o ticer {}".format(el,pos[el_pos]))
                            pos_d1_sell.append(pos[el_pos])
                            lista_signal_sell_daily.append(-1 * module_signal)
                            qtty_sells += 1

                        # 3.2:Hold posicao do ativo
                        else:

                            pos_d1.append(pos[el_pos])
                            lista_signal_daily.append(None)

                # apenda os sinais do dia a lista_lista_sinais
                lista_lista_signals.append(lista_signal_daily)
                lista_lista_pos.append(pos_d1)

                lista_lista_signals_sell.append(lista_signal_sell_daily)
                lista_lista_sells.append(pos_d1_sell)
                lista_datas.append(datas[el])
                # print(pos_d1_sell)

        # joining stuff to generate a whole dataframe !
        a = [lista_lista_pos[el].extend(lista_lista_sells[el]) for el in range(len(lista_lista_pos))]
        a = [lista_lista_signals[el].extend(lista_lista_signals_sell[el]) for el in range(len(lista_lista_signals))]
        lista_of_listaDatas = [len(lista_lista_signals[el]) * [lista_datas[el]] for el in
                               range(len(lista_lista_signals))]

        datas_full = sum(lista_of_listaDatas, [])
        signals_full = sum(lista_lista_signals, [])
        lista_lista_pos = sum(lista_lista_pos, [])

        df_final = pd.DataFrame({"date": datas_full, "ticker": lista_lista_pos, "signals": signals_full})

        # vende na zerada da estratégia(data maxima). Zera caso haja posicoa. Pode nao haver caso a estratégia tivesse
        # sido desligada antes por conta de outros motivos
        _temp = df_final[df_final["date"] != max_date]
        _temp2 = df_final[df_final["date"] == max_date]
        if not _temp2.empty:
            _temp2["signals"] = -1 * module_signal
            df_final = pd.concat([_temp, _temp2])

        # agrup por ticker para se calcular o report
        df_final.columns = ["date", "ticker", "signal"]
        lista_df = []
        for ticker, df in df_final.groupby("ticker"):
            lista_df.append(df)

        return lista_df, df_final

    def generate_signal_type_2(self, datas, rank_in, rank_out, lista_lista_buy_cond, lista_lista_sell_cond,
                               lista_tickers, tam_rank_in, type_trade='long'):

        import time
        max_date = max(datas)
        lista_df = []
        var_control = []
        var_control.extend([1] * len(lista_lista_buy_cond[0]))

        # var_control[65] = 0
        # var_control[50] = 0

        # if long trade
        if type_trade == 'long':
            module_signal = 1
        else:
            module_signal = -1

        tam_rank_in = tam_rank_in

        # lista de listas para posicoes (hold + buy) E lista de lista de sinais (hold + buy)
        lista_lista_pos = []
        lista_lista_signals = []

        # lista de listas para posicoes (sell) E lista de lista de sinais (sell)
        lista_lista_sells = []
        lista_lista_signals_sell = []
        lista_datas = []

        # itera nas datas: gera a posicao da estrategia a cada data
        for el in range(len(lista_lista_buy_cond[0])):
            # print(el)
            # sinais diarios
            lista_signal_daily = []
            lista_signal_sell_daily = []

            # posicoes diarias
            pos_d1 = []
            pos_d1_sell = []

            if len(lista_lista_pos) == 0:
                pos = []


            else:
                pos = lista_lista_pos[-1]

            # 1) estrategia estava desligada
            if (len(pos) == 0) & (var_control[el]):
                # print("noa comecou")
                for el_ in range(len(rank_in[el])):

                    # print("nao comecou el: {}".format(el))
                    buy = lista_lista_buy_cond[lista_tickers.index(rank_in[el][el_])][el]
                    sell = lista_lista_sell_cond[lista_tickers.index(rank_in[el][el_])][el]

                    print(buy)
                    if (buy & (not sell)):
                        pos_d1.append(rank_in[el][el_])
                        lista_signal_daily.append(1)

                    else:
                        pass

                lista_lista_signals.append(lista_signal_daily)
                lista_lista_pos.append(pos_d1)

                # append NoNe lists to sincronization purpose !
                lista_lista_signals_sell.append([])
                lista_lista_sells.append([])

                if pos_d1 != []:
                    lista_datas.append(datas[el])

            # 2) desligada e nao tem condicao de entrada
            elif (len(pos) == 0) & (not var_control[el]):

                pass


            # 3) ligada e verifica os rankings
            else:
                qtty_sells = 0
                ticker_sell = []
                for el_pos in range(len(pos)):

                    # getting conditions
                    sell = lista_lista_sell_cond[lista_tickers.index(pos[el_pos])][el]
                    in_rank_out = pos[el_pos] in rank_out[el]

                    # Houve venda do papel: SELL
                    if (sell | (not in_rank_out) | (not var_control[el])):

                        pos_d1_sell.append(pos[el_pos])
                        lista_signal_sell_daily.append(-1)
                        qtty_sells += 1

                    # nao houve venda do papel: HOLD
                    else:
                        pos_d1.append(pos[el_pos])
                        lista_signal_daily.append(None)

                # obtaining tickers to buy: we buy  tam_rank_in (desired position) - len(curren position) - len(sold tickers)
                # the elegible tickers are the ones that are not on the position and cond_buy = True and cond_sell = False
                tickers_buy = [rank_in[el][el_buy] for el_buy in range(len(rank_in[el])) if (
                            (rank_in[el][el_buy] not in pos) & (
                    lista_lista_buy_cond[lista_tickers.index(rank_in[el][el_buy])][el]) & (
                                not lista_lista_sell_cond[lista_tickers.index(rank_in[el][el_buy])][el]))]
                m = tam_rank_in - len(pos)

                n_buys = len(tickers_buy[0:(qtty_sells + m)])

                # print(datas[el])
                pos_d1.extend(tickers_buy[0:(qtty_sells + m)])

                sigs_buy = [1] * len(tickers_buy[0:n_buys])
                lista_signal_daily.extend(sigs_buy)

                # apenda os sinais do dia a lista_lista_sinais
                lista_lista_signals.append(lista_signal_daily)
                lista_lista_pos.append(pos_d1)

                lista_lista_signals_sell.append(lista_signal_sell_daily)
                lista_lista_sells.append(pos_d1_sell)
                lista_datas.append(datas[el])
                # print(pos_d1_sell)

        # joining stuff to generate a whole dataframe !
        a = [lista_lista_pos[el].extend(lista_lista_sells[el]) for el in range(len(lista_lista_pos))]
        a = [lista_lista_signals[el].extend(lista_lista_signals_sell[el]) for el in range(len(lista_lista_signals))]
        lista_of_listaDatas = [len(lista_lista_signals[el]) * [lista_datas[el]] for el in
                               range(len(lista_lista_signals))]

        datas_full = sum(lista_of_listaDatas, [])
        signals_full = sum(lista_lista_signals, [])
        lista_lista_pos = sum(lista_lista_pos, [])

        df_final = pd.DataFrame({"date": datas_full, "ticker": lista_lista_pos, "signals": signals_full})

        # agrup por ticker para se calcular o report
        df_final.columns = ["date", "ticker", "signal"]
        lista_df = []
        for ticker, df in df_final.groupby("ticker"):
            lista_df.append(df)

        return lista_df, df_final

    '''

     - Retorna o dataframe ja com as boletas de zeragem.

    '''

    def fix_zeragem(self, rep, type_trade='long', hedge=True):

        # if long trade
        if type_trade == 'long':
            module = 1
        else:
            module = -1

        _temp1 = rep[rep["signal"] == (-1 * module)]
        _temp2 = rep[rep["signal"] != (-1 * module)]
        _temp4 = _temp1.copy()

        # fixing the last sell(no pnl)
        # Transforma a ultima venda para compra.
        _temp3 = _temp1.copy()
        _temp3["qtty"] = (-1) * _temp3["qtty"]
        _temp3["signal"] = (-1) * _temp3["signal"]
        _temp3["signal_2"] = (-1) * _temp3["signal_2"]
        _temp3["notional"] = (-1) * _temp3["notional"]

        _temp3["carry_eqt"] = (-1) * _temp3["carry_eqt"]
        _temp3["mkt_value"] = (-1) * _temp3["mkt_value"]
        _temp3["pnl_daily_nominal_EQT"] = (-1) * _temp3["pnl_daily_nominal_EQT"]
        _temp3["qtty_hedge"] = (-1) * _temp3["qtty_hedge"]
        _temp3["carry_hedge"] = (-1) * _temp3["carry_hedge"]
        _temp3["pnl_cdi_hedge"] = (-1) * _temp3["pnl_cdi_hedge"]
        _temp3["pnl_daily_nominal_Hedge"] = (-1) * _temp3["pnl_daily_nominal_Hedge"]
        _temp3["txBroker"] = 0
        _temp3["txBrokerEqt"] = 0
        _temp3["txBrokerHedge"] = 0

        # hege
        if hedge:
            _temp3["pnl_cdi_eqt"] = ((-1) * _temp3["pnl_daily_nominal_EQT"]) + ((-1) * _temp3["carry_eqt"]) + (
                        (-1) * _temp3["pnl_daily_nominal_Hedge"]) + ((-1) * _temp3["carry_hedge"])
            _temp3["pnl_total"] = _temp3["pnl_cdi_eqt"]

        else:
            _temp3["pnl_cdi_eqt"] = ((-1) * _temp3["pnl_daily_nominal_EQT"]) + ((-1) * _temp3["carry_eqt"])
            _temp3["pnl_total"] = _temp3["pnl_cdi_eqt"]

        _temp4["pnl_daily_nominal_EQT"] = 0
        _temp4["pnl_cdi_hedge"] = 0
        _temp4["pnl_cdi_hedge"] = 0
        _temp4["pnl_daily_nominal_Hedge"] = 0
        _temp4["pnl_cdi_eqt"] = _temp4["txBrokerEqt"] + _temp4["borrow_cost"]
        _temp4["pnl_total"] = _temp4["txBrokerEqt"] + _temp4["borrow_cost"]
        _temp4["carry_eqt"] = 0
        _temp4["carry_hedge"] = 0
        _temp4["px_entry"] = _temp4["close"]
        _temp4["px_entry_hedge"] = _temp4["close_index"]

        if hedge:
            _temp4["pnl_cdi_hedge"] = _temp3["txBrokerHedge"]
            _temp4["pnl_daily_nominal_Hedge"] = _temp3["txBrokerHedge"]

        # flag das vendas
        _temp2['flag_casflow'] = np.where(_temp2['signal'] == module, True, False)

        # flag das compras
        _temp3["flag_casflow"] = False
        _temp4["flag_casflow"] = True

        df_final = pd.concat([_temp2, _temp3, _temp4]).sort_values("date")

        return df_final

    '''
        * retorna um dataframe como numero de trades positivos e negativos para todo o espaço de parametros

        * recebe a lista de dataframes de reps
    '''

    '''
    posterior: adiciona a coluna com a  data d0 + N dias
    anterior: adiciona a coluna com a  data d0 - N dias

    output: dataframe com as tadas calendarios  +/- offsets

    '''


def generate_list_chunks(self, _df_params, _df_params_full, period_reset):
    current_pos = 0
    lista_df_param = []
    lista_df_param_full = []

    # em order crescente !
    datas_backtest = _df_params_full.sort_values("date")["date"].unique().tolist()

    len_period = len(datas_backtest)
    n_resets = math.ceil(len_period / period_reset)

    for i in range(n_resets):
        chunk_of_dates = datas_backtest[current_pos:(period_reset + current_pos)]
        dt_min = min(chunk_of_dates)
        dt_max = max(chunk_of_dates)

        df_chunk_param = _df_params[(_df_params["date"] >= dt_min) & (_df_params["date"] <= dt_max)]
        df_chunk_param_full = _df_params_full[(_df_params_full["date"] >= dt_min) & (_df_params_full["date"] <= dt_max)]

        lista_df_param.append(df_chunk_param)
        lista_df_param_full.append(df_chunk_param_full)

        current_pos += period_reset

    return lista_df_param, lista_df_param_full

    def pos_ant_dates(self, df_params, df_cal, lista_ofs=[0, 1, 2, 3, 4, 5, 6], tipo='both'):

        dts = df_params[["date"]].drop_duplicates().sort_values("date")

        if tipo == 'both':

            # anterior
            for of in lista_ofs:
                dts["date_d-{}".format(of)] = dts["date"].shift(of)

            # posterior
            for of in lista_ofs:
                dts["date_d+{}".format(of)] = dts["date"].shift(-of)

        elif tipo == 'anterior':

            for of in lista_ofs:
                dts["date_d-{}".format(of)] = dts["date"].shift(of)

        elif tipo == 'posterior':

            for of in lista_ofs:
                dts["date_d+{}".format(of)] = dts["date"].shift(-of)

        else:

            pass

        dts_annt = pd.merge(df_cal, dts, left_on=["date"], right_on=["date"], how="inner").drop("ticker", axis=1)
        dts_annt["date_annt"] = dts_annt["date"]

        return dts_annt

    def count_number_NegPos(self, lista_rep):

        lista_df_trades = []
        for temp in lista_rep:
            lista = []

            if not temp.empty:

                temp = temp.sort_values("date")
                dates = temp["date"].unique().tolist()

                lista_pnl_trades = []
                lista_temp = []
                acc = 0
                for el in range(len(dates)):

                    print(el)

                    temp2 = temp[temp["date"] == dates[el]]

                    if temp2.__len__() > 1:

                        acc = acc + temp2["pnl_total"].sum()
                        lista_pnl_trades.append(acc)
                        acc = 0

                    else:

                        acc = acc + temp2["pnl_total"].sum()

                df_trades = pd.DataFrame({"pnl_trades": lista_pnl_trades})
                df_trades["param"] = temp2['param'].iloc[0]
                df_trades["type_trade"] = temp2['type_trade'].iloc[0]
                lista_df_trades.append(df_trades)

        resumo = pd.concat(lista_df_trades)
        pos = resumo[resumo["pnl_trades"] > 0].groupby(["param", "type_trade"]).agg(
            {"pnl_trades": "count"}).reset_index()
        neg = resumo[resumo["pnl_trades"] < 0].groupby(["param", "type_trade"]).agg(
            {"pnl_trades": "count"}).reset_index()

        pos.columns = ["param", "type_trade", "n_trades_positives"]
        neg.columns = ["param", "type_trade", "n_trades_negatives"]
        neg_pos = pd.merge(neg, pos, left_on=["param", "type_trade"], right_on=["param", "type_trade"], how="outer")

        return neg_pos

    def generateCashflow(self, rep, df_params, df_cdi, flag_cash_pnl=True, cash_subscription=400000, type_trade='long',
                         hedge=True):

        rep = rep.copy()
        cash_subscription = cash_subscription
        rep["mkt_value"] = rep["qtty"] * rep["close"]
        # rep = rep[(rep["ticker"]=='TNLP3 BS Equity')&(rep["date"]<=datetime(2007,6,18).date())]
        rep = self.fix_zeragem(rep, type_trade=type_trade)

        tp = rep.copy()
        tp['borrow_cost_acc'] = tp.groupby(by=['ticker'])['borrow_cost'].transform(lambda x: x.cumsum())

        if not hedge:
            saida_caixa = tp[(tp["signal"] == 1) & (tp["flag_casflow"] == True)]
            saida_caixa["Tocash"] = (-1) * saida_caixa["px_entry"] * saida_caixa["qtty"] + saida_caixa["txBroker"] + \
                                    saida_caixa["borrow_cost_acc"]

            entrada_caixa = tp[(tp["signal"] == -1) & (tp["flag_casflow"] == True)]
            entrada_caixa["Tocash"] = (-1) * entrada_caixa["close"] * entrada_caixa["qtty"] + entrada_caixa["txBroker"]


        else:

            saida_caixa = tp[(tp["signal"] == 1) & (tp["flag_casflow"] == True)]
            saida_caixa["Tocash"] = (-1) * saida_caixa["px_entry"] * saida_caixa["qtty"] + saida_caixa["txBrokerEqt"] + \
                                    saida_caixa["borrow_cost_acc"]

            entrada_caixa = tp[(tp["signal"] == -1) & (tp["flag_casflow"] == True)]
            entrada_caixa["Tocash"] = (-1) * entrada_caixa["close"] * entrada_caixa["qtty"] + entrada_caixa[
                "txBrokerEqt"]

            saida_caixa_hedge = tp[(tp["signal"] == -1) & (tp["flag_casflow"] == True)]
            saida_caixa_hedge["Tocash"] = (-1) * saida_caixa_hedge["px_entry_hedge"] * saida_caixa_hedge["qtty_hedge"] + \
                                          saida_caixa_hedge["txBrokerHedge"]

            entrada_caixa_hedge = tp[(tp["signal"] == 1) & (tp["flag_casflow"] == True)]
            entrada_caixa_hedge["Tocash"] = (-1) * entrada_caixa_hedge["close_index"] * entrada_caixa_hedge[
                "qtty_hedge"] + entrada_caixa_hedge["txBrokerHedge"]

            to_cash = pd.concat([saida_caixa, entrada_caixa, saida_caixa_hedge, entrada_caixa_hedge]).groupby(
                ["date"]).agg({"Tocash": "sum"}).reset_index()

            df_cashflow = pd.DataFrame({"date": rep["date"].unique().tolist()})
            df_cashflow = pd.merge(df_cashflow, to_cash, left_on="date", right_on="date", how="left")

        # valor inicial de cash. Em d0 ja tem movimentacao de caixa pela compra de stocks
        df_cashflow["cash"] = cash_subscription
        df_cashflow["Tocash"].fillna(0, inplace=True)
        df_cashflow["Tocash"] = df_cashflow["Tocash"].cumsum()

        # arrumar antes
        df_cashflow["cash"] = df_cashflow["cash"] + df_cashflow["Tocash"]
        df_cashflow["cash"].fillna(method='ffill', inplace=True)

        if not hedge:

            rep["mkt_value"] = rep["qtty"] * rep["close"]

        else:
            rep["mkt_value"] = (rep["qtty"] * rep["close"]) + (rep["qtty_hedge"] * rep["close_index"])

        mkt_value = rep.groupby("date").agg({"mkt_value": "sum"}).reset_index()
        df_cashflow = pd.merge(df_cashflow, mkt_value, left_on="date", right_on="date", how="left")
        df_cashflow["pl"] = df_cashflow["cash"] + df_cashflow["mkt_value"]
        df_cashflow = df_cashflow.sort_values("date", ascending=True)

        _dts = df_params[["date"]].drop_duplicates().sort_values("date")
        df_cashflow = pd.merge(_dts, df_cashflow, left_on="date", right_on="date", how="left")

        dd = df_cashflow.groupby("date").agg({"Tocash": "mean"}).reset_index()
        df_cdi = pd.merge(dd, df_cdi, left_on="date", right_on="date", how="left")[["date", "carry"]]
        df_cdi["carry"].fillna(method='ffill', inplace=True)
        df_cdi["carry_2"] = df_cdi["carry"] + 1
        df_cdi["carry_2"].iloc[0] = 1

        df_cashflow = pd.merge(df_cashflow, df_cdi, left_on="date", right_on="date", how="inner")

        if flag_cash_pnl:
            df_cashflow["cash_pnl_daily"] = (df_cashflow["cash"] * df_cashflow["carry"])
            df_cashflow["cash_pnl"] = df_cashflow["cash_pnl_daily"].cumsum()
            df_cashflow["pl"] = df_cashflow["pl"] + df_cashflow["cash_pnl"]

        else:
            df_cashflow["cash_pnl_daily"] = 0
            df_cashflow["cash_pnl"] = 0

        df_cashflow.ffill(axis=0, inplace=True)

        df_cashflow = df_cashflow[~pd.isnull(df_cashflow["pl"])]

        df_cashflow["rent"] = df_cashflow["pl"] / df_cashflow["pl"].iloc[0]
        df_cashflow["carry_2"] = df_cashflow.apply(lambda x: 1 if x["mkt_value"] == 0 else x["carry_2"], axis=1)
        df_cashflow["acc_rateReturn"] = df_cashflow["carry_2"].cumprod()
        df_cashflow["ratio"] = 100 * (df_cashflow["rent"] / df_cashflow["acc_rateReturn"])

        return df_cashflow


def generae_statistcs_interface(temp, cota_cdi, ret_dd=False):
    lista_res = []
    lista_res_dd = []
    lista_res2 = []
    lista_names = ['Total Return', 'CDI+', 'Pos/Total-Y(%)-roll(nominal)',
                   '(months > CDI)/total', 'Sharpe(M)', 'Sharpe(250d)', 'Kelly_C(250d)',
                   'MMD(%)', 'Worst_750D', 'Worst_250D', 'Min_Daily(%)', 'Max_Recover(days)'
                   ]

    cols = temp.columns.tolist()[1:]
    cols = [el for el in cols if el != 'month']
    lista_shift = [22, 252]
    for col in cols[:]:

        # print(" a coluna :{}".format(col))

        temp2 = temp[["date", col]].copy()
        temp2 = pd.merge(temp2, cota_cdi, left_on=["date"], right_on=["date"], how="inner")
        temp2["month"] = temp2["date"].apply(lambda x: x.month)
        temp2[col] = temp2[col] / temp2[col].iloc[0]
        temp2["cota_cdi"] = temp2["cota_cdi"] / temp2["cota_cdi"].iloc[0]

        for w in lista_shift:
            temp2["shift_{}".format(w)] = temp2[col].shift(w)
            temp2["shift_{}_cdi".format(w)] = temp2["cota_cdi"].shift(w)

            temp2["return_nom_{}".format(w)] = temp2[col] / temp2["shift_{}".format(w)] - 1
            temp2["return_CDI_{}".format(w)] = temp2["cota_cdi"] / temp2["shift_{}_cdi".format(w)] - 1
            # temp2["return_nom_{}".format(w)] = temp2[col]/temp2["shift_{}".format(w)]-1

            temp2["excess_return_{}".format(w)] = temp2["return_nom_{}".format(w)] - temp2["return_CDI_{}".format(w)]

        temp2["cota_sobre_cdi"] = temp2[col] / temp2["cota_cdi"]

        temp2["close_d-1"] = temp2[col].shift(1)
        temp2["close_d-1"] = temp2["close_d-1"].fillna(method="bfill")

        temp2["close_d-1_cdi"] = temp2["cota_cdi"].shift(1)
        temp2["close_d-1_cdi"] = temp2["close_d-1_cdi"].fillna(method="bfill")

        temp2["log_return"] = sqrt(252) * (np.log(temp2[col]) - np.log(temp2["close_d-1"]))

        temp2["pnl_cdi"] = temp2["cota_cdi"] - temp2["close_d-1_cdi"]

        temp2["pnl"] = temp2[col] - temp2["close_d-1"]

        # calc DD norminal
        temp2["rollmax"] = temp2[col].cummax()
        temp2["rel_to_max"] = 100 * (temp2[col] / temp2["rollmax"] - 1)
        temp2["drawdown"] = temp2["rel_to_max"].apply(lambda x: x if x < 0 else 0)

        # calc DD CDI
        temp2["rollmax_CDI"] = temp2["cota_sobre_cdi"].cummax()
        temp2["rel_to_max_CDI"] = 100 * (temp2["cota_sobre_cdi"] / temp2["rollmax_CDI"] - 1)
        temp2["drawdown_CDI"] = temp2["rel_to_max_CDI"].apply(lambda x: x if x < 0 else 0)

        temp2["daily_excess_return"] = temp2["pnl"] - temp2["pnl_cdi"]

        # 1) retorno total nominal
        retorno_norminal = 100 * (temp2[temp2["date"] == temp2["date"].max()].iloc[0][col] / 1 - 1)

        # 2) rendimento acima do CDI
        val_abova_cdi = 100 * (temp2[temp2["date"] == temp2["date"].max()].iloc[0][col] -
                               temp2[temp2["date"] == temp2["date"].max()].iloc[0]["cota_cdi"])

        # 3)
        max_drawdow = (temp2["drawdown"].min())

        # 4)
        min_daily = 100 * (temp2["pnl"].min())

        # 5)
        max_daily = 100 * (temp2["pnl"].max())

        # 6)
        sharpe_250d = (252 * temp2["daily_excess_return"].mean()) / (
                    np.sqrt(252) * np.std(temp2["daily_excess_return"]))

        # 7 vol
        vol_252 = 100 * (np.std(temp2["log_return"].tolist(), ddof=1))

        ################################################ RECOVERY dawdown NOMINAL
        ds = temp2["drawdown"].tolist()
        lista_rec = []
        cc = 0
        for i in range(len(ds)):

            if ds[i] == 0:

                lista_rec.append(cc)
                cc = 0

            else:

                cc += 1

        if cc > 0:
            lista_rec.append(cc)

        lista_rec = [el for el in lista_rec if el != 0]

        # 7)
        max_rec_dd = max(lista_rec)

        # 8)
        mean_rec_dd = np.mean(lista_rec)

        ################################################ RECOVERY dawdown CDI

        ds = temp2["drawdown_CDI"].tolist()
        lista_rec = []
        cc = 0
        for i in range(len(ds)):

            if ds[i] == 0:

                lista_rec.append(cc)
                cc = 0

            else:

                cc += 1

        if cc > 0:
            lista_rec.append(cc)

        lista_rec = [el for el in lista_rec if el != 0]

        # 7)
        max_rec_dd_cdi = max(lista_rec)

        # 8)
        mean_rec_dd_cdi = np.mean(lista_rec)

        ################################################ RECOVERY dawdown CDI

        # 9) meses acima cdi
        meses_acima_cdi_perc = 100 * (temp2[temp2["excess_return_{}".format(22)] > 0].__len__() / temp2.__len__())

        # 10) anos (252) acima CDI
        anos_acima_cdi_perc = 100 * (temp2[temp2["excess_return_{}".format(252)] > 0].__len__() / temp2.__len__())

        #################### PARA O KELLY
        # 11) meses > 0
        PERC_POS_MES = 100 * (temp2[temp2["return_nom_{}".format(22)] > 0].__len__() / temp2.__len__())
        MEAN_POS_MES = abs(temp2[temp2["return_nom_{}".format(22)] > 0]["return_nom_{}".format(22)].mean()) / abs(
            temp2[temp2["return_nom_{}".format(22)] <= 0]["return_nom_{}".format(22)].mean())

        PERC_POS_ANO = 100 * (temp2[temp2["return_nom_{}".format(252)] > 0].__len__() / temp2.__len__())
        MEAN_POS_ANO = abs(temp2[temp2["return_nom_{}".format(252)] > 0]["return_nom_{}".format(252)].mean()) / abs(
            temp2[temp2["return_nom_{}".format(252)] <= 0]["return_nom_{}".format(252)].mean())

        kelly_mes = ((PERC_POS_MES / 100) * MEAN_POS_MES - (1 - (PERC_POS_MES / 100))) / MEAN_POS_MES
        kelly_ano = ((PERC_POS_ANO / 100) * MEAN_POS_ANO - (1 - (PERC_POS_ANO / 100))) / MEAN_POS_ANO

        df = pd.DataFrame({"Total Return(%)": [retorno_norminal],
                           'CDI+': [val_abova_cdi],
                           "volatility": [vol_252],
                           'Pos/Total-Y(%)-roll(nominal)': [PERC_POS_ANO],
                           'Pos/Total-M(%)-roll(nominal)': [PERC_POS_MES],
                           'Pos/Total-Y(%)-roll(CDI)': [anos_acima_cdi_perc],
                           'Pos/Total-M(%)-roll(CDI)': [meses_acima_cdi_perc],
                           'KellyC_MES': [kelly_mes],
                           'KellyC_ANO': [kelly_ano],
                           'Sharpe(Y)': [sharpe_250d],
                           'MaxDrawDown': [max_drawdow],
                           'Min_DailyPnL': [min_daily],
                           'MaxRecovery_Nominal(days)': [max_rec_dd],
                           'MeanRecovery_Nominal(days)': [mean_rec_dd],
                           'MaxRecovery_CDI(days)': [max_rec_dd_cdi],
                           'MeanRecovery_CDI(days)': [mean_rec_dd_cdi],
                           })

        df_before = df.copy()

        df['Total Return(%)'] = df['Total Return(%)'].map('{:,.0f}%'.format)
        df['CDI+'] = df['CDI+'].map('{:,.0f}%'.format)
        # return df
        df['Pos/Total-Y(%)-roll(nominal)'] = df['Pos/Total-Y(%)-roll(nominal)'].map('{:,.0f}%'.format)
        df['Pos/Total-M(%)-roll(nominal)'] = df['Pos/Total-M(%)-roll(nominal)'].map('{:,.0f}%'.format)
        df['Pos/Total-Y(%)-roll(CDI)'] = df['Pos/Total-Y(%)-roll(CDI)'].map('{:,.0f}%'.format)
        df['Pos/Total-M(%)-roll(CDI)'] = df['Pos/Total-M(%)-roll(CDI)'].map('{:,.0f}%'.format)
        df['KellyC_MES'] = df['KellyC_MES'].map('{:,.1f}'.format)
        df['KellyC_ANO'] = df['KellyC_ANO'].map('{:,.1f}'.format)
        df['Sharpe(Y)'] = df['Sharpe(Y)'].map('{:,.2f}'.format)
        df['MaxDrawDown'] = df['MaxDrawDown'].map('{:,.1f}%'.format)
        df['Min_DailyPnL'] = df['Min_DailyPnL'].map('{:,.1f}%'.format)

        df['MeanRecovery_Nominal(days)'] = df['MeanRecovery_Nominal(days)'].map('{:,.0f}'.format)
        df['MeanRecovery_CDI(days)'] = df['MeanRecovery_CDI(days)'].map('{:,.0f}'.format)

        df['volatility'] = df['volatility'].map('{:,.1f}%'.format)

        df = df.T

        df = df.reset_index()
        df.columns = ["param", col]
        lista_res.append(df)

        df2 = df_before.T.reset_index()
        df2.columns = ["param", col]

        temp2_d = temp2[["date", "drawdown"]].copy()
        temp2_d.columns = ["date", "MaxDD_{}".format(col)]
        lista_res_dd.append(temp2_d)

        lista_res2.append(df2)

    for el in range(len(lista_res)):
        if el == 0:
            _df = lista_res[el]

        else:
            _df = pd.merge(_df, lista_res[el], left_on=["param"], right_on=["param"])

    _df.columns = ['_' if el == 'param' else el for el in _df.columns.tolist()]

    print("vamos ver como estaaaaaaaaaaaaaaaaaaaaaa bbbb 111")
    print(_df)

    for el in range(len(lista_res2)):

        if el == 0:
            _df2 = lista_res2[el]

        else:
            _df2 = pd.merge(_df2, lista_res2[el], left_on=["param"], right_on=["param"])

    _df2.columns = ['_' if el == 'param' else el for el in _df2.columns.tolist()]

    #print("vamos ver como estaaaaaaaaaaaaaaaaaaaaaa bbbb")
    #print(_df2)

    for el in range(len(lista_res_dd)):

        if el == 0:
            _df_dd = lista_res_dd[el]

        else:
            print(_df_dd)
            _df_dd = pd.merge(_df_dd, lista_res_dd[el], left_on=["date"], right_on=["date"])

    cols = _df2.columns.tolist()[1:]

    cols_neg = ['MaxDrawDown', 'Min_DailyPnL', 'MaxRecovery_Nominal(days)', 'MeanRecovery_Nominal(days)',
                'MaxRecovery_CDI(days)', 'MeanRecovery_CDI(days)', 'volatility']

    print("PRIMEIRO")
    print(_df)

    print("SEGUNDO")
    print(_df2)

    print("comecando o improtvemnt ")

    #_df2[cols[0]] = 10000

    _df2["Avg_Improvement"] = _df2.apply(lambda x: np.mean([100 * ((x[cols[el + 1]] - x[cols[0]]) / x[cols[0]]) for el in range(len(cols[1:]))]) if ((x["_"] not in cols_neg)&( abs(x[cols[0]]) > 0)) else (-np.mean([100 * ((x[cols[el + 1]] - x[cols[0]]) / x[cols[0]]) for el in range(len(cols[1:]))]) if abs(x[cols[0]]) > 0 else np.nan)  , axis=1)

    print("SEGUNDO APOS IMPROVE")
    print(_df2)

    _df = pd.merge(_df, _df2[["_", "Avg_Improvement"]], left_on=["_"], right_on="_", how="inner")

    print("PRIMEIRO APOS MERGE")
    print(_df2)

    _df['Avg_Improvement'] = _df['Avg_Improvement'].map('{:,.1f}%'.format)

    print("AFTER 1")
    print(_df)

    print("AFTER 1")
    print(_df2)

    if not ret_dd:

        return _df

    else:

        return _df_dd


class LowVol(bm):

    def __init__(self, country='brazil', start=4350, offset=4350, index='IBX', flag_div_adj=1,
                 type_trades=['long', 'short'], flag_signal=False, local_database=False, dict_param=False, nbin=7,
                 backtest_di=True):

        bm.__init__(self)

        self.index = index
        self.country = country
        self.start = start
        self.offset = offset
        self.flag_control = False
        self.flag_index = False
        self.flag_crowd = False
        self.flag_div_adj = flag_div_adj
        self.lista_window = [20]
        self.lista_window_angle = [120]
        self.lista_window_beta = [500]
        self.lista_mas = [10, 20, 30]
        self.lista_return_price = [250, 180, 250]
        self.lista_window_beta = [500]
        self.lista_excess_return = [250, 180]
        # self.window_vol = [20,30,60,90,120,180,250,300,350,400,500]
        self.window_vol = [100, 120, 180, 240]
        self.df_params = pd.DataFrame()
        self.hedge = False
        self.type_trades = type_trades
        self.current_vol = 30
        self.flag_signal = flag_signal
        self.flag_trade = None
        self.col_stop = None
        self.least = None
        self.TOL = None
        self.window_enter = 3
        self.col_rr = None
        self.tam_rank_in_b = None
        self.tam_rank_out_b = None
        self.local_database = local_database
        self.dict_param = dict_param
        self.nbin = nbin
        self.multiplo = None
        self.backtest_di = backtest_di
        self.freq = None

    def calc_all_tickers(self, tickers, df_precos):

        lista_df = []

        for ticker in tickers:

            df_desc = df_precos[df_precos["ticker"] == ticker]

            try:

                ss = [(0.3), (0), (0.75), 1.2]
                print("calculando o ticker: {}".format(ticker))
                df_desc = compute_all(df_desc,
                                      spread=False,
                                      release_memory=False,
                                      ma=True,
                                      control_liq=True,
                                      window_liq=[60],
                                      col_ma_names=["close"],
                                      lista_mas=[[5, 10, 7, 21, 30, 45, 20, 40, 180]],
                                      want_daily_return=True,
                                      # relative_return_general = True,
                                      returns_general=False,
                                      col_return_names=['close'],
                                      lista_returns=[[60, 90, 120, 180, 240]],
                                      k_liq=8,
                                      prop=1,
                                      betas=False
                                      # window_betas = [100,180,400]
                                      )

                lista_df.append(df_desc)
                # print("computed")

            except:
                print("Coud not compute features for: {}".format(ticker))

        return pd.concat(lista_df)

    def BuySellConditions(self, df, type_trade='long', col_condition='excess_250'):

        # col = 'return_{}'.format(N*nbin)
        def help_buy_condition(linha):

            # print(linha["close"])
            # if (linha["flag_buy"])&(linha["pnl_total_ranking"]>0):
            # if ((linha["flag_buy"])&(linha["pnl_total_ranking"]!=-1.000000e+12)):
            if ((linha["flag_buy"])):

                return True

            else:

                return False

        def help_sell_condition(linha):
            if linha["flagsell"]:
                # if not linha["flag_buy"]:

                return True

            else:
                return False

        # NOT IMPLEMENTED
        def help_buy_condition_short(linha):
            # print(linha["close"])
            # if ((linha["flag_buy"])&(linha["pnl_total_ranking"]!=-1.000000e+12)):
            if ((linha["flag_buy"])):

                return True


            else:

                return False

        def help_sell_condition_short(linha):

            # if not linha["flag_buy"]:
            if linha["flagsell"]:

                return True

            else:

                return False

        # gerando flags de condicoes de compra e venda
        if type_trade == 'long':

            # df["buy_condition"] = df.apply(help_buy_condition,axis=1)
            # df["sell_condition"] = df.apply(help_sell_condition,axis=1)

            #             df["buy_condition"] = True
            #             df["sell_condition"] = False

            df["buy_condition"] = df["flag_buy_long"]
            df["sell_condition"] = df["flagsell_long"]

            df["buy_d02"] = df["buy_condition"]
            df["sell_d02"] = df["sell_condition"]
            df["buy_dn2"] = df["buy_condition"]
            df["sell_dn2"] = df["sell_condition"]

        else:

            # df["buy_condition"] = df.apply(help_buy_condition_short,axis=1)
            # df["sell_condition"] = df.apply(help_sell_condition_short,axis=1)
            # df["buy_condition"] = True
            # df["sell_condition"] = False
            self.why = df.copy()
            df["buy_condition"] = df["flag_buy_short"]
            df["sell_condition"] = df["flagsell_short"]

            df["buy_d02"] = df["buy_condition"]
            df["sell_d02"] = df["sell_condition"]
            df["buy_dn2"] = df["buy_condition"]
            df["sell_dn2"] = df["sell_condition"]

        return df

    def getBacktestLists(self, df_params, df_params_full, column_ranking, column_ranking2, tam_rank_in, tam_rank_out,
                         type_trade='long', least=60):

        # df_params = df_params[df_params["flag_liq"]==False]

        self.pp1 = df_params.copy()

        if type_trade == 'long':

            # order = self.list_order[0]
            order = True

            df_params["rank_1"] = df_params.groupby("date")[self.col_ranking_long].rank(method="max", ascending=order,
                                                                                        pct=True, na_option="bottom")

        else:

            order = False
            # order = self.list_order[1]
            df_params["rank_1"] = df_params.groupby("date")[self.col_ranking_short].rank(method="max", ascending=order,
                                                                                         pct=True, na_option="bottom")

        df_params["rank_1"] = 100 * df_params["rank_1"]
        df_params["rank_1"] = round(df_params["rank_1"])

        self.pp = df_params.copy()

        rank_in = df_params[df_params["rank_1"] < (self.tam_rank_in_b + 1)].groupby("date")['ticker'].apply(list).apply(
            list).tolist()
        rank_out = df_params[df_params["rank_1"] < (self.tam_rank_out_b + 1)].groupby("date")['ticker'].apply(
            list).apply(list).tolist()

        # self.rank_out_back = rank_out
        # self.rank_in_back = rank_in

        ######################################## INICIO NOVA ORDENAÇÃ0 #########################################

        # ignore
        # result = list(set(sum(rank_in, []) + sum(rank_out, [])))

        result = list(set(reduce(lambda x, y: x + y, rank_in) + reduce(lambda x, y: x + y, rank_out)))

        # lista de tickers
        tickers_pos = pd.DataFrame({"ticker": result})
        lista_tickers = tickers_pos.sort_values("ticker")["ticker"].values

        # self.df_params_full2 = df_params_full.copy()

        # serie historica completa com os tickers que ja pertenceram ao rank_in UNION rank_out
        df_new = pd.merge(df_params_full, tickers_pos, left_on="ticker", right_on="ticker", how="inner")
        df_new = self.BuySellConditions(df_new, type_trade, col_condition='angle_180')

        df_new = df_new.sort_values("date")

        # self.df_new = df_new.copy()

        # ja foi retirado os nans !
        lista_lista_buydn_cond = df_new[["buy_dn2", "ticker"]].groupby(['ticker'])['buy_dn2'].apply(list).apply(
            list).tolist()
        lista_lista_selldn_cond = df_new[["sell_dn2", "ticker"]].groupby(['ticker'])['sell_dn2'].apply(list).apply(
            list).tolist()
        lista_lista_buyd0_cond = df_new[["buy_d02", "ticker"]].groupby(['ticker'])['buy_d02'].apply(list).apply(
            list).tolist()
        lista_lista_selld0_cond = df_new[["sell_d02", "ticker"]].groupby(['ticker'])['sell_d02'].apply(list).apply(
            list).tolist()

        ############ creating new sell conditions ################
        lista_tickers = lista_tickers.tolist()

        # to be used later on right ordre --------------- IMPORTANT ---------------------
        df_params_full = df_params_full.sort_values("date")
        df_new = df_new.sort_values("date")

        datas = df_new[df_new["ticker"] == lista_tickers[0]]["date"].tolist()
        # print("as datas")
        # print(datas)
        return rank_in, rank_out, lista_lista_buydn_cond, lista_lista_selldn_cond, lista_lista_buyd0_cond, lista_lista_selld0_cond, lista_tickers, df_params_full, datas,
        # return rank_in,rank_out,rank2_in,rank2_out,lista_lista_buydn_cond,lista_lista_selldn_cond,lista_lista_buyd0_cond,lista_lista_selld0_cond,lista_tickers,df_params_full,datas

    def get_data(self):

        rank_in = 15
        rank_out = 25
        rebalance_frequence = 20
        self.limit_buy = [0, 1, 2, 3, 4]
        self.limit_sell = [0, 1, 2, 3, 4]

        # get rate name for given index/country
        self.rate_name, self.index_lenght = self.get_rateName()

        # numero de papeis no rank)_in
        # tam_rank_in = round(self.index_lenght*(rank_in/100))
        tam_rank_in = round(self.index_lenght * (rank_in / 100))
        tam_rank_out = round(self.index_lenght * (rank_out / 100))

        print("the rank_in is: {}".format(tam_rank_in))
        print("the rank_out is: {}".format(tam_rank_out))

        if not self.local_database:

            print("veio no nao local")

            query = '''SELECT date,ticker,close FROM backtesting.rates_global where country ='{}' and prazo ='1y' order by date desc;'''.format(
                self.country)
            df_precos = pd.read_sql(query, self.sql_con)

            # filling cdi
            df_precos = self.fill_carry_inf(df_precos)

            # self.uy = df_precos.copy()

            if not self.backtest_di:

                self.pc_ibx = priceIndex(self.sql_con, (self.start + 1000), (self.offset + 1 + max(janelas) + 100),
                                         index=self.index)
                # adding index price columns
                df_precos = pd.merge(df_precos, self.pc_ibx, left_on=["date"], right_on=["date"], how="inner")

            else:

                df_precos = df_precos[df_precos["close"] != 'LAST_PRICE']
                precos = df_precos.sort_values("date", ascending=False)[["date", "close"]]
                _l = precos["close"].iloc[1:].tolist()
                _l.append(None)
                precos["close_d-1"] = _l
                precos = precos.iloc[:-1]
                precos.columns = ["date", "close_index", "close_d-1_index"]
                self.pc_ibx = precos.copy()
                self.pc_ibx["close_index"] = 1
                self.pc_ibx["close_d-1_index"] = 1
                df_precos = pd.merge(df_precos, self.pc_ibx, left_on=["date"], right_on=["date"], how="inner")

        else:

            # df_precos = pd.read_excel("ativos_sinteticos.xlsx")
            # df_precos = pd.read_csv("turnover3.csv").drop("Unnamed: 0",axis=1)
            # df_precos = pd.read_csv("turnover3_sptsx.csv").drop("Unnamed: 0",axis=1)[["date","ticker","close"]].drop_duplicates()
            # df_precos = pd.read_csv("sxxe_base.csv").drop("Unnamed: 0",axis=1)[["date","ticker","close"]].drop_duplicates()
            if self.index == 'IPSA':

                df_precos = pd.read_csv("ipsa_base.csv").drop("Unnamed: 0", axis=1)[
                    ["date", "ticker", "close"]].drop_duplicates()

            elif self.index == 'IBX':

                df_precos = pd.read_csv("base_simulacao_j.csv").drop("Unnamed: 0", axis=1)
                # df_precos = pd.read_excel("CUPOM2.xlsx")[["date","ticker","close"]].drop_duplicates()
                # df_precos["date"] = df_precos["date"].apply(lambda x: x.date())

            elif self.index == 'SPTSX':

                df_precos = pd.read_csv("turnover3_sptsx.csv").drop("Unnamed: 0", axis=1)[
                    ["date", "ticker", "close"]].drop_duplicates()

            elif self.index == 'AS51':

                df_precos = pd.read_csv("turnover3_as51.csv").drop("Unnamed: 0", axis=1)[
                    ["date", "ticker", "close"]].drop_duplicates()

            elif self.index == 'MEXBOL':

                df_precos = pd.read_csv("mexbol_databse_2022_04.csv").drop("Unnamed: 0", axis=1)[
                    ["date", "ticker", "close"]].drop_duplicates()


            elif self.index == 'SXXE':

                df_precos = pd.read_csv("sxxe_base.csv").drop("Unnamed: 0", axis=1)[
                    ["date", "ticker", "close"]].drop_duplicates()


            elif self.index == 'TOP40':
                # ("top40_base.csv")
                df_precos = pd.read_csv("top40_base.csv").drop("Unnamed: 0", axis=1)[
                    ["date", "ticker", "close"]].drop_duplicates()

            else:

                print("pais não encontrado ")

            df_precos = df_precos[df_precos["close"] != 'LAST_PRICE']
            df_precos["date"] = df_precos["date"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").date())
            # df_precos["date_next"] = df_precos["date_next"].apply(lambda x: datetime.strptime(x,"%Y-%m-%d").date())
            df_precos = df_precos[df_precos["date"] >= datetime(2001, 12, 1).date()]
            df_precos["close"] = df_precos["close"].apply(lambda x: float(x))
            df_precos = df_precos.sort_values("date", ascending=False)

            # ADDING INDEX
            _df_precos = df_precos[df_precos["ticker"] == '{} Index'.format(self.index)]
            precos = _df_precos.sort_values("date", ascending=False)[["date", "close"]]
            _l = precos["close"].iloc[1:].tolist()
            _l.append(None)
            precos["close_d-1"] = _l
            precos = precos.iloc[:-1]
            precos.columns = ["date", "close_index", "close_d-1_index"]
            self.pc_ibx = precos.copy()

            # filling cdi
            df_precos = self.fill_carry_inf(df_precos)

            # adding index price columns
            df_precos = pd.merge(df_precos, self.pc_ibx, left_on=["date"], right_on=["date"], how="inner")

        # computing features

        ################################################## ADDDING VOLS COLUMN ###########################################
        df_desc = compute_all(df_precos[df_precos["ticker"] == '{} Index'.format(self.index)],
                              vol=True,
                              window_vol=[250],
                              entropy_features=False,
                              release_memory=False,
                              control_liq=False,
                              want_daily_return=True)

        vols_ibx = df_desc[["date", "vol_250"]].sort_values("date", ascending=False)
        vols_ibx.columns = ["date", "vol_250_index"]
        df_precos = pd.merge(df_precos, vols_ibx, left_on="date", right_on=["date"], how="left")
        df_precos["vol_250_index"] = df_precos["vol_250_index"].fillna(method="ffill")
        df_precos["vol_250_index"] = df_precos["vol_250_index"].fillna(method="bfill")
        ################################################## ADDDING VOLS COLUMN ###########################################

        self.ooo = df_precos.copy()

        # df_precos["vol_250_index"] = df_precos["vol_250_index"] /np.sqrt(252)
        # df_precos["daily_return"] = df_precos["daily_return"] /np.sqrt(252)

        tickers = df_precos["ticker"].unique().tolist()
        # tickers = tickers[0:10]
        df_params_full = self.calc_all_tickers(tickers, df_precos)

        if not 'year' in df_params_full.columns.tolist():
            df_params_full["year"] = df_params_full["date"].apply(lambda x: x.year)

        #self.df_indice = self.get_indexCompos(index=self.index)

        if self.index != 'IBX':

            if not self.backtest_di:

                df_params = pd.merge(df_params_full, self.df_indice, left_on=["year", "ticker"],
                                     right_on=["year", "ticker"], how="inner")

            else:

                df_params = pd.merge(df_params_full, self.df_indice, left_on=["year", "ticker"],
                                     right_on=["year", "ticker"], how="inner")

        else:

            # pass

            # if ibx, existe a base granular
            # df_params = self.fix_composition(df_params_full,n_min =-1)
            df_params = df_params_full.copy()
            # df_params = self.fix_composition(df_params_full)

        print("finished calculading features")

        return df_params, df_params_full, tam_rank_in, tam_rank_out

    def gen_report_beta(self, df_final, df_params_full, type_trade='long', notional=100000, borrow_cost=3,
                        postive_borrow_factor=0, flag_hedge=True, multi_tx=1, beta_adjust=False, beta_number=400,
                        normalize=False):

        # print("porra do relatorio 2")
        if type_trade == 'long':

            module_signal = 1

        else:

            module_signal = -1

        df_entradas = df_final[df_final["signal"] == module_signal * 1]
        df_vendas = df_final[df_final["signal"] == module_signal * -1]
        df_resto = df_final[df_final["signal"] != module_signal * 1]

        if flag_hedge:

            multi_hedge = 1

        else:

            multi_hedge = 0

        daily_cost = ((1 + (borrow_cost / 100)) ** (1 / 252)) - 1

        if not beta_adjust:

            df_entradas = pd.merge(df_entradas, df_params_full[["date", "ticker", "close", "close_index"]],
                                   left_on=["date", "ticker"], right_on=["date", "ticker"], how="inner")
            df_entradas.columns = ["signal", "date", "ticker", "px_entry", "px_entry_hedge"]
            df_entradas["qtty"] = round(notional / (module_signal * df_entradas["px_entry"]))
            # df_entradas["qtty"] = notional/(module_signal*df_entradas["px_entry"])


        else:

            # BETA NEUTRAL
            if not normalize:

                df_entradas = pd.merge(df_entradas, df_params_full[
                    ["date", "ticker", "close", "close_index", "beta_{}".format(beta_number), "flag_liq"]],
                                       left_on=["date", "ticker"], right_on=["date", "ticker"], how="inner")

            else:

                df_entradas = pd.merge(df_entradas, df_params_full[
                    ["date", "ticker", "close", "close_index", "beta_norm", "flag_liq"]], left_on=["date", "ticker"],
                                       right_on=["date", "ticker"], how="inner")

            df_entradas.columns = ["signal", "date", "ticker", "px_entry", "px_entry_hedge", "beta_entry", "flag_liq"]
            self.df_entradas = df_entradas.copy()

            def help_notional(linha):

                if linha["flag_liq"] == True:

                    return 100000

                else:

                    return 100000 / (linha["beta_entry"])

            df_entradas["notional"] = df_entradas.apply(help_notional, axis=1)

            # df_entradas["notional"] = 100000*((1)/(0.2/df_entradas["vol_mean"]))
            # df_entradas["qtty"] = round(df_entradas["notional"]/(module_signal*df_entradas["px_entry"]))
            df_entradas["qtty"] = df_entradas["notional"] / (module_signal * df_entradas["px_entry"])

        if flag_hedge:

            if not beta_adjust:

                df_entradas["qtty_hedge"] = round(notional / (-module_signal * df_entradas["px_entry_hedge"]))

            else:

                df_entradas["qtty_hedge"] = round(
                    df_entradas["notional"] / (-module_signal * df_entradas["px_entry_hedge"]))
                # df_entradas["qtty"] = round(df_entradas["notional"]/(module_signal*df_entradas["px_entry"]))#
        else:

            df_entradas["qtty_hedge"] = 0
            df_entradas["qtty_hedge"] = None

        df_all = pd.concat([df_entradas, df_resto]).sort_values(["ticker", "date"], ascending=[True, True])
        df_all = df_all.replace((np.nan, ''), (None, None))

        df_all["px_entry"] = df_all["px_entry"].fillna(method='ffill')
        df_all["qtty"] = df_all["qtty"].fillna(method='ffill')

        if flag_hedge:
            df_all["px_entry_hedge"] = df_all["px_entry_hedge"].fillna(method='ffill')
            df_all["qtty_hedge"] = df_all["qtty_hedge"].fillna(method='ffill')

        else:
            df_all["px_entry_hedge"] = 0
            df_all["qtty_hedge"] = 0

        df_all2 = pd.merge(df_all, df_params_full[
            ["date", "ticker", "close", "close_d-1", "close_index", "close_d-1_index", "carry"]],
                           left_on=["date", "ticker"], right_on=["date", "ticker"], how="inner")

        entradas_saidas = df_all2[~pd.isnull(df_all2["signal"])]
        positions = df_all2[pd.isnull(df_all2["signal"])]

        entradas_saidas["txBrokerEqt"] = (-1) * abs(
            entradas_saidas["close"] * abs(entradas_saidas["qtty"]) * 0.001 * multi_tx)

        if flag_hedge:

            entradas_saidas["txBrokerHedge"] = (-1) * abs(
                entradas_saidas["close_index"] * abs(entradas_saidas["qtty_hedge"]) * 0.0005 * multi_tx)

        else:

            entradas_saidas["txBrokerHedge"] = 0

        positions["txBrokerEqt"] = 0
        positions["txBrokerHedge"] = 0
        positions["txBroker"] = 0
        entradas_saidas["txBroker"] = entradas_saidas["txBrokerEqt"] + entradas_saidas["txBrokerHedge"]
        entradas_saidas["pnl_daily_nominal_Hedge"] = 0
        entradas_saidas["pnl_daily_nominal_EQT"] = 0
        entradas_saidas["carry_eqt"] = 0
        entradas_saidas["carry_hedge"] = 0
        entradas_saidas["borrow_cost_hedge"] = 0
        # borrow cost eqt
        if type_trade == 'long':

            entradas_saidas["borrow_cost"] = 0

        else:

            entradas_saidas_p1 = entradas_saidas[entradas_saidas["signal"] == module_signal]
            entradas_saidas_p2 = entradas_saidas[entradas_saidas["signal"] == -module_signal]

            entradas_saidas_p1["borrow_cost"] = 1 * module_signal * 1 * abs(
                entradas_saidas_p1["qtty"] * entradas_saidas_p1["px_entry"] * daily_cost)
            entradas_saidas_p2["borrow_cost"] = 0
            entradas_saidas = pd.concat([entradas_saidas_p1, entradas_saidas_p2])

        # start = time.time()
        positions["pnl_daily_nominal_EQT"] = (positions["close"] - positions["close_d-1"]) * positions["qtty"]
        positions["pnl_daily_nominal_Hedge"] = (positions["close_index"] - positions["close_d-1_index"]) * positions[
            "qtty_hedge"]
        positions["carry_hedge"] = (-1) * positions["close_index"] * positions["qtty_hedge"] * positions["carry"]
        positions["carry_eqt"] = (-1) * positions["close"] * positions["qtty"] * positions["carry"]

        # hedge borrow
        if flag_hedge:

            if type_trade == 'long':

                positions["borrow_cost_hedge"] = -1 * module_signal * abs(
                    positions["qtty_hedge"] * positions["px_entry_hedge"] * daily_cost)

            else:

                positions["borrow_cost_hedge"] = 0

        else:
            positions["borrow_cost_hedge"] = 0

        if type_trade == 'long':

            positions["borrow_cost"] = 1 * module_signal * postive_borrow_factor * abs(
                positions["qtty"] * positions["px_entry"] * daily_cost)

        else:

            positions["borrow_cost"] = 1 * module_signal * 1 * abs(
                positions["qtty"] * positions["px_entry"] * daily_cost)

        df_all = pd.concat([entradas_saidas, positions]).sort_values(["ticker", "date"], ascending=[True, True])

        df_all["pnl_cdi_eqt"] = df_all["pnl_daily_nominal_EQT"] + df_all["carry_eqt"] + df_all["borrow_cost"]
        df_all["pnl_cdi_hedge"] = df_all["pnl_daily_nominal_Hedge"] + df_all["carry_hedge"] + df_all[
            "borrow_cost_hedge"]
        df_all["pnl_total"] = df_all["pnl_cdi_hedge"] + df_all["pnl_cdi_eqt"] + df_all["txBroker"]

        daily_rep = df_all.groupby(["date"]).agg({"pnl_daily_nominal_EQT": "sum", "pnl_daily_nominal_Hedge": "sum",
                                                  "txBroker": "sum", "borrow_cost": "sum", "borrow_cost_hedge": "sum",
                                                  "carry_eqt": "sum", "carry_hedge": "sum",
                                                  "pnl_total": "sum"}).reset_index()

        daily_rep["year"] = daily_rep["date"].apply(lambda x: x.year)
        anual_rep = daily_rep.groupby("year").agg({"pnl_daily_nominal_EQT": "sum",
                                                   "pnl_daily_nominal_Hedge": "sum",
                                                   "txBroker": "sum",
                                                   "borrow_cost": "sum",
                                                   "borrow_cost_hedge": "sum",
                                                   "carry_eqt": "sum",
                                                   "carry_hedge": "sum",
                                                   "pnl_total": "sum"}).reset_index()

        df_all["year"] = df_all["date"].apply(lambda x: x.year)
        trades = df_all[(df_all["signal"] == (-module_signal * 1))]
        trades2 = trades.groupby("year").agg({"ticker": "count"}).reset_index()
        trades2.rename(columns={'ticker': 'n_trades'}, inplace=True)

        anual_rep = pd.merge(anual_rep, trades2, left_on="year", right_on="year", how="left")
        anual_rep["pnl/trade"] = anual_rep["pnl_total"] / anual_rep['n_trades']
        anual_rep["pnl/trade(%)"] = 100 * (anual_rep["pnl/trade"] / 100000)

        return df_all, daily_rep, anual_rep

    # type 1, with control variable and rebalance(dias corridos).
    # No external control variables
    def generate_signal_fast_J(self, type_trade, rank_in, rank_out, lista_lista_buydn_cond, lista_lista_selldn_cond,
                               lista_lista_buyd0_cond, lista_lista_selld0_cond, lista_tickers, df_params_full, datas,
                               rebalance_frequence=20):

        print("comeceeeeeeeeeee jjjj")
        lista_df = []

        if type_trade == 'long':

            module_signal = 1

        else:
            module_signal = -1

        count_days = 0
        data_max = max(datas)

        for el in range(len(lista_tickers)):

            flag_init = True
            lista_signal = []
            datas_ticker = []
            pos = False
            ticker = lista_tickers[el]

            count_days = 0

            # itera nas datas
            for el2 in range(len(lista_lista_buyd0_cond[el])):

                if not flag_init:

                    # nao tem posicoa, verifica se compra indepenen do sinal de rebalance
                    if not pos:

                        count_days += 1

                        if (((count_days - 1) % rebalance_frequence) == 0):

                            if (ticker in rank_in[el2]) & (not pos) & lista_lista_buydn_cond[el][el2] & (
                            not (lista_lista_selldn_cond[el][el2])):

                                if datas[el2] != data_max:

                                    pos = True
                                    # count_days += 1
                                    lista_signal.append(1 * module_signal)
                                    datas_ticker.append(datas[el2])

                                else:

                                    pass

                            else:

                                pass

                        else:

                            if (pos):

                                lista_signal.append(None)
                                datas_ticker.append(datas[el2])

                            else:

                                lista_signal.append(2)
                                datas_ticker.append(datas[el2])

                    else:

                        # tem posicao, incrementa a contagem
                        count_days += 1
                        # if rebalnace
                        if (((count_days - 1) % rebalance_frequence) == 0):

                            if ((lista_lista_selldn_cond[el][el2])) & (pos):

                                pos = False

                                lista_signal.append(-1 * module_signal)
                                lista_signal.append(None)
                                datas_ticker.append(datas[el2])
                                datas_ticker.append(datas[el2])


                            # esta no rank_in e ja tem posicao: hold
                            elif (ticker in rank_in[el2]) & (pos):

                                datas_ticker.append(datas[el2])
                                lista_signal.append(None)


                            # ja tem posicao e esta no rank_out: hold
                            elif (ticker in rank_out[el2]) & (pos):

                                lista_signal.append(None)
                                datas_ticker.append(datas[el2])

                            else:

                                pass


                        # if not rebalance, hold
                        else:
                            if (pos):
                                lista_signal.append(None)
                                datas_ticker.append(datas[el2])

                            else:

                                pass


                else:

                    flag_init = False

                    count_days += 1
                    if (((count_days - 1) % rebalance_frequence) == 0):

                        # nao tem posicao e esta no rank_in: compra
                        if (ticker in rank_in[el2]) & (not pos) & lista_lista_buyd0_cond[el][el2] & (
                        not (lista_lista_selld0_cond[el][el2])):

                            # count_days += 1
                            pos = True
                            lista_signal.append(1 * module_signal)
                            datas_ticker.append(datas[el2])

                        else:

                            pass

                    else:

                        if (pos):

                            lista_signal.append(None)
                            datas_ticker.append(datas[el2])

                        else:

                            pass

            if pos == True:
                lista_signal.append(-1 * module_signal)
                datas_ticker.append(datas[el2])

            df = pd.DataFrame({"signal": lista_signal, "date": datas_ticker})
            df["ticker"] = ticker
            lista_df.append(df)
            df_final = pd.concat(lista_df)

        return lista_df, df_final

    def backtest_run(self, df_params, df_params_full, tam_rank_in, tam_rank_out):

        lista_daily = []
        lista_anual = []
        lista_dailyTicker = []

        # print("wtf is going on")

        # print("generating lists")
        for type_trade in self.type_trades:
            tam_rank_in = self.tam_rank_in_b
            tam_rank_out = self.tam_rank_out_b

            # print("rout : {}".format(tam_rank_out))
            rank_in, rank_out, lista_lista_buydn_cond, lista_lista_selldn_cond, lista_lista_buyd0_cond, lista_lista_selld0_cond, lista_tickers, df_params_full, datas = self.getBacktestLists(
                df_params, df_params_full, self.col_rr, self.col_rr, tam_rank_in, tam_rank_out, type_trade=type_trade,
                least=self.least)

            self.rank_in_b = rank_in
            self.rank_in = rank_in
            self.rank_out = rank_out
            self.lista_lista_buydn_cond = lista_lista_buydn_cond
            self.lista_lista_selldn_cond = lista_lista_selldn_cond
            self.lista_lista_buyd0_cond = lista_lista_buyd0_cond
            self.lista_lista_selld0_cond = lista_lista_selld0_cond
            self.lista_tickers = lista_tickers
            self.df_params_full = df_params_full.copy()
            self.datas = datas

            # print("o tamanho do rank in: {}, o tamanho das datas: {}, conds: {}, tickers: {}".format(len(rank_in),len(datas),len(lista_lista_buydn_cond[0]),len(lista_tickers)))

            lista_df, df_final = self.generate_signal_fast(type_trade, rank_in, rank_out, lista_lista_buydn_cond,
                                                           lista_lista_selldn_cond, lista_lista_buyd0_cond,
                                                           lista_lista_selld0_cond, lista_tickers, df_params_full,
                                                           datas, rebalance_frequence=self.reb)

            # print("generating report")
            self.hedge = self.type_hedge

            df_final = df_final[df_final["signal"] != 2]

            self.df_final = df_final.copy()

            # def gen_report_beta(self,df_final,df_params_full,type_trade = 'long',notional =100000,borrow_cost = 3,postive_borrow_factor = 0,flag_hedge = True,multi_tx=1,beta_adjust=False,beta_number = 400):
            # rep,daily_rep,anual_rep = self.gen_report_fast2(df_final,df_params_full,type_trade = type_trade,notional = 100000,borrow_cost = 0,postive_borrow_factor = 0,flag_hedge = self.hedge,multi_tx = 1,vol_adjust =False)
            rep, daily_rep, anual_rep = self.gen_report_fast2(df_final, df_params_full, type_trade=type_trade,
                                                              notional=self.notional_pos, borrow_cost=0,
                                                              postive_borrow_factor=0, flag_hedge=self.hedge,
                                                              multi_tx=1, vol_adjust=False)
            # rep,daily_rep,anual_rep = self.gen_report_beta(df_final,df_params_full,type_trade = type_trade,notional = 100000,borrow_cost = 3,postive_borrow_factor = 0,flag_hedge = self.hedge,multi_tx = 1,beta_adjust = True, beta_number = self.beta_number,normalize = True)
            lista_dailyTicker.append(rep)
            lista_anual.append(anual_rep)

            return rep, daily_rep, anual_rep

        long_short_daily, long_short_anual = self.joinLongShort(self.type_trades, lista_dailyTicker, lista_daily,
                                                                lista_anual)
        return long_short_daily, long_short_anual

    def process_weight(self, pesos_rv2, min_w_allowed=0.0005):

        pesos_rv2 = [1 if el >= 1 else el for el in pesos_rv2]

        pesos_rv = pesos_rv2
        ss = sum(pesos_rv)

        if abs(ss - 1) < min_w_allowed:

            pass

        else:

            pass

            dif = 1 - ss

            # TUDO CERTO !
            if dif > 0:

                inc = dif / len([el for el in pesos_rv if abs(el) > 0])
                pesos_rv = [el + inc if abs(el) > 0 else el for el in pesos_rv]

            else:

                inc = dif / len([el for el in pesos_rv if abs(el) > 0])

                maior_que_inc = [el for el in pesos_rv if el > abs(inc)]

                # TUDO CERTO
                if len(maior_que_inc) == len([el for el in pesos_rv if abs(el) > 0]):

                    pesos_rv = [el + inc if abs(el) > 0 else el for el in pesos_rv]

                # PRECISA ITERAR
                else:

                    for i in range(100):

                        if abs(1 - ss) < 0.001:

                            break

                        else:

                            dif = 1 - ss
                            if dif > 0:

                                pesos_rv = [el + inc if abs(el) > 0 else el for el in pesos_rv]

                                break

                            else:

                                inc = dif / len([el for el in pesos_rv if abs(el) > 0])
                                maior_que_inc = [el for el in pesos_rv if el > abs(inc)]

                                if len(maior_que_inc) == len(pesos_rv):
                                    pesos_rv = [el + inc if abs(el) > 0 else el for el in pesos_rv]

                                    break

                                else:
                                    # atualizou
                                    pesos_rv = [el + inc if el in maior_que_inc else el for el in pesos_rv]
                                    ss = sum(pesos_rv)

        mins = [el for el in pesos_rv if el < min_w_allowed]
        inc = sum(mins) / len([el for el in pesos_rv if el not in mins])
        pesos_rv = [el + inc if el not in mins else 0 for el in pesos_rv]
        return pesos_rv

    def grafico_iteracoes_fundo(self, df_params_full_pure, dt_min=datetime(2016, 1, 5).date(),
                                dt_max=datetime(2022, 9, 11).date(),
                                lista_pesos_fundo=[0, 0.05, 0.075],
                                notional_inicial=100000000, if_return_df=False, ativo_variavel='Fund', title='teste',
                                x_title='x teste', y_title='y teste', imab5p_w=0.3, imab5_w=0.2, dipre_w=0.15,
                                cdi_w=0.05, rv_w=0.3, ibx_w=0.7, small11_w=0.1, divo11_w=0.1, sp500_w=0.1, fund_w=0):

        dt_min = datetime(int(dt_min.split("-")[0]), int(dt_min.split("-")[1]), int(dt_min.split("-")[2])).date()
        dt_max = datetime(int(dt_max.split("-")[0]), int(dt_max.split("-")[1]), int(dt_max.split("-")[2])).date()
        print("o dt maxxxxxxxxxxxxxx")
        print(dt_max)
        lista_comp = []
        for peso_variado in lista_pesos_fundo:

            # CADATRO DE TICKES
            nucleo_rv = ['IBX Index', 'SMAL11 BS Equity', 'SPX_REAL', 'DIVO11 BS Equity']
            nucleo_imab5_p = ['IMAB_5P']
            nucleo_imab5 = ['IMAB_5']
            nucleo_di = ['GTBRL3Y Govt']
            nucleo_cdi = ['CDI']
            nucleo_turing = ['TURING_MASTER']

            rest = 0.8
            ################################################### CONFIG ##########################################################
            notional_total = notional_inicial
            notional_total_inicial = notional_total

            if ativo_variavel == 'Fund':

                weight_rv = rv_w
                weight_imab5_p = imab5p_w
                weight_imab5 = imab5_w
                weight_di_pre = dipre_w
                weight_cdi = cdi_w
                weight_turing = peso_variado
                pesos_rv = [ibx_w, small11_w, sp500_w, divo11_w]
                pesos_rv = self.process_weight(pesos_rv)
                pesos_upper = [weight_rv, weight_imab5_p, weight_imab5, weight_di_pre, weight_cdi, weight_turing]


            elif ativo_variavel == 'IMAB5':

                weight_rv = rv_w
                weight_imab5_p = imab5p_w
                weight_imab5 = peso_variado
                weight_di_pre = dipre_w
                weight_cdi = cdi_w
                weight_turing = fund_w
                pesos_rv = [ibx_w, small11_w, sp500_w, divo11_w]
                pesos_rv = self.process_weight(pesos_rv)
                pesos_upper = [weight_rv, weight_imab5_p, weight_imab5, weight_di_pre, weight_cdi, weight_turing]


            elif ativo_variavel == 'IMAB5_P':

                weight_rv = rv_w
                weight_imab5_p = peso_variado
                weight_imab5 = imab5_w
                weight_di_pre = dipre_w
                weight_cdi = cdi_w
                weight_turing = fund_w
                pesos_rv = [ibx_w, small11_w, sp500_w, divo11_w]
                pesos_rv = self.process_weight(pesos_rv)
                pesos_upper = [weight_rv, weight_imab5_p, weight_imab5, weight_di_pre, weight_cdi, weight_turing]


            elif ativo_variavel == 'DI_PRE_3Y':

                weight_rv = rv_w
                weight_imab5_p = imab5p_w
                weight_imab5 = imab5_w
                weight_di_pre = peso_variado
                weight_cdi = cdi_w
                weight_turing = fund_w
                pesos_rv = [ibx_w, small11_w, sp500_w, divo11_w]
                pesos_rv = self.process_weight(pesos_rv)
                pesos_upper = [weight_rv, weight_imab5_p, weight_imab5, weight_di_pre, weight_cdi, weight_turing]


            elif ativo_variavel == 'CDI':

                weight_rv = rv_w
                weight_imab5_p = imab5p_w
                weight_imab5 = imab5_w
                weight_di_pre = dipre_w
                weight_cdi = peso_variado
                weight_turing = fund_w
                pesos_rv = [ibx_w, small11_w, sp500_w, divo11_w]
                pesos_rv = self.process_weight(pesos_rv)
                pesos_upper = [weight_rv, weight_imab5_p, weight_imab5, weight_di_pre, weight_cdi, weight_turing]


            elif ativo_variavel == 'RV':

                weight_rv = peso_variado
                weight_imab5_p = imab5p_w
                weight_imab5 = imab5_w
                weight_di_pre = dipre_w
                weight_cdi = cdi_w
                weight_turing = fund_w
                pesos_rv = [ibx_w, small11_w, sp500_w, divo11_w]
                pesos_rv = self.process_weight(pesos_rv)
                pesos_upper = [weight_rv, weight_imab5_p, weight_imab5, weight_di_pre, weight_cdi, weight_turing]


            elif ativo_variavel == 'IBOV':

                weight_rv = rv_w
                weight_imab5_p = imab5p_w
                weight_imab5 = imab5_w
                weight_di_pre = dipre_w
                weight_cdi = cdi_w
                weight_turing = fund_w
                pesos_rv = [peso_variado, small11_w, sp500_w, divo11_w]
                pesos_rv = self.process_weight(pesos_rv)
                pesos_upper = [weight_rv, weight_imab5_p, weight_imab5, weight_di_pre, weight_cdi, weight_turing]


            elif ativo_variavel == 'S&P':

                weight_rv = rv_w
                weight_imab5_p = imab5p_w
                weight_imab5 = imab5_w
                weight_di_pre = dipre_w
                weight_cdi = cdi_w
                weight_turing = fund_w
                pesos_rv = [ibx_w, small11_w, peso_variado, divo11_w]
                pesos_rv = self.process_weight(pesos_rv)
                pesos_upper = [weight_rv, weight_imab5_p, weight_imab5, weight_di_pre, weight_cdi, weight_turing]



            elif ativo_variavel == 'SMALL11':

                weight_rv = rv_w
                weight_imab5_p = imab5p_w
                weight_imab5 = imab5_w
                weight_di_pre = dipre_w
                weight_cdi = cdi_w
                weight_turing = fund_w
                pesos_rv = [ibx_w, peso_variado, sp500_w, divo11_w]
                pesos_rv = self.process_weight(pesos_rv)
                pesos_upper = [weight_rv, weight_imab5_p, weight_imab5, weight_di_pre, weight_cdi, weight_turing]


            elif ativo_variavel == 'DIVO11':

                weight_rv = rv_w
                weight_imab5_p = imab5p_w
                weight_imab5 = imab5_w
                weight_di_pre = dipre_w
                weight_cdi = cdi_w
                weight_turing = fund_w
                pesos_rv = [ibx_w, small11_w, sp500_w, peso_variado]
                pesos_rv = self.process_weight(pesos_rv)
                pesos_upper = [weight_rv, weight_imab5_p, weight_imab5, weight_di_pre, weight_cdi, weight_turing]


            else:

                pass

            # todos_tickers = [nucleo_rv, nucleo_imab5_p, nucleo_imab5, nucleo_di, nucleo_cdi]
            todos_tickers = [nucleo_rv, nucleo_imab5_p, nucleo_imab5, nucleo_di, nucleo_cdi, nucleo_turing]

            # PARCIAL
            todos_pesos2 = [weight_rv] + [weight_imab5_p] + [weight_imab5] + [weight_di_pre] + [weight_cdi] + [
                weight_turing]
            todos_pesos2 = self.process_weight(todos_pesos2)

            weight_rv_P = todos_pesos2[0]
            todos_pesos2 = todos_pesos2[1:]

            pesos_rv = [el * weight_rv_P for el in pesos_rv]
            todos_pesos = pesos_rv + todos_pesos2

            todos_tickers = list(chain(*todos_tickers))

            if abs(1 - sum(todos_pesos)) > 0.1:
                print("erro pesos portfolio")
                # pass


            else:
                print("portfolio: Ok")
                # pass

            todos_pesos_raw = todos_pesos
            todos_pesos = [el * notional_total for el in todos_pesos]

            ################################################### CONFIG ##########################################################
            self.col_rr = 'close'
            self.col_ranking_long = 'close'
            self.col_ranking_short = 'close'
            self.tam_rank_in_b = 250
            self.tam_rank_out_b = 250
            self.reb = 1
            self.type_hedge = False

            df_params_full_pure["carry"] = 0
            df_params_full_pure["taxa"] = 0

            # tickers = ['GD12 Curncy','GD6 Curncy']
            # tickers = ['GD12 Curncy']
            # dt_min = datetime(2014,1,5).date()
            df_params_full_pure_trade2 = df_params_full_pure[
                (df_params_full_pure["date"] >= dt_min) & (df_params_full_pure["date"] <= dt_max)]

            ############################# CONDICOES DE ZERADA
            dts = df_params_full_pure_trade2[df_params_full_pure_trade2["date"] >= dt_min][["date"]].drop_duplicates()

            tipo = 'anual'

            dts["year"] = dts["date"].apply(lambda x: x.year)
            dts["month"] = dts["date"].apply(lambda x: x.month)
            dts["day"] = dts["date"].apply(lambda x: x.day)

            if tipo == 'anual':

                zeradas = dts.groupby("year").agg({"date": "min"}).reset_index().sort_values("date")[["date"]]


            elif tipo == 'semestral':

                zeradas = dts[(dts["month"] == 12) | (dts["month"] == 6)].groupby(["year", "month"]).agg(
                    {"date": "min"}).reset_index().sort_values("date")[["date"]]

            zeradas = zeradas["date"].tolist()
            min_date = min(zeradas)
            max_date = max(zeradas)

            ############################# CONDICOES DE ZERADA
            lista_media_curta = [7]
            lista_media_longa = [21]

            lista_res = []
            lista_res_anuais = []
            prazo = 12
            lista_daily_long = []

            # for ticker in todos_tickers[0:]:
            for j in range(len(zeradas)):

                # UPDATE
                if j == 0:

                    pass

                else:

                    if abs(pl_fundo) > 0:

                        # print("o PL atual é::::::::: {}".format(pl_fundo))
                        notional_total = pl_fundo
                        todos_pesos = [round(el * notional_total) for el in todos_pesos_raw]
                        # todos_pesos2

                    else:
                        pass

                if len(zeradas) > 1:

                    data_zera_next = zeradas[j]

                    if data_zera_next == min_date:

                        df_params_full_pure_trade3 = df_params_full_pure_trade2[
                            (df_params_full_pure_trade2["date"] <= data_zera_next)]


                    elif data_zera_next == max_date:

                        # print("max dateeeeeeeeeeeeeeeeeeeeeeeee")
                        data_zera_prev = zeradas[j - 1]
                        df_params_full_pure_trade3 = df_params_full_pure_trade2[
                            (df_params_full_pure_trade2["date"] >= data_zera_prev)]

                    else:

                        data_zera_prev = zeradas[j - 1]
                        df_params_full_pure_trade3 = df_params_full_pure_trade2[
                            (df_params_full_pure_trade2["date"] > data_zera_prev) & (
                                        df_params_full_pure_trade2["date"] <= data_zera_next)]


                else:

                    df_params_full_pure_trade3 = df_params_full_pure_trade2.copy()

                i = 0
                sub_lista = []
                for ticker in todos_tickers[0:]:

                    if todos_pesos_raw[i] != 0:

                        df_params_full_pure_trade = df_params_full_pure_trade3[
                            (df_params_full_pure_trade3["ticker"] == ticker)]

                        if ticker == 'CDI':

                            df_params_full_pure_trade = df_params_full_pure_trade.sort_values("date")

                            df_params_full_pure_trade["carry_t"] = (1 + (df_params_full_pure_trade["close"] / 100)) ** (
                                        1 / 252)
                            df_params_full_pure_trade["cumprod"] = df_params_full_pure_trade["carry_t"].cumprod()
                            df_params_full_pure_trade["cumprod"] = df_params_full_pure_trade["cumprod"] / \
                                                                   df_params_full_pure_trade["cumprod"].iloc[0]
                            df_params_full_pure_trade["close"] = df_params_full_pure_trade["cumprod"]

                            df_cd = df_params_full_pure_trade.copy()

                            menos = [None] + df_params_full_pure_trade["close"].tolist()[0:-1]
                            df_params_full_pure_trade["close_d-1"] = menos
                            df_params_full_pure_trade["close_d-1"] = df_params_full_pure_trade["close_d-1"].fillna(
                                method="ffill")
                            df_params_full_pure_trade["close_d-1"] = df_params_full_pure_trade["close_d-1"].fillna(
                                method="bfill")

                            df_params_full_pure_trade = df_params_full_pure_trade.sort_values("date", ascending=False)

                            df_cd = df_params_full_pure_trade.copy()

                            df_cd["ticker"] = 'CDI - FULL'


                        else:

                            pass

                        self.notional_pos = todos_pesos[i]

                        df_params_full_pure_trade["flag_buy_long"] = True
                        df_params_full_pure_trade["flagsell_long"] = False

                        df_params_full_pure_trade["flag_buy_short"] = False
                        df_params_full_pure_trade["flagsell_short"] = True

                        param = 'teste'

                        anual_long, anual_short, daily_long, daily_short = self.train_test_backtest(
                            df_params_full_pure_trade[(df_params_full_pure_trade["date"] >= dt_min) & (
                                        df_params_full_pure_trade["ticker"] == ticker)], df_params_full_pure_trade[
                                (df_params_full_pure_trade["date"] >= dt_min) & (
                                            df_params_full_pure_trade["ticker"] == ticker)], 'teste', tipos=['long'])
                        gs = Gen_Statistics()
                        res_stats, res_anuais = gs.gen_full_stats(anual_long, anual_short, daily_long, daily_short)
                        res_stats["param"] = param
                        res_anuais["param"] = param

                        lista_res.append(res_stats)
                        lista_res_anuais.append(res_anuais)
                        lista_daily_long.append(daily_long)
                        sub_lista.append(daily_long)


                    else:

                        pass

                    i = i + 1

                # apos rodr todos os tickes :::::::::::: FAZ UPDATE DO NOTIOANL DO TOTA
                _temp = pd.concat(sub_lista)
                _temp = _temp[(_temp["date"] == _temp["date"].max()) & (pd.isnull(_temp["signal"]))]
                _temp["notional_en"] = _temp["qtty"] * _temp["px_entry"]
                pl_fundo = _temp["notional"].sum()
                # pl_fundo = 100000000

            ######################################################### GERANDO COTA DO CDI ###############################################
            df_params_full_pure_trade = df_params_full_pure_trade2[(df_params_full_pure_trade2["ticker"] == 'CDI')]

            # df_params_full_pure_trade = df_params_full_pure_trade2[df_params_full_pure_trade2["ticker"]=='CDI']
            df_params_full_pure_trade = df_params_full_pure_trade.sort_values("date")

            df_params_full_pure_trade["carry_t"] = (1 + (df_params_full_pure_trade["close"] / 100)) ** (1 / 252)
            df_params_full_pure_trade["cumprod"] = df_params_full_pure_trade["carry_t"].cumprod()
            df_params_full_pure_trade["cumprod"] = df_params_full_pure_trade["cumprod"] / \
                                                   df_params_full_pure_trade["cumprod"].iloc[0]
            df_params_full_pure_trade["close"] = df_params_full_pure_trade["cumprod"]

            # df

            df_cd = df_params_full_pure_trade.copy()

            menos = [None] + df_params_full_pure_trade["close"].tolist()[0:-1]
            df_params_full_pure_trade["close_d-1"] = menos
            df_params_full_pure_trade["close_d-1"] = df_params_full_pure_trade["close_d-1"].fillna(method="ffill")
            df_params_full_pure_trade["close_d-1"] = df_params_full_pure_trade["close_d-1"].fillna(method="bfill")

            df_params_full_pure_trade = df_params_full_pure_trade.sort_values("date", ascending=False)

            df_cd = df_params_full_pure_trade.copy()

            df_cd["ticker"] = 'CDI - FULL'

            self.notional_pos = notional_total_inicial

            df_cd["flag_buy_long"] = True
            df_cd["flagsell_long"] = False

            # factor_deviation_6
            # high betas
            df_cd["flag_buy_short"] = False
            df_cd["flagsell_short"] = True

            # param = 'MACD-J:{}/{} | prazo: {}'.format(media_curta,media_longa,prazo)
            param = 'teste'

            anual_long, anual_short, daily_long, daily_short = self.train_test_backtest(df_cd, df_cd, 'teste',
                                                                                        tipos=['long'])
            gs = Gen_Statistics()
            res_stats, res_anuais = gs.gen_full_stats(anual_long, anual_short, daily_long, daily_short)
            res_stats["param"] = param
            res_anuais["param"] = param

            lista_res.append(res_stats)
            lista_res_anuais.append(res_anuais)
            lista_daily_long.append(daily_long)

            ######################################################### GERANDO COTA DO CDI ###############################################

            ######################################################### JUNTANDO TUDO ###############################################

            # 1) FUNDO
            todos = pd.concat(lista_daily_long)
            todos = todos[todos["ticker"] != 'CDI - FULL'].groupby("date").agg({"pnl_total": "sum"}).reset_index()
            todos["pnl_acc"] = todos["pnl_total"].cumsum()
            todos["pnl_acc2"] = todos["pnl_acc"] + notional_total_inicial

            # 2) CDI
            cdi = pd.concat(lista_daily_long)
            cdi = cdi[cdi["ticker"] == 'CDI - FULL'].groupby("date").agg({"pnl_total": "sum"}).reset_index()
            cdi["pnl_acc"] = cdi["pnl_total"].cumsum()
            cdi["pnl_acc2"] = cdi["pnl_acc"] + notional_total_inicial

            cota_fundo = todos[["date", "pnl_acc2"]]
            cota_fundo.columns = ["date", "cota_fundo"]

            cota_cdi = cdi[["date", "pnl_acc2"]]
            cota_cdi.columns = ["date", "cota_cdi"]
            comp = pd.merge(cota_fundo, cota_cdi[cota_cdi["date"] >= cota_fundo["date"].min()], left_on=["date"],
                            right_on=["date"], how="outer")
            comp["(%) cdi"] = 100 * (comp["cota_fundo"] / comp["cota_cdi"])

            # comp["peso_turing_{}".format(peso_turing)]
            comp["peso_turing"] = peso_variado
            lista_comp.append(comp)
            # ploty_basic(comp,"date",["cota_fundo","cota_cdi"], title = "Cota - Retorno Nominal")
            # ploty_basic(comp,"date",["(%) cdi"], title ="Cota - (%) CDI")

        for i in range(len(lista_comp)):

            if i == 0:

                temp = lista_comp[i][["date", "cota_fundo"]]
                temp.columns = ["date",
                                "cota_fundo_{}%_{}".format(lista_comp[i]["peso_turing"].iloc[0], ativo_variavel)]

            else:

                temp = pd.merge(temp, lista_comp[i][["date", "cota_fundo"]], left_on=["date"], right_on=["date"],
                                how="inner")
                temp.rename(columns={
                    'cota_fundo': "cota_fundo_{}%_{}".format(lista_comp[i]["peso_turing"].iloc[0], ativo_variavel)},
                            inplace=True)

        if not if_return_df:

            return ploty_basic_API(temp, "date", temp.columns.tolist()[1:], title_graph, x_title=x_title,
                                   y_title=y_title, title=title)

        else:

            # return temp
            return pd.merge(temp, cota_cdi, left_on=["date"], right_on=["date"])



def ploty_basic_API(df, x_data, y_data, mode_plot='line', title=None,
                    y_title=None, x_title=None, type_plot="lines",
                    multi_yaxes=False, anotations=None, width=900, height=700,
                    not_pair=True, color_background='white', showgrid=True):
    if not isinstance(y_data, list):
        y_data = [y_data]
        x_data = [x_data]
        names = y_data.copy()

    else:
        if not isinstance(x_data, list):
            x_data = len(y_data) * [x_data]
            names = y_data.copy()
        else:
            x_data = len(y_data) * x_data
            names = y_data.copy()

    # todos os tracos iguaos
    if not isinstance(type_plot, list):
        if len(y_data) != 1:
            type_plot = len(y_data) * [type_plot]

        else:
            type_plot = [type_plot]


    else:

        if len(y_data) == 1:
            type_plot = len(y_data) * type_plot

    '''

     - Caso se deseje destacar os pontos no grafico. Recebe o dataframe de pontos que se deseja 
    destacar

     - Esta implementado apenas para 1 plot

    '''

    lista_dict = []

    # anotacoes simples
    if ((anotations is not None) & (not_pair)):

        # dicionario layout
        d1 = dict(x=4, y=4, xref='x', yref='y', text='Annotation Text 2', showarrow=True, arrowhead=7, ax=0, ay=-40)

        lista_dict = []

        vals_x = anotations[x_data[0]].tolist()
        vals_y = anotations[y_data[0]].tolist()
        for el in range(len(vals_y)):
            dd = d1.copy()
            dd["x"] = vals_x[el]
            dd["y"] = vals_y[el]
            dd["text"] = 'trades_{}'.format(el)
            lista_dict.append(dd)

    # anotacoes de trades contendo o par de compra e venda de cada trade
    elif ((anotations is not None) & (not not_pair)):

        lista_dict = []

        vals_x_buy = anotations[x_data[0] + '_buy'].tolist()
        vals_y_buy = anotations[y_data[0] + '_buy'].tolist()
        vals_x_sell = anotations[x_data[0] + '_sell'].tolist()
        vals_y_sell = anotations[y_data[0] + '_sell'].tolist()

        d1 = dict(x=4, y=4, xref='x', yref='y', text='Annotation Text 2', showarrow=True, arrowhead=7, ax=0, ay=-40,
                  arrowcolor='#636363')

        for el in range(len(vals_y_sell)):
            dd = d1.copy()
            dd2 = d1.copy()
            dd["x"] = vals_x_buy[el]
            dd["y"] = vals_y_buy[el]
            dd["text"] = 'trades_buy_{}'.format(el)
            dd["arrowcolor"] = '#636363'
            lista_dict.append(dd)

            dd2["x"] = vals_x_sell[el]
            dd2["y"] = vals_y_sell[el]
            dd2["text"] = 'trades_sell_{}'.format(el)
            dd2["arrowcolor"] = '#d9f441'
            lista_dict.append(dd2)

    data = []
    ## criamos uma lista de traces
    count = 1
    for el in range(len(x_data)):
        if mode_plot == 'line':
            trace = go.Scatter(
                x=df['{}'.format(x_data[el])],
                y=df['{}'.format(y_data[el])],
                name=names[el],
                mode=type_plot[el],
                yaxis='y{}'.format(count)
            )

        elif mode_plot == 'bar':
            trace = go.Bar(
                x=df['{}'.format(x_data[el])],
                y=df['{}'.format(y_data[el])],
                name=names[el],
                opacity=0.8)

        else:
            print("tipo invalido de Modo de plot")

        if multi_yaxes:
            count += 1

        data.append(trace)

    if not multi_yaxes:

        layout = dict(
            width=width,
            height=height,
            title='{}'.format(title),
            yaxis=dict(title='{}'.format(y_title), showgrid=True, gridcolor='#bdbdbd'),
            xaxis=dict(title='{}'.format(x_title), showgrid=True, gridcolor='#bdbdbd'),
            annotations=lista_dict,
            # showgrid = showgrid,
            plot_bgcolor=color_background
        )

    else:

        layout = go.Layout(
            width=width,
            height=height,
            title=title,
            yaxis=dict(
                title='yaxis title',
                showgrid=True, gridcolor='#bdbdbd'
            ),
            yaxis2=dict(
                showgrid=True, gridcolor='#bdbdbd',
                title='yaxis2 title',
                titlefont=dict(
                    color='rgb(148, 103, 189)'
                ),
                tickfont=dict(
                    color='rgb(148, 103, 189)'
                ),
                overlaying='y',
                side='right'
            ),
            annotations=lista_dict,
            plot_bgcolor=color_background
        )

    # ata = [trace]

    fig = dict(data=data, layout=layout)
    # py.iplot(fig, filename = "-")

    return fig


lv = LowVol(index='IBX', country='brazil', type_trades=["long", "short"], flag_signal=False, local_database=True,
            dict_param='gss', nbin=7, backtest_di=True)

df_params_pure, df_params_full_pure, tam_rank_in, tam_rank_out = lv.get_data()
lv.df_params_full_pure = df_params_full_pure.copy()

print("ole ole")
print(df_params_pure.head())

df = lv.grafico_iteracoes_fundo(lv.df_params_full_pure,
                                lista_pesos_fundo =[0, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25][0:(0 + 2)],
                                if_return_df = True, ativo_variavel='Fund', dt_min='2022-01-01', dt_max='2022-09-09',
                                imab5p_w=0.3, imab5_w=0.2, dipre_w=0.15, cdi_w=0.05, rv_w=0.3,
                                ibx_w=0.7, small11_w=0.1, divo11_w=0.1, sp500_w=0.1, fund_w=0)

print("transformar em json 222")

print(df)

print("testando jsonnify")
print(pd.DataFrame({"a": [1, 2, 34]}).to_json(date_format='iso', orient='split'))

df_json = df.to_json(date_format='iso', orient='split')

#print("o df em json eh righ after")

# time.sleep(2)

#print("o df em json eh")
#print(df_json)


#print(" veio tentar ler lendo primeiro 22")

dff = pd.read_json(df_json, orient='split')

#print(" veio tentar ler lendo primeiro 33")

#print(dff)

dff = generae_statistcs_interface(dff.drop("cota_cdi", axis=1), dff[["date", "cota_cdi"]], ret_dd=True)


#print(" veio tentar ler lendo primeiro 4444444444444444444444")

#print(dff)

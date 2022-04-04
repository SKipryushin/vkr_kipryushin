import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, Normalizer


def input_columns() -> list:
    columns = [
        "Плотность, кг/м3",
        "модуль упругости, ГПа",
        "Количество отвердителя, м.%",
        "Содержание эпоксидных групп,%_2",
        "Температура вспышки, С_2",
        "Поверхностная плотность, г/м2",
        "Потребление смолы, г/м2",
        "Угол нашивки, град",
        "Шаг нашивки",
        "Плотность нашивки"
    ]
    return columns


def preprocess_df(df: pd.DataFrame, preprocessor: callable) -> pd.DataFrame:
    res = preprocessor.transform(df)
    preprocessed_df = pd.DataFrame(res, columns=df.columns)
    return preprocessed_df


def create_dataframe_from_formdata(form_data: dict, mapper: dict) -> pd.DataFrame:
    temp = dict()
    for i_key, i_value in form_data.items():
        if mapper.get(i_key):
            temp[mapper[i_key]] = [i_value]
    return pd.DataFrame(data=temp)


def convert_formdata(form_data: dict, converter: callable) -> dict:
    return {i_key: converter(i_value) for i_key, i_value in form_data.items()}

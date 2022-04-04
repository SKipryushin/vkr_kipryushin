from predictor import GradientBoostingPredictor
import joblib

TARGET_VAR = 'Прочность при растяжении, МПа'

INPUTS_MAPPER = {
    "density": "Плотность, кг/м3",
    "elastic_modulus": "модуль упругости, ГПа",
    "hardener_amount": "Количество отвердителя, м.%",
    "epoxy_groups": "Содержание эпоксидных групп,%_2",
    "flashpoint_temp": "Температура вспышки, С_2",
    "surface_density": "Поверхностная плотность, г/м2",
    "resin_cons": "Потребление смолы, г/м2",
    "patching_angle": "Угол нашивки, град",
    "patching_step": "Шаг нашивки",
    "patching_density": "Плотность нашивки",
}



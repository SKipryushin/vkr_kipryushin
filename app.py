from flask import Flask, request, render_template
from config import TARGET_VAR, INPUTS_MAPPER
from predictor import GradientBoostingPredictor
from utils import convert_formdata, create_dataframe_from_formdata, preprocess_df

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def predict_result():
    prediction_results = 'Расчёт не запущен'
    error = dict()
    if request.method == 'POST':
        form_data = request.form
        try:
            float_data = convert_formdata(form_data=form_data, converter=converter)
        except ValueError:
            for i_key, i_value in form_data.items():
                try:
                    converter(i_value)
                except ValueError:
                    error['name'] = INPUTS_MAPPER[i_key]
                    error['value'] = i_value
                    break
        if not error:
            df = create_dataframe_from_formdata(form_data=float_data, mapper=INPUTS_MAPPER)
            normalized_df = preprocess_df(df, preprocessor=predictor.preprocessor)
            prediction_results = predictor.model.predict(normalized_df)[0]
    return render_template('index.html', result=prediction_results, target=TARGET_VAR, error=error)


if __name__ == '__main__':
    predictor = GradientBoostingPredictor()
    converter = lambda value: float(value)
    app.run(host='localhost', port=8080, debug=True)

import joblib


class GradientBoostingPredictor:

    def __init__(self, model_path='./models/grad_booster_model.bin', preprocessor_path='./models/fitted_normalizer.bin'):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self._preprocessor = None
        self._model = None

    def load_model(self) -> None:
        self._model = joblib.load(self.model_path)

    def load_preprocessor(self) -> None:
        self._preprocessor = joblib.load(self.preprocessor_path)

    @property
    def model(self):
        if self._model is None:
            self.load_model()
        return self._model

    @property
    def preprocessor(self):
        if self._preprocessor is None:
            self.load_preprocessor()
        return self._preprocessor


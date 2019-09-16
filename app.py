import pickle

from flask import Flask, request
from flask_restful import Resource, Api
from hierarchical_hate_speech_abusive import HierarchicalHateSpeechAbusive


hmc = pickle.load(open('hmc_model.pkl', 'rb'))


class Prediction(Resource):
    def post(self):
        data = request.get_json()
        text = data['text']
        prediction = hmc.predict(text)
        return prediction


app = Flask(__name__)
api = Api(app)

api.add_resource(Prediction, '/')

if __name__ == '__main__':
    app.run(debug=True)

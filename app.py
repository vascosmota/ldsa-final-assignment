# the request object does exactly what the name suggests: holds
# all of the contents of an HTTP request that someone is making
# the Flask object is for creating an HTTP server - you'll
# see this a few lines down.
# the jsonify function is useful for when we want to return
# json from the function we are using.
from flask import Flask, request, jsonify
from peewee import (
    SqliteDatabase, PostgresqlDatabase, Model, IntegerField,
    FloatField, BooleanField, TextField,
)
import pickle, json, logging
import pandas as pd
from aux_functions import pipeline_2
import datetime
# here we use the Flask constructor to create a new
# application that we can add routes to
app = Flask(__name__)

# create logger with 'spam_application'
logger = logging.getLogger('predict_unemployment')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('predict.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

DB = SqliteDatabase('predictions.db')

class Prediction(Model):
    observation_id = IntegerField(unique=True)
    observation = TextField()
    proba = FloatField()
    true_class = IntegerField(null=True)

    class Meta:
        database = DB

DB.create_tables([Prediction], safe=True)
                                          

with open('columns.json') as fh:
    columns = json.load(fh)

with open('pipeline.pickle', 'rb') as fh:
    pipeline = pickle.load(fh)

with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)
    
def convert_input(req):
    payload = request.get_json()
    logger.debug("Payload")
    logger.debug(payload)
    observation = payload['observation']
    observation['id'] = payload['id']
    if 'target' in observation:
        observation.pop('target')
    logger.debug("Observation")
    logger.debug(observation)
    return observation
    


@app.route('/predict', methods=['POST'])
def predict():
    logger.info("=========== NEW PREDICTION =============")
    observation = convert_input(request)
    logger.info("New observation")
    logger.info(observation)
    obs = pd.DataFrame([observation], columns=columns).astype(dtypes)
    logger.debug("Running pipeline")
    proba = pipeline.predict(obs).tolist()[0]
    logger.debug("Value predicted: %s", proba)
    p = Prediction(
        observation_id=observation['id'],
        proba=proba,
        observation=observation,
    )
    p.save()
    return jsonify({
        'prediction': proba
    })


if __name__ == "__main__":
    app.run(debug=True)

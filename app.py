# the request object does exactly what the name suggests: holds
# all of the contents of an HTTP request that someone is making
# the Flask object is for creating an HTTP server - you'll
# see this a few lines down.
# the jsonify function is useful for when we want to return
# json from the function we are using.
from flask import Flask, request, jsonify
import pickle, json
from aux_functions import pipeline_2
# here we use the Flask constructor to create a new
# application that we can add routes to
app = Flask(__name__)

with open('columns.json') as fh:
    columns = json.load(fh)

with open('pipeline.pickle', 'rb') as fh:
    pipeline = pickle.load(fh)

with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)


@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json()
    obs = pd.DataFrame([payload], columns=columns).astype(dtypes)
    proba = pipeline.predict(obs)
    return jsonify({
        'prediction': proba
    })


if __name__ == "__main__":
    app.run(debug=True)

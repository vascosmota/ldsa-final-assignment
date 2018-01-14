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


    
def pipeline_2(df_in, **kwargs):
    df = copy.deepcopy(df_in)
    df = rename_columns(df)
    df = drop_variables(df, ['id', 'gender', 'earned_dividends', 'country_of_origin', 'ethnicity'])
    
    # Get age
    df['age'] = years_old(df)    
    df = drop_variables(df, ['birth_date'])
    
    # Convert school level in years
    df['school_years'] = df.school_level.apply(convert_school_level) 
    df = drop_variables(df, ['school_level'])
    
    # Group spouses and extract relevant domestic relationship types
    df['domestic_relationship_type'] = df.domestic_relationship_type.apply(group_spouses)
    df['domestic_relationship_type'] = df.domestic_relationship_type.apply(extract_relevant_domestic_relationship_types)
    df = get_dummies(df, 'domestic_relationship_type', ['has spouse', 'living with child', 'never married', 'not living with family', 'other'])
    # df = pd.get_dummies(df, prefix='domestic_relationship_type', columns=['domestic_relationship_type'], prefix_sep='.')  
    
    # Extract relevant professions
    df['profession'] = df.profession.apply(extract_relevant_professions)
    df = get_dummies(df, 'profession', ['C-level', 'specialist technician', 'other'])
    # df = pd.get_dummies(df, prefix='profession', columns=['profession'], prefix_sep='.')  
    
    # Extrct relevant job types
    df['job_type'] = df.job_type.apply(extract_relevant_job_types)
    df = get_dummies(df, 'job_type', ['self-emp-inc', 'self-emp-not-inc', 'other'])
    # df = pd.get_dummies(df, prefix='job_type', columns=['job_type'], prefix_sep='.')      
        
    # Extract relevant domestic status
    df['domestic_status'] = df.domestic_status.apply(extract_relevant_domestic_status)
    df = get_dummies(df, 'domestic_status', ['married 2', 'single', 'd', 'divorce pending', 'other'])
    # df = pd.get_dummies(df, prefix='domestic_status', columns=['domestic_status'], prefix_sep='.')      
     
    return df                                          

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

def rename_columns(df):    
    names_mapping = {}
    for col in df.columns:
        names_mapping[col] = col.replace(' ', '_')
    return df.rename_axis(names_mapping, axis='columns')
    
    
school_level_mapping = {
    'kindergarten': 0,
    'primary 1 through 4': 2,
    'primary school': 4,
    'secondary-5 through 6':6,
    'secondary-7 through 8': 8,    
    'secondary-9': 9,
    '10th': 10,
    'secondary 11': 11,
    'secondary': 12,
    'secondary 12': 12,
    'basic vocational': 14,
    'entry level college': 14,
    'advanced vocational': 15,
    'college graduate': 16,
    'some post graduate': 18,
    'advanced post graduate': 20,  
}


def extract_relevant_professions(profession):
    return profession if profession in ('C-level', 'specialist technician') else 'other'


def extract_relevant_job_types(job_type):
    return job_type if job_type in ('self-emp-inc', 'self-emp-not-inc') else 'other'


def extract_relevant_domestic_status(domestic_status):
    return domestic_status if domestic_status in ('married 2', 'single', 'd', 'divorce pending') else 'other'


def extract_relevant_domestic_relationship_types(domestic_relationship_type):
    return domestic_relationship_type if domestic_relationship_type in ('has spouse', 'living with child', 'never married', 'not living with family') else 'other'


def drop_variables(df_input, variables_to_drop):
    df = copy.deepcopy(df_input)
    for variable in variables_to_drop:
        df = df.drop(variable, axis=1)
    return df


def group_spouses(record):
    return ("has spouse" if record in ('has husband', 'has wife') else record)


def years_old(df):
    return (datetime.datetime.now().date() - pd.to_datetime(df.birth_date)).dt.days/365


def convert_school_level(val):
    return school_level_mapping[val]


def binarize_interest_earned(df):
    return df.interest_earned >0


def is_immigrant(df):
    return df.country_of_origin == 'u.s.'


def group_job_types(val):
    return job_type_map[val]


def is_white(df):
    return df.ethnicity == 'white and privileged'


def is_currently_single(df):
    return (
        (df.domestic_status ==  'single') | 
        (df.domestic_status ==   'd') | 
        (df.domestic_status ==   'spouse passed')
    )
    
def get_dummies(df, feature, relevant_labels):
    for l in relevant_labels:
        df[feature + l] = (df[feature] == l)
    df = df.drop(feature, axis=1)
    return df


if __name__ == "__main__":
    app.run(debug=True)

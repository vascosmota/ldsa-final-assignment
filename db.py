from peewee import (
    SqliteDatabase, PostgresqlDatabase, Model, IntegerField,
    FloatField, BooleanField, TextField,
)

DB = SqliteDatabase('predictions.db')

class Prediction(Model):
    observation_id = IntegerField(unique=True)
    observation = TextField()
    proba = FloatField()
    true_class = IntegerField(null=True)

    class Meta:
        database = DB

DB.create_tables([Prediction], safe=True)

from mongoengine import (
    Document, StringField, DateTimeField, ListField, DictField, IntField
)


class PredictionResults(Document):
    request_id = IntField(required=True)
    file_id = StringField(required=True)
    file_name = StringField(required=True)
    saved_to = StringField(required=True)
    datetime = DateTimeField(required=True)
    defects = ListField(DictField(required=False))

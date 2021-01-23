from flask import url_for

from flaskr import db
from flaskr.auth.models import User
from flaskr.mlModels.modelType import modelType
from enum import Enum

class ColumnName(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    selected = db.Column(db.Boolean, nullable=False)
    name = db.Column(db.String(120), nullable=False)
    owner_id = db.Column(db.Integer, db.ForeignKey('ml_model.id'),
        nullable=True)


class mlModel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    author_id = db.Column(db.ForeignKey(User.id), nullable=False)
    created = db.Column(
        db.DateTime, nullable=False, server_default=db.func.current_timestamp()
    )
    title = db.Column(db.String, nullable=False)
    body = db.Column(db.String, nullable=False)
    data = db.Column(db.Binary, nullable=True)
    model_type = db.Column(db.Enum(modelType))
    trained_model = db.Column(db.Binary, nullable=True)

    # User object backed by author_id
    # lazy="joined" means the user is returned with the post in one query
    author = db.relationship(User, lazy="joined", backref="ml_model")

    column_names = db.relationship(ColumnName, backref='ml_model', lazy=True)

    unknown_atribute = db.Column(db.String, nullable=True)

    @property
    def update_url(self):
        return url_for("mlModels.update", id=self.id)

    @property
    def delete_url(self):
        return url_for("mlModels.delete", id=self.id)

    @property
    def predict_url(self):
        return url_for("mlModels.predict", id=self.id)

    @property
    def params_url(self):
        return url_for("mlModels.params", id=self.id)

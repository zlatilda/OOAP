import os

import click
from flask import Flask
from flask.cli import with_appcontext
from flask_sqlalchemy import SQLAlchemy
import flask_monitoringdashboard as dashboard

__version__ = (1, 0, 0, "dev")

alchemy_database = SQLAlchemy()

def create_app(test_config=None):
    """Create and configure an instance of the Flask application."""
    app = Flask(__name__, instance_relative_config=True)
    dashboard.config.init_from(file='./../config.cfg')
    dashboard.bind(app)

    # some deploy systems set the database url in the environ
    db_url = os.environ.get("DATABASE_URL")

    if db_url is None:
        # default to a sqlite database in the instance folder
        db_url = "sqlite:///" + os.path.join(app.instance_path, "flaskr.sqlite")
        # ensure the instance folder exists
        os.makedirs(app.instance_path, exist_ok=True)
    ALLOWED_EXTENSIONS = {'csv', 'xlsx'}
    app.config.from_mapping(
        # default secret that should be overridden in environ or config
        SECRET_KEY=os.environ.get("SECRET_KEY", "dev"),
        SQLALCHEMY_DATABASE_URI=db_url,
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        SQLALCHEMY_ECHO=True,
        UPLOAD_FOLDER = "/uploads",
        ALLOWED_EXTENSIONS = ALLOWED_EXTENSIONS
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile("config.py", silent=True)
    else:
        # load the test config if passed in
        app.config.update(test_config)

    # initialize Flask-SQLAlchemy and the init-db command
    alchemy_database.init_app(app)
    app.cli.add_command(init_db_command)

    # apply the blueprints to the app
    from flaskr import auth, mlModels

    app.register_blueprint(auth.bp)
    app.register_blueprint(mlModels.bp)

    # make "index" point at "/", which is handled by "mlModels.index"
    app.add_url_rule("/", endpoint="index")

    return app


def init_db():
    alchemy_database.drop_all()
    alchemy_database.create_all()


@click.command("init-db")
@with_appcontext
def init_db_command():
    """Clear existing data and create new tables."""
    init_db()
    click.echo("Initialized the database.")

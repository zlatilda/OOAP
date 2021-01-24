from flask import Blueprint
from flask import flash
from flask import g
from flask import redirect
from flask import render_template
from flask import request
from flask import url_for
from flask import current_app
from flask import send_file
from flask import send_from_directory

from werkzeug.exceptions import abort
import matplotlib.pyplot as plt
from flaskr import db
import flaskr.fileReader.converter as converter
from flaskr.auth.views import login_required
from flaskr.mlModels.models import mlModel
from flaskr.mlModels.models import ColumnName
from flaskr.mlModels.modelType import modelType
import flaskr.mlModels.namesDictionary as namesDict
import flaskr.mlModels.modelUtils as modelUtils
import pickle
import click
import numpy as np
import pandas as pd
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from six import StringIO
from sklearn import metrics
from IPython.display import Image
import seaborn as sn
import pydotplus

bp = Blueprint("mlModels", __name__)

def isImage(file) :
    filename = file.filename
    return filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png")

def writeBinaryToFile(data, format):
    fileWithModelWrite = open("../data" + format, 'wb')
    write_file(data, fileWithModelWrite)
    return "../data" + format

def readDataFrame(file):
    return pd.read_csv(file, header=0)

def toStringList(columnNamesList):
    return list(map(lambda column_name: column_name.name,
        list(filter(lambda column_name: column_name.selected == True, columnNamesList))))

def toColumnNames(id, names):
    click.echo(id)
    click.echo(names)
    l = []
    for name in names:
        columnName = ColumnName(owner_id=id, name=name, selected=name!=names[-1])
        l.append(columnName)
        db.session.add(columnName)
    return l

@bp.route("/")
def index():
    """Show all the posts, most recent first."""
    models = mlModel.query.order_by(mlModel.created.desc()).all()
    return render_template("mlModels/index.html", models=models)


def get_model(id, check_author=True):
    """Get a ml-model and its author by id.
    Checks that the id exists and optionally that the current user is
    the author.
    :param id: id of post to get
    :param check_author: require the current user to be the author
    :return: the post with author information
    :raise 404: if a post with the given id doesn't exist
    :raise 403: if the current user isn't the author
    """
    model = mlModel.query.get_or_404(id, f"Model with id {id} doesn't exist.")

    if check_author and model.author != g.user:
        abort(403)

    return model


@bp.route("/create", methods=("GET", "POST"))
@login_required
def create():
    """Create a new mddel for the current user."""
    if request.method == "POST":
        title = request.form["title"]
        body = request.form["body"]
        trainedModel, column_names, unknown_atribute, file = upload_file(request, True)
        error = None
        if not title:
            error = "Title is required."

        if error is not None:
            flash(error)
        else:
            db.session.add(mlModel(title=title, body=body, author=g.user, data=file, trained_model=trainedModel
                ,column_names=toColumnNames(names=column_names, id=None), unknown_atribute=unknown_atribute, model_type
                = modelType.DECISION_TREE_GINI_CLASSIFIER))
            db.session.commit()
            return redirect(url_for("mlModels.index"))

    return render_template("mlModels/create.html")


@bp.route("/<int:id>/predict", methods=("GET", "POST"))
@login_required
def predict(id):

    model = get_model(id)
    if request.method == "POST":
        file = upload_file(request, False)
        makePredictions(model, file)
        return send_file("/home/danylo/prediction.csv", as_attachment = True)

    return render_template("mlModels/predict.html", model=model)

@bp.route("/<int:id>/change_params", methods=("GET", "POST"))
@login_required
def params(id):

    model = get_model(id)
    if request.method == "POST":
        model.unknown_atribute = request.form.get('secret-attribute')
        for col_name in model.column_names:
            present = len(request.form.getlist('model-attributes')) == 0 and col_name.name != "label" and col_name.name != "Unnamed: 0"
            for column_name in request.form.getlist('model-attributes'):
                if col_name.name == column_name:
                    present = True
            col_name.selected = present
        path = writeBinaryToFile(model.data, '.csv')
        model.model_type = namesDict.types_dictionary[request.form.get('model-type')]
        model.trained_model = updateTrainedModel(toStringList(model.column_names), [model.unknown_atribute],
            readDataFrame(open(path, "rb")), model.model_type)
        db.session.commit()
        return redirect(url_for("mlModels.index"))
    return render_template("mlModels/params.html", model=model, types=namesDict.types_dictionary.keys())

@bp.route("/<int:id>/update", methods=("GET", "POST"))
@login_required
def update(id):
    """Update a model if the current user is the author."""
    model = get_model(id)

    if request.method == "POST":
        title = request.form["title"]
        body = request.form["body"]
        trainedModel, column_names, unknown_atribute, file = upload_file(request, True)
        error = None

        if not title:
            error = "Title is required."

        if error is not None:
            flash(error)
        else:
            model.title = title
            model.body = body
            model.trained_model = trainedModel
            model.unknown_atribute = unknown_atribute
            model.file = file
            model.column_names = toColumnNames(id, column_names)
            model.model_type = modelType.DECISION_TREE_GINI_CLASSIFIER
            db.session.commit()
            return redirect(url_for("mlModels.index"))

    return render_template("mlModels/update.html", model=model)


@bp.route("/<int:id>/delete", methods=("POST",))
@login_required
def delete(id):
    """Delete a machine learning model.
    Ensures that the machine learning model exists and that the logged in user is the
    author of the machine learning model.
    """
    model = get_model(id)
    db.session.delete(model)
    db.session.commit()
    return redirect(url_for("mlModels.index"))

def allowed_file(filename):
    click.echo(filename)
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

def upload_file(request, trainModel):
    if request.method == 'POST':
        click.echo('request')
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            binaryData = convertToBinaryData(file)
            if trainModel == False:
                return file
            data = readData(file)
            return data[0], data[1], data[2], binaryData
        else:
            click.echo('Not allowed file name')
            flash('Not allowed file name')
            return redirect(request.url)

def write_file(data, file):
    # Convert binary data to proper format and write it on Hard Disk
    file.write(data)

def convertToBinaryData(file):
    #Convert digital data to binary format
    blobData = file.read()
    file.seek(0)
    return blobData

def readData(file):
    pima = readDataFrame(file)
    feature_cols = pima.columns[:-1]
    toPredict = pima.columns[-1:]
    return None, list(pima.columns), toPredict[0]



def makePredictions(model, file):
    fileWithModelWrite = open("../model", 'wb')
    write_file(model.trained_model, fileWithModelWrite)
    fileWithModelRead = open("../model", "rb")
    trainedModel = pickle.load(fileWithModelRead)
    if isImage(file) :
        pima = converter.convert(file)
    else:
        pima = pd.read_csv(file, header=0)
    toPredict = model.unknown_atribute
    feature_cols = toStringList(model.column_names)
    X = pima[feature_cols]
    data = trainedModel.predict(X)
    y_pred = pd.DataFrame(data=data,index=None,columns=[toPredict])
    result = y_pred.join(X, how="outer")
    result.to_csv("/home/danylo/prediction.csv", index=False)
    return open("/home/danylo/prediction.csv", "rb")

def updateTrainedModel(feature_cols, toPredict, pima, type_of_model):
    replace_map = {'day' : {'mon':1,'tue':2,'wed':3,'thu':4,'fri':5,'sat':6,'sun':7},
    'month':{'dec':1,'jan':2,'feb':3,'mar':4,'apr':5,'may':6,'jun':7,'jul':8,'aug':9,'sep':10,'oct':11,'nov':12}}
    pima.replace(replace_map, inplace=True)
    X = pima[feature_cols] # Features
    y = pima[toPredict]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    clf, isRegressor = modelUtils.getModel(type_of_model)
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    importances_rf = pd.Series(clf.feature_importances_, index = X.columns)
    sorted_importances_rf = importances_rf.sort_values()
    sorted_importances_rf.plot(kind = 'barh', color = 'lightgreen')
    #dot_data = StringIO()
    #export_graphviz(clf, out_file=dot_data,
    #                filled=True, rounded=True,
    #                special_characters=True)
    #graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    #with open("../dt.png", "wb") as file:
    #    file.write(graph.create_png())
    #sn.jointplot(x="alcohol", y="quality", data=pima, kind="kde");
    #cm = metrics.confusion_matrix(y_test.to_numpy().flatten(), y_pred, [i for i in range(10)])
    #df_cm = pd.DataFrame(cm, [i for i in range(10)], [i for i in range(10)])
    #sn.heatmap(df_cm, annot=True, linewidth=.5, cmap="YlGnBu")
    plt.savefig('importances.png')
    click.echo("---------------")
    click.echo("Accuracy score:")
    if isRegressor:
        click.echo(metrics.mean_absolute_error(y_test.to_numpy().flatten(), y_pred))
    else:
        click.echo(metrics.accuracy_score(y_test.to_numpy().flatten(), y_pred))
    click.echo("---------------")
    fileWithModelWrite = open("../model", "wb")
    pickle.dump(clf, fileWithModelWrite)
    fileWithModelRead = open("../model", "rb")
    return fileWithModelRead.read()

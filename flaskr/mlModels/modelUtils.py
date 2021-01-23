from flaskr.mlModels.modelType import modelType
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor

def getModel(type_of_model):
    if type_of_model == modelType.DECISION_TREE_GINI_CLASSIFIER:
        return DecisionTreeClassifier(max_depth=5), False
    if type_of_model == modelType.DECISION_TREE_ENTROPHY_CLASSIFIER:
        return DecisionTreeClassifier(max_depth=5, criterion="entropy"), False
    elif type_of_model == modelType.RANDOM_FOREST_CLASSIFIER:
        return RandomForestClassifier(n_estimators = 100,
            random_state = 2), False
    elif type_of_model == modelType.RANDOM_FOREST_REGRESSOR:
        return RandomForestRegressor(n_estimators = 100,
            random_state = 2), True
    elif type_of_model == modelType.DECISION_TREE_GINI_REGRESSOR:
        return DecisionTreeRegressor(criterion="mae"), True
    elif type_of_model == modelType.GRADIENT_BOOSTING_REGRESSOR:
        return GradientBoostingRegressor(max_depth = 5,
            #subsample = 0.9, # set the fraction of training set to use in each tree
            #max_features = 0.75,
            n_estimators = 200,
            random_state = 2), True
    elif type_of_model == modelType.GRADIENT_BOOSTING_CLASSIFIER:
        return GradientBoostingClassifier(max_depth = 5,
            subsample = 0.9, # set the fraction of training set to use in each tree
            max_features = 0.75,
            n_estimators = 200,
            random_state = 2), False
    elif type_of_model == modelType.ADA_BOOST_REGRESSOR:
        dt = DecisionTreeRegressor(max_depth = 2, random_state=1)
        return AdaBoostRegressor(base_estimator = dt, n_estimators = 180, random_state = 1), True
    elif type_of_model == modelType.ADA_BOOST_CLASSIFIER:
        dt = DecisionTreeClassifier(max_depth = 2, random_state=1)
        return AdaBoostClassifier(base_estimator = dt, n_estimators = 180, random_state = 1), False

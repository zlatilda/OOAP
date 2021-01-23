from flaskr.mlModels.modelType import modelType

types_dictionary = {'Дерево рішень з критерієм Джині (класифікація)' :
    modelType.DECISION_TREE_GINI_CLASSIFIER,
    'Дерево рішень (ентропія, класифікація)':
    modelType.DECISION_TREE_ENTROPHY_CLASSIFIER,
    'Випадковий ліс (класифікація)':
    modelType.RANDOM_FOREST_CLASSIFIER,
    'Випадковий ліс (регресія)':
    modelType.RANDOM_FOREST_REGRESSOR,
    'Дерево рішень з критерієм Джині (регресія)' :
    modelType.DECISION_TREE_GINI_REGRESSOR,
    'Ентропічне дерево рішень (регресія)':
    modelType.DECISION_TREE_ENTROPHY_REGRESSOR,
    'Адаптивний бустинг (регресія)':
    modelType.ADA_BOOST_REGRESSOR,
    'Адаптивний бустинг (класифікація)':
    modelType.ADA_BOOST_CLASSIFIER,
    'Градієнтний бустинг (регресія)':
    modelType.GRADIENT_BOOSTING_REGRESSOR,
    'Градієнтний бустинг (класифікація)':
    modelType.GRADIENT_BOOSTING_CLASSIFIER}

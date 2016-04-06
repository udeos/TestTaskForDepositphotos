import re
import numpy as np
import pandas as pd
from scipy import sparse

from sklearn import ensemble, linear_model, cross_validation, preprocessing


def handle_recipe(recipe):
    chars_to_remove = ''.join(['.', ',', '(', ')', '&', '%', '"', '\\', '\'', '/'])
    item_ingredients = set()
    for ingredient in recipe:
        ingredient = ingredient.lower()
        ingredient = re.sub(u'(?u)[' + re.escape(chars_to_remove) + ']', '', ingredient)
        ingredient = ingredient.split(' ')
        ingredient = [term for term in ingredient if not term.isdigit()]
        item_ingredients.update(ingredient)
    return list(item_ingredients)


def build_ingredients_matrix(recipes, n_features):
    indptr = [0]
    indices = []
    for recipe in recipes:
        indices.extend(recipe)
        indptr.append(len(indices))
    return sparse.csr_matrix(([1] * len(indices), indices, indptr), dtype=np.bool,
                             shape=(len(recipes), n_features))


def handle_test_recipes(recipe, allowed_ingredients):
    recipe = handle_recipe(recipe)
    recipe = list(allowed_ingredients.intersection(recipe))
    return recipe


if __name__ == '__main__':
    # load and handle train raw data
    train_raw_data = pd.read_json('train.json')
    train_raw_data.drop('id', axis=1, inplace=True)
    train_raw_data['ingredients'] = train_raw_data['ingredients'].apply(handle_recipe)

    # find unique cuisines and ingredients
    known_cuisines = set(train_raw_data['cuisine'])
    known_ingredients = set()
    for ingredients in train_raw_data['ingredients']:
        known_ingredients.update(ingredients)

    # encode ingredients and cuisines
    le_ingredients = preprocessing.LabelEncoder()
    le_ingredients.fit(list(known_ingredients))
    le_cuisines = preprocessing.LabelEncoder()
    le_cuisines.fit(list(known_cuisines))
    train_raw_data['ingredients'] = train_raw_data['ingredients'].apply(le_ingredients.transform)
    train_raw_data['cuisine'] = train_raw_data['cuisine'].apply(le_cuisines.transform)

    train_dataset = build_ingredients_matrix(train_raw_data['ingredients'],
                                             n_features=len(known_ingredients))
    train_dataset_classes = train_raw_data['cuisine'].as_matrix()

    # load and handle test raw data
    test_raw_data = pd.read_json('test.json')
    test_raw_data['ingredients'] = test_raw_data['ingredients'].apply(handle_test_recipes,
                                                                      args=(known_ingredients,))
    test_raw_data['ingredients'] = test_raw_data['ingredients'].apply(le_ingredients.transform)
    test_dataset = build_ingredients_matrix(test_raw_data['ingredients'],
                                            n_features=len(known_ingredients))

    # train model and predict
    clf = ensemble.BaggingClassifier(linear_model.Perceptron(), n_estimators=15)
    print cross_validation.cross_val_score(clf, train_dataset, train_dataset_classes, cv=10, n_jobs=-1)
    # >>> [0.76813049 0.77811245 0.7873996 0.78196433 0.78417085 0.77319588 0.7705083 0.77920443 0.77405542 0.78699269]
    clf.fit(train_dataset, train_dataset_classes)
    prediction = clf.predict(test_dataset)

    # build submission
    ids = test_raw_data['id'].as_matrix()
    submission = np.concatenate((ids.reshape((ids.shape[0], 1)), prediction.reshape((prediction.shape[0], 1))), axis=1)
    submission = pd.DataFrame(submission, columns=['id', 'cuisine'])
    submission['cuisine'] = submission['cuisine'].apply(le_cuisines.inverse_transform)
    submission.to_csv('submission.csv', index=False)

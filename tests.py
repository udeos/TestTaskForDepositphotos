import pandas as pd
import json

from unittest import TestCase

from whats_cooking_prototype import handle_recipe, handle_test_recipes, build_ingredients_matrix


raw_data_mock = [
    {
        "id": 18009,
        "ingredients": [
            "Baking Powder 55",
            ",55 .(\"eggs\")&%\\\'/",
            "\"white\" sugar"
        ]
    }
]


class WhatsCookingPrototypeTests(TestCase):

    def test_handle_recipe(self):
        dataset = pd.read_json(json.dumps(raw_data_mock))
        result = dataset['ingredients'].apply(handle_recipe)
        self.assertEqual(len(result[0]), 5)
        self.assertSetEqual(set(result[0]),
                            {'eggs', 'baking', 'powder', 'white', 'sugar'})

    def test_handle_test_recipes(self):
        dataset = pd.read_json(json.dumps(raw_data_mock))
        allowed_ingredients = {'eggs', 'white', 'sugar', 'dog'}
        result = dataset['ingredients'].apply(handle_test_recipes, args=(allowed_ingredients,))
        self.assertEqual(len(result[0]), 3)
        self.assertSetEqual(set(result[0]),
                            {'eggs', 'white', 'sugar'})

    def test_build_ingredients_matrix(self):
        recipes_mock = [[1, 2], [0, 2]]
        matrix = build_ingredients_matrix(recipes_mock, 4)
        matrix = matrix.toarray().tolist()
        self.assertSequenceEqual(matrix[0], [False, True, True, False])
        self.assertSequenceEqual(matrix[1], [True, False, True, False])

# command line interface for users to search for recipes
STOPWORDS_PATH = 'stopwords.txt'
DATASET_PATH = 'cleanedRecipes.jsonl'
INGREDIENT_INDEX_PATH = 'ingredient_index'
FOOD_INDEX_PATH = 'food_index'
ID_TO_RECIPE_PATH = 'id_to_recipe.json'


import ingredient_preprocessor as ip
import ingredient_indexing as ingredient_indexing
from food_ranker import *
import food_indexing as food_indexing
import food_preprocessor as fp


ingredient_tokenizer = ip.SplitTokenizer()

ingredient_tokenizer.tokenize("This is a test sentences, with a comma...., chicken breasts")

stopwords = set()

with open(STOPWORDS_PATH, 'r', encoding='utf-8') as file:
    for stopword in file:
        stopwords.add(stopword.strip())

print(f'Stopwords collected {len(stopwords)}')

# stopwords = {'and', 'the', 'or', 'is', 'for'}
text_key = 'NER'
doc_augment_dict = {}
food_preprocessor = fp.RegexTokenizer('/w+')

ingredient_index = ingredient_indexing.InvertedIndex()
ingredient_index.load(INGREDIENT_INDEX_PATH)

food_index = food_indexing.InvertedIndex()
food_index.load(FOOD_INDEX_PATH)

preprocessor = fp.RegexTokenizer('\w+', lowercase=True, multiword_expressions=None)
ranker = Ranker(food_index, ingredient_index, preprocessor, ingredient_tokenizer, stopwords, BM25)

top25 = ranker.query(query_ingr='pie, flour, cream, apples, blueberries', query_freetext='sweet and spicy pie', query_NOT='eggs, pecans, nuts, almonds')[:25]

#print(top25)

with open(ID_TO_RECIPE_PATH, 'r') as json_file:
    id_to_recipe = json.load(json_file)

#print(top25)

print('RESULTS:')

for i, doc in enumerate(top25):  # (docid, score)
    print(f'{i + 1}: {id_to_recipe[str(doc[0])][0]} - {id_to_recipe[str(doc[0])][1]}')


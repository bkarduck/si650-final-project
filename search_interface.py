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
import webbrowser


def get_browser_choice():
    while True:
        try:
            print('\nChoose your preferred browser:')
            print('1. Google Chrome')
            print('2. Firefox')
            print('3. Safari')

            choice = int(input('Enter the number corresponding to your choice: '))
            if 1 <= choice <= 3:
                return ['chrome', 'firefox', 'safari'][choice - 1]
            else:
                print('Invalid choice. Please enter a number between 1 and 3.')
        except ValueError:
            print('Invalid input. Please enter a valid number.')

def get_user_input():
    while True:
        try:
            query_freetext = input('Enter a general query (e.g., Indian food, sweet and spicy pie, etc.): ')
            query_ingr = input('Enter desired ingredients separated by commas (e.g., flour, basil, oranges, steak), this can be left blank: ')
            query_NOT = input('Enter unwanted ingredients separated by commas (e.g., eggs, pecans, milk, celery) this can be left blank: ')

            if not query_freetext:
                raise ValueError("Please provide valid input for all fields.")
            
            return query_freetext, query_ingr, query_NOT
        except ValueError as ve:
            print(f"Invalid input: {ve}")


print('\nLoading necessary data...')

ingredient_tokenizer = ip.SplitTokenizer()

stopwords = set()

with open(STOPWORDS_PATH, 'r', encoding='utf-8') as file:
    for stopword in file:
        stopwords.add(stopword.strip())

#print(f'Stopwords collected {len(stopwords)}')  # DEBUGGING

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

with open(ID_TO_RECIPE_PATH, 'r') as json_file:
    id_to_recipe = json.load(json_file)

print('Necessary data loaded!')

# top25 = ranker.query(query_ingr='pie, flour, cream, apples, blueberries', query_freetext='sweet and spicy pie', query_NOT='eggs, pecans, nuts, almonds')[:25]

# #print(top25)  # DEBUGGING

browser = get_browser_choice()

while True:
    try:
        user_choice = input('\nDo you want to search for recipes? (yes/no): ').lower()

        if user_choice == 'yes':
            while True:
                query_freetext, query_ingr, query_NOT = get_user_input()

                top25 = ranker.query(
                    query_ingr=query_ingr,
                    query_freetext=query_freetext,
                    query_NOT=query_NOT
                )[:25]

                print('\nRESULTS:')
                for i, doc in enumerate(top25):  # (docid, score)
                    # display a number (1 thru 25): a recipe title - its url
                    print(f'{i + 1}: {id_to_recipe[str(doc[0])][0]} - {id_to_recipe[str(doc[0])][1]}')

                while True:
                    user_choice = input('\nEnter the number (1-25) to open the recipe URL in your web browser, '
                                        'type "back" to go back to the recipe search, '
                                        'type "exit" to stop querying, '
                                        'or anything else to continue: ')

                    if user_choice.lower() == 'back':
                        break
                    elif user_choice.lower() == 'exit':
                        print('\nThanks for using our recipe search engine!\n')
                        exit()

                    try:
                        recipe_number = int(user_choice)
                        if 1 <= recipe_number <= 25:
                            recipe_url = id_to_recipe[str(top25[recipe_number - 1][0])][1]
                            print(f'\nOpening {recipe_url} in your web browser.')
                            webbrowser.get(browser).open(recipe_url, new=2)
                        else:
                            print('Invalid number. Please enter a number between 1 and 25.')
                    except ValueError:
                        print('Invalid input. Please enter a valid number, "back", "exit", or anything else.')

                user_choice = input('\nDo you want to continue searching for recipes? (yes/no): ').lower()
                if user_choice != 'yes':
                    print('\nThanks for using our recipe search engine!\n')
                    exit()
        elif user_choice == 'no':
            print('\nThanks for using our recipe search engine!\n')
            break
        else:
            print('Invalid input. Please enter either "yes" or "no".')
    except Exception as e:
        print(f'An error occurred: {e}')



import numpy as np
from collections import Counter, defaultdict
from sentence_transformers import CrossEncoder
from food_indexing import InvertedIndex
from food_preprocessor import RegexTokenizer, Doc2QueryAugmenter, Tokenizer

import math
from tqdm import tqdm
import json

import ast


class Ranker:
    '''
    The ranker class is responsible for generating a list of documents for a given query, ordered by their
    scores using a particular relevance function (e.g., BM25). A Ranker can be configured with any RelevanceScorer.
    '''


    def __init__(self, food_index: InvertedIndex, ingredient_index: InvertedIndex, food_preprocessor, ingredient_preprocessor, stopwords: set[str], scorer: 'RelevanceScorer', id_to_recipe: dict[int: tuple[str, str, list[str]]]) -> None:
        '''
        Initializes the state of the Ranker object 
        '''
        self.food_index = food_index
        self.ingredient_index = ingredient_index
        self.food_tokenize = food_preprocessor.tokenize
        self.ingredient_tokenize = ingredient_preprocessor.tokenize
        if isinstance(scorer, type):
            scorer = scorer(self.food_index)
        self.scorer = scorer
        self.stopwords = stopwords
        self.id_to_recipe = id_to_recipe


    def query(self, query_ingr: str, query_freetext:str, query_NOT:str) -> list[tuple[int, float]]:
        '''
        Searches the collection for relevant documents to the query and returns a list 
        of documents ordered by their relevance (most relevant first).

        Args:
            query_ingr (str): The string of the list of the ingredients to search for
            query_freetext (str): The string of the free text the user can input 
            query_NOT (str): The string of the list of the ingredients to not include

        Returns:
            list: a list of dictionary objects with keys "docid" and "score" where docid is a particular document in the collection and score is that document's relevance
        '''
        # append ingredients and free text
        query = query_ingr + ' ' + query_freetext
        #print(query)  # DEBUGGING
        tokenized_query = self.food_tokenize(query)
        if self.stopwords is not None and len(self.stopwords) > 0:
            tokenized_query = [word for word in tokenized_query if word not in self.stopwords]
        #print(tokenized_query)  # DEBUGGING

        
        # TODO Fetch a list of possible documents from the index and create a mapping from a document ID to a dictionary of the counts of the query terms in that document.
        # You will pass the dictionary to the RelevanceScorer as input.
        
        possible_docs = []
        docQueryCounts = defaultdict(dict)
        for term in set(tokenized_query):
            if term == None:
                continue
            else:
                postings = self.food_index.get_postings(term)

                if postings is None:
                    continue

                for docid, freq in postings:
                    
                    docQueryCounts[docid][term] = freq
                    possible_docs.append(int(docid))
        
        cleaned_docs = set(possible_docs)
        # use ingredient tokenizer which lemmatizes query for food items
        # people don't want
        query_NOT_tokenized = self.ingredient_tokenize(query_NOT)
        if self.stopwords is not None and len(self.stopwords) > 0:
            query_NOT_tokenized = [word for word in query_NOT_tokenized if word not in self.stopwords]
        
        for term in query_NOT_tokenized:
            #print(term)  # DEBUGGING
            # use ingredient index to get the allergy/not wanted postings
            bad_postings = self.ingredient_index.get_postings(term)
            # all_terms = list(self.ingredient_index.index.keys())
            bad_postings_list = []
            if len(term.split()) == 1:
                all_terms = list(self.ingredient_index.index.keys())
                bad_postings_list = []
                for word in all_terms:
                
                    if term in word and term != word:
                        bad_postings_list.append(self.ingredient_index.get_postings(word))
                for postings in bad_postings_list:
                    if postings is None:
                        continue
                    for docid, freq in postings:
                        if docid in cleaned_docs:
                            cleaned_docs.remove(docid)
                # bad_postings = self.ingredient_index.get_postings(term)
            if bad_postings is None:
                continue
            for docid, freq in bad_postings:

                if docid in cleaned_docs:
                    cleaned_docs.remove(docid)
          

        # TODO Rank the documents using a RelevanceScorer (like BM25 from the below classes) 
        results = []
       
        if len(cleaned_docs) == 0:
            return results
        for doc in cleaned_docs:
   
            score = self.scorer.score(docid=doc, doc_word_counts=docQueryCounts[doc], query_parts=tokenized_query)
       
            results.append((doc, score))

        results.sort(key=lambda x: x[1], reverse=True)

        # LET'S TRY RERANKING THE TOP 100 BASED ON INGREDIENTS!!!
        # the main idea will boost recipes higher up based on the
        # PROPORTION of requested ingredients, e.g., if two recipes, A
        # and B, have 5 ingredients and 20 ingredients respectively, and
        # they both contain 3 ingredients the user specifically requested,
        # then document A will be higher (3/5) than B (3/20), but BOTH
        # will be higher than documents that contian none of the requested
        # ingredients, but still might match the free text query
        top_100, after_top_100 = results[:100], results[100:]
        new_top_100 = []

        for doc in top_100:  # doc is (id, score)
            doc_NERs = ast.literal_eval(self.id_to_recipe[str(doc[0])][2])
            #print(doc_NERs[:15])  # DEBUGGING
            doc_NERs = [self.ingredient_tokenize(ner)[0] for ner in doc_NERs]
            #print(doc_NERs[:15])  # DEBUGGING
            num_NERs = len(doc_NERs)
            num_wanted_in_NERs = 0
            wanted_ing = set(self.ingredient_tokenize(query_ingr))
            num_wanted_ing = len(wanted_ing)

            for ingredient in wanted_ing:
                if ingredient in doc_NERs:
                    num_wanted_in_NERs += 1
            
            user_prop_score = num_wanted_in_NERs / num_wanted_ing
            doc_prop_score = num_wanted_in_NERs / num_NERs  # WORSE

            new_top_100.append((doc[0], user_prop_score))  # * doc[1]
            #new_top_100.append((doc[0], doc_prop_score))  # WORSE
            #new_top_100.append((doc[0], (0.8 * user_prop_score + 0.2 * doc_prop_score)))  # INTERP
            # try:  # HARMONIC MEAN
            #     new_top_100.append((doc[0], 2 / ((1 / user_prop_score) + (1 / doc_prop_score))))
            # except ZeroDivisionError:
            #     new_top_100.append((doc[0], 0))
        
        new_top_100.sort(key=lambda x: x[1], reverse=True)

        # combine the results again
        results = new_top_100 + after_top_100
        # COMMENT OUT THE ABOVE RERANKING STEPS TO SEE DIFFERENCES
        # this code reranks based on ingredients as mentioned above
        # the original scores are lost but who cares I think???
        
        # TODO Return the **sorted** results as format [{docid: 9, score:0.5}, {{docid: 10, score:0.2}}] in descending score
        return results


class RelevanceScorer:
    """
    This is the base interface for all the relevance scoring algorithm.
    It will take a document and attempt to assign a score to it.
    """
    # TODO Implement the functions in the child classes (WordCountCosineSimilarity, DirichletLM, BM25, PivotedNormalization, TF_IDF) and not in this one
    def __init__(self, index: InvertedIndex, parameters) -> None:
        self.index = index
        self.parameters = parameters
        self.statistics = index.get_statistics()
        self.avg_doc_length = self.statistics['mean_document_length']
        self.total_docs = self.statistics['number_of_documents']
        self.total_token_count = self.statistics['total_token_count']

  
    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Returns a score for how relevance is the document for the provided query.

        Args:
            docid: The ID of the document
            doc_word_counts: A dictionary containing all words in the document and their frequencies
            query_parts: A list of all the words in the query
                Words that have been filtered will be None.

        Returns:
            A score for how relevant the document is (Higher scores are more relevant.)
        """
        raise NotImplementedError


# TODO (HW1): Implement unnormalized cosine similarity on word count vectors
class WordCountCosineSimilarity(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters={}) -> None:
        super().__init__(index, parameters)
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]):
        # 1. Find the dot product of the word count vector of the document and the word count vector of the query
        queryList = []
        docList = []
        countQuery = Counter(query_parts)

        for query in set(query_parts):

            if query == None:
                continue
            if query not in doc_word_counts.keys():
                continue
            docCount = doc_word_counts[query]

            queryList.append(countQuery[query])
            docList.append(docCount)

        vector1 = np.array(docList)
        vector2 = np.array(queryList)
 
        score = np.dot(vector1, vector2)

        # 2. Return the score
        return score  


# TODO  Implement DirichletLM
class DirichletLM(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters={'mu': 2000}) -> None:
        super().__init__(index, parameters)
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]):
        # 1. Get necessary information from index

        # 2. Compute additional terms to use in algorithm

        # 3. For all query_parts, compute score
        score = 0
        countQuery = Counter(query_parts)
        mu = self.parameters['mu']
        docLength = self.index.get_doc_metadata(docid)['length']
        queryLength = len(query_parts)
        smoother = queryLength * math.log(mu / (docLength + mu))

        for query in set(query_parts):
            if query == None:
                continue

            query_count = countQuery[query]
            if query not in doc_word_counts.keys():
                continue
            docCount = doc_word_counts[query]



            numTimesTermInDoc = self.index.get_term_metadata(query)['term_total_count']
            print(query, numTimesTermInDoc)

            tfIDF = math.log(1 + (docCount / ( mu * (numTimesTermInDoc / self.total_token_count))))

            queryScore = query_count * tfIDF 

            score += queryScore

        score += smoother

        # 4. Return the score
        return score  


# TODO Implement BM25
class BM25(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters={'b': 0.75, 'k1': 1.2, 'k3': 8}) -> None:
        super().__init__(index, parameters)
        self.index = index
        self.b = parameters['b']
        self.k1 = parameters['k1']
        self.k3 = parameters['k3']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]):
        score = 0

        countQuery = Counter(query_parts)
        for query in query_parts:
            if query == None:
                continue

            query_count = countQuery[query]
            if query not in doc_word_counts.keys():
                continue
           
            docCount = doc_word_counts[query]
           
            postings = self.index.get_postings(query)
            docContainTerm = len(postings)

            docLength = self.index.get_doc_metadata(docid)['length']
            variantIDF = math.log((self.total_docs - docContainTerm + 0.5) / (docContainTerm + 0.5))
            variantTF = ((self.k1 + 1) * docCount) / (self.k1 * ((1 - self.b) + self.b * (docLength / self.avg_doc_length)) + docCount)
            normalizedQTF = ((self.k3 + 1) * query_count) / (self.k3 + query_count)
            queryScore = variantIDF * variantTF * normalizedQTF
            score += queryScore
        # 1. Get necessary information from index
 
        # 2. Find the dot product of the word count vector of the document and the word count vector of the query

        # 3. For all query parts, compute the TF and IDF to get a score

        # 4. Return score
        return score  


# TODO Implement Pivoted Normalization
class PivotedNormalization(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters={'b': 0.2}) -> None:
        super().__init__(index, parameters)
        self.index = index
        self.b = parameters['b']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]):
        score = 0
        countQuery = Counter(query_parts)

        for query in set(query_parts):
            if query == None:
                continue

            query_count = countQuery[query]
            if query not in doc_word_counts.keys():
                continue
     
            docCount = doc_word_counts[query]
            postings = self.index.get_postings(query)
            docContainTerm = len(postings)
         
            idf = math.log((self.total_docs + 1) / docContainTerm)
            docLength = self.index.get_doc_metadata(docid)['length']

            if docCount == 0:
                continue


            fraction = (1 + math.log(1 + math.log(docCount))) / (1 - self.b + (self.b * (docLength / self.avg_doc_length)))

            score += (query_count * fraction * idf)
        # 1. Get necessary information from index

        # 2. Compute additional terms to use in algorithm

        # 3. For all query parts, compute the TF, IDF, and QTF values to get a score
            # hint: 
            ## term_frq will always be >0
            ## doc_frq will always be >0 since only looking at terms that are in both query and doc

        # 4. Return the score
        return score  


# TODO Implement TF-IDF
class TF_IDF(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters={}) -> None:
        super().__init__(index, parameters)
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]):
        score = 0
        for query in set(query_parts):
            if query == None:
                continue
            if query not in doc_word_counts.keys():
                continue
            docCount = doc_word_counts[query]

            postings = self.index.get_postings(query)
            docContainTerm = len(postings)
            tf = math.log(docCount + 1)
            idf = 1 + math.log(self.total_docs/ docContainTerm)
            
            score += (tf * idf)
        # 1. Get necessary information from index

        # 2. Compute additional terms to use in algorithm

        # 3. For all query parts, compute the TF, IDF, and QTF values to get a score

        # 4. Return the score
        return score  


# TODO The CrossEncoderScorer class uses a pre-trained cross-encoder model from the Sentence Transformers package to score a given query-document pair; Used in rankings with a document augmentation step
class CrossEncoderScorer:
    def __init__(self, raw_text_dict: dict[int, str], cross_encoder_model_name: str = 'cross-encoder/msmarco-MiniLM-L6-en-de-v1') -> None:
        """
        Initializes a CrossEncoderScorer object.

        Args:
            raw_text_dict: A dictionary where the document id is mapped to a string with the first 500 words
                in the document
            cross_encoder_model_name: The name of a cross-encoder model
        """
        # TODO: Save any new arguments that are needed as fields of this class
        self.raw_text_dict = raw_text_dict
        self.cross_encoder_model = CrossEncoder(cross_encoder_model_name, max_length=512, device='mps')

    def score(self, docid: int, query: str) -> float:
        """
        Gets the cross-encoder score for the given document.
        
        Args:
            docid: The id of the document
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            The score returned by the cross-encoder model
        """
        # NOTE: Do not forget to handle an edge case
        # (e.g., docid does not exist in raw_text_dict or empty query, both should lead to 0 score)
        if docid not in self.raw_text_dict.keys():
            return 0
        if query == None:
            return 0

        # TODO Get a score from the cross-encoder model
      
        document = self.raw_text_dict[docid]
      
        scores = self.cross_encoder_model.predict([(query, document)])
        return scores[0]


class SampleScorer(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters) -> None:
        pass

    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Scores all documents as 10
        """
        # Print randomly ranked results
        return 10
    
if __name__ == '__main__':
    MAIN_INDEX = 'main_index'
    main_index = InvertedIndex()
    main_index.load(MAIN_INDEX)
    stopwords = set()
    STOPWORD_PATH = 'stopwords.txt'
    with open(STOPWORD_PATH, 'r', encoding='utf-8') as file:
        for stopword in file:
            stopwords.add(stopword.strip())

    food_preprocessor = RegexTokenizer('\w+')
    print('querytime')

    queries = [
        'CHICKEN',
        'machine learning',
        'information retrieval',
        'university of waterloo',
        'computer science is amazing',
    ]
   
 


import numpy as np
from collections import Counter, defaultdict
from sentence_transformers import CrossEncoder
from food_indexing import InvertedIndex
from food_preprocessor import RegexTokenizer, Doc2QueryAugmenter, Tokenizer

import math
from tqdm import tqdm
import json


class Ranker:
    '''
    The ranker class is responsible for generating a list of documents for a given query, ordered by their
    scores using a particular relevance function (e.g., BM25). A Ranker can be configured with any RelevanceScorer.
    '''


    def __init__(self, food_index: InvertedIndex, ingredient_index: InvertedIndex, food_preprocessor, ingredient_preprocessor, stopwords: set[str], scorer: 'RelevanceScorer') -> None:
        '''
        Initializes the state of the Ranker object 
        '''
        self.index = food_index
        self.ingredient_index = ingredient_index
        self.tokenize = food_preprocessor.tokenize
        self.ingredient_tokenize = ingredient_preprocessor.tokenize
        if isinstance(scorer, type):
            scorer = scorer(self.index)
        self.scorer = scorer
        self.stopwords = stopwords
        

    # def query(self, query: str) -> list[dict]:
        # append based off of ingredients??? for MWF
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
        print(query)
        tokenized_query = self.tokenize(query)
        if self.stopwords is not None and len(self.stopwords) > 0:
            tokenized_query = [word for word in tokenized_query if word not in self.stopwords]
        print(tokenized_query)

        
        # TODO (hw2): Fetch a list of possible documents from the index and create a mapping from
        # a document ID to a dictionary of the counts of the query terms in that document.
        # You will pass the dictionary to the RelevanceScorer as input.
        #
        possible_docs = []
        docQueryCounts = defaultdict(dict)
        for term in set(tokenized_query):
            if term == None:
                continue
            else:
                postings = self.index.get_postings(term)

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
            print(term)
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
          

            
        # NOTE: The accumulate_doc_term_counts() method for L2RRanker in l2r.py does something very
        # similar to what is needed for this step

        # TODO (hw1): Rank the documents using a RelevanceScorer (like BM25 from the below classes) 
        results = []
       
        if len(cleaned_docs) == 0:
            return results
        for doc in cleaned_docs:
            #print(tokenized_query)
            score = self.scorer.score(docid=doc, doc_word_counts=docQueryCounts[doc], query_parts=tokenized_query)
            #results.append({'docid': doc, 'score': score})
            results.append((doc, score))

        #results.sort(key=lambda x: x['score'], reverse=True)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # TODO (hw1): Return the **sorted** results as format [{docid: 100, score:0.5}, {{docid: 10, score:0.2}}]
        return results


class RelevanceScorer:
    """
    This is the base interface for all the relevance scoring algorithm.
    It will take a document and attempt to assign a score to it.
    """
    # TODO (HW1): Implement the functions in the child classes (WordCountCosineSimilarity, DirichletLM,
    #             BM25, PivotedNormalization, TF_IDF) and not in this one
    def __init__(self, index: InvertedIndex, parameters) -> None:
        self.index = index
        self.parameters = parameters
        self.statistics = index.get_statistics()
        self.avg_doc_length = self.statistics['mean_document_length']
        self.total_docs = self.statistics['number_of_documents']
        self.total_token_count = self.statistics['total_token_count']

    # NOTE (hw2): Note the change here: `score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]) -> float`
    #             See more in README.md.
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
        return score  # (`score` should be defined in your code; you can call it whatever you want)


# TODO (HW1): Implement DirichletLM
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

            # if docCount == 0:
            #     continue


            numTimesTermInDoc = self.index.get_term_metadata(query)['term_total_count']
            print(query, numTimesTermInDoc)

            tfIDF = math.log(1 + (docCount / ( mu * (numTimesTermInDoc / self.total_token_count))))

            queryScore = query_count * tfIDF 

            score += queryScore

        score += smoother

        # 4. Return the score
        return score  # (`score` should be defined in your code; you can call it whatever you want)


# TODO (HW1): Implement BM25
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
        return score  # (`score` should be defined in your code; you can call it whatever you want)


# TODO (HW1): Implement Pivoted Normalization
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
            #docCount = doc_word_counts[docid][query]
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
        return score  # (`score` should be defined in your code; you can call it whatever you want)


# TODO (HW1): Implement TF-IDF
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

            #docCount = doc_word_counts[docid][query]
            postings = self.index.get_postings(query)
            docContainTerm = len(postings)
            tf = math.log(docCount + 1)
            idf = 1 + math.log(self.total_docs/ docContainTerm)
            
            score += (tf * idf)
        # 1. Get necessary information from index

        # 2. Compute additional terms to use in algorithm

        # 3. For all query parts, compute the TF, IDF, and QTF values to get a score

        # 4. Return the score
        return score  # (`score` should be defined in your code; you can call it whatever you want)


# TODO (HW3): The CrossEncoderScorer class uses a pre-trained cross-encoder model from the Sentence Transformers package
#             to score a given query-document pair; check README for details
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
        self.cross_encoder_model = CrossEncoder(cross_encoder_model_name, max_length=512)

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

        # TODO (HW3): Get a score from the cross-encoder model
        #             Refer to IR_Encoder_Examples.ipynb in Demos folder on Canvas if needed
        document = self.raw_text_dict[docid]
        # pair = [(query, document)]
        scores = self.cross_encoder_model.predict([(query, document)])
        return scores[0]


# TODO (HW1): Implement your own ranker with proper heuristics
class YourRanker(RelevanceScorer):
    pass


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

    main_index = InvertedIndex()
    main_index.load('main_index_augmented')
    stopwords = set()
    STOPWORD_PATH = 'stopwords.txt'
    with open(STOPWORD_PATH, 'r', encoding='utf-8') as file:
        for stopword in file:
            stopwords.add(stopword.strip())

    preprocessor = RegexTokenizer('\w+')
    print('querytime')

    queries = [
        'hello my beautiful people',
        'machine learning',
        'information retrieval',
        'university of waterloo',
        'computer science is amazing',
    ]
    rankerboi = Ranker(main_index, preprocessor, stopwords, BM25)
    for query in tqdm(queries):
        rankerboi.query(query)

  
  
    # docid_to_categories = {}
    # DATASET_PATH = 'wikipedia_200k_dataset.jsonl'
    # with open(DATASET_PATH, 'rt', encoding='utf-8') as file:
    #     for line in tqdm(file, total=200_000):
    #         document = json.loads(line)
    #         docid_to_categories[document['docid']] = document['categories']
    # category_counts = Counter()
    # for cats in tqdm(docid_to_categories.values(), total=len(docid_to_categories)):
    #     for c in cats:
    #         category_counts[c] += 1
    # recognized_categories = set(
    #     [cat for cat, count in category_counts.items() if count >= 1000])
    # doc_category_info = {}
    # for docid, cats in tqdm(docid_to_categories.items(), total=len(docid_to_categories)):
    #     valid_cats = [c for c in cats if c in recognized_categories]
    #     doc_category_info[docid] = valid_cats
    # network_features = {}
    # NETWORK_STATS_PATH = 'network_stats.csv'

    # with open(NETWORK_STATS_PATH, 'r', encoding='utf-8') as file:
    #         for idx, line in enumerate(file):
    #             if idx == 0:
    #                 continue
    #             else:
    #                 # the indexes may change depending on your CSV
    #                 splits = line.strip().split(',')
    #                 network_features[int(splits[0])] = {
    #                     'pagerank': float(splits[1]),
    #                     'authority_score': float(splits[2]),
    #                     'hub_score': float(splits[3])
    #                 }


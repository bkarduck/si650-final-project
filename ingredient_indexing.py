from enum import Enum
import json
import os
from tqdm import tqdm
import orjson
from collections import Counter, defaultdict
import shelve
from ingredient_preprocessor import Tokenizer, SplitTokenizer
import gzip
import csv


class IndexType(Enum):
    # The index currently support is InvertedIndex, 
    InvertedIndex = 'BasicInvertedIndex'
   
    SampleIndex = 'SampleIndex'


class InvertedIndex:
    def __init__(self) -> None:
        """
        The base interface representing the data structure for all index classes.
        The functions are meant to be implemented in the actual index classes and not as part of this interface.
        """
        self.statistics = defaultdict(Counter)  # Central statistics of the index
        self.index = {}  # Index
        self.document_metadata = {}  # Metadata like length, number of unique tokens of the documents
        self.vocabulary = set() 
        self.term_metadata = {}

    # NOTE: The following functions have to be implemented in the three inherited classes and not in this class

    def remove_doc(self, docid: int) -> None:
        """
        Removes a document from the index and updates the index's metadata on the basis of this
        document's deletion.

        Args:
            docid: The id of the document
        """
        # TODO: Implement this to remove a document from the entire index and statistics
        if len(self.document_metadata) > 0 and docid in self.document_metadata.keys():
       
            self.document_metadata.pop(docid)
        if len(self.index) == 0:
            return
        for term in self.index:
            if docid in self.index[term]:
                
                if len(self.index[term]) == 1:
                    self.index.pop(term)
                else:
             
                    self.index[term].remove(docid)

        vocab = self.index.keys()
        self.vocabulary = set(vocab)
        self.get_statistics()

    def add_doc(self, docid: int, tokens: list[str]) -> None:
        """
        Adds a document to the index and updates the index's metadata on the basis of this
        document's addition (e.g., collection size, average document length).

        Args:
            docid: The id of the document
            tokens: The tokens of the document
                Tokens that should not be indexed will have been replaced with None in this list.
                The length of the list should be equal to the number of tokens prior to any token removal.
        """
        # TODO: Implement this to add documents to the index
        length = len(tokens)
        if length == 0:
            self.document_metadata[docid] = {'length': 0, 'unique_tokens': 0}
            return
        
        unique_token_count = 0
        unique_tokens = len(set(tokens))
        counterTokens = Counter(tokens)

        for token, freq in counterTokens.items():
            if token:
                self.vocabulary.add(token)
                termCs = self.index.get(token, [])
                termCs.append((docid, freq))
                self.index[token] = termCs
                unique_token_count += freq

        self.document_metadata[docid] = {'length': length, 'unique_tokens': unique_tokens, 'unique_token_count': unique_token_count}


    def get_postings(self, term: str) -> list[tuple[int, int]]:
        """
        Returns the list of postings, which contains (at least) all the documents that have that term.
        In most implementation, this information is represented as list of tuples where each tuple
        contains the docid and the term's frequency in that document.
        
        Args:
            term: The term to be searched for

        Returns:
            A list of tuples containing a document id for a document
            that had that search term and an int value indicating the term's frequency in 
            the document
        """
        # TODO: Implement this to fetch a term's postings from the index
        if term in self.index:
           
            return self.index[term]
        else:
            return None

    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        """
        For the given document id, returns a dictionary with metadata about that document.
        Metadata should include keys such as the following:
            "unique_tokens": How many unique tokens are in the document (among those not-filtered)
            "length": how long the document is in terms of tokens (including those filtered)

        Args:
            docid: The id of the document

        Returns:
            A dictionary with metadata about the document
        """
        # TODO: Implement to fetch a particular document stored in metadata
        if doc_id in self.document_metadata.keys():
            return self.document_metadata[doc_id]
        
        else:
            return {}

    def get_term_metadata(self, term: str) -> dict[str, int]:
        """
        For the given term, returns a dictionary with metadata about that term in the index.
        Metadata should include keys such as the following:
            "count": How many times this term appeared in the corpus as a whole

        Args:
            term: The term to be searched for

        Returns:
            A dictionary with metadata about the term in the index
        """
        # TODO: Implement to fetch a particular term stored in metadata
        if term in self.index:
           
            self.term_metadata['term_freq'] = len(self.index[term])
            self.term_metadata['term_total_count'] = sum([i[1] for i in self.index[term]])
           

            return self.term_metadata
        else:
            return None

    def get_statistics(self) -> dict[str, int]:
        """
        Returns a dictionary mapping statistical properties (named as strings) about the index to their values.  
        Keys should include at least the following:
            "unique_token_count": how many unique terms are in the index
            "total_token_count": how many total tokens are indexed including filterd tokens), 
                i.e., the sum of the lengths of all documents
            "stored_total_token_count": how many total tokens are indexed excluding filterd tokens
            "number_of_documents": the number of documents indexed
            "mean_document_length": the mean number of tokens in a document (including filter tokens)

        Returns:
              A dictionary mapping statistical properties (named as strings) about the index to their values
        """
        # TODO: Calculate statistics like 'unique_token_count', 'total_token_count',
        #       'number_of_documents', 'mean_document_length' and any other relevant central statistic
        unique_token_count = len(self.vocabulary)
        total_token_count = 0
        number_of_documents = len(self.document_metadata)
        document_length = 0
        stored_total_token_count = 0

        if len(self.index) == 0:
            self.statistics['unique_token_count'] = 0
            self.statistics['total_token_count'] = 0
            self.statistics['number_of_documents'] = 0
            self.statistics['mean_document_length'] = 0
        else: 
            for doc in self.document_metadata:
                document_length += self.document_metadata[doc]['length']
            
                total_token_count += self.document_metadata[doc]['length']
                stored_total_token_count += self.document_metadata[doc]['unique_token_count']

            
            number_of_documents = len(self.document_metadata)
            if number_of_documents == 0:
                mean_document_length = 0
            else:
                mean_document_length = document_length / number_of_documents
        self.statistics['unique_token_count'] = unique_token_count
           
        self.statistics['total_token_count'] = total_token_count
        self.statistics['number_of_documents'] = number_of_documents
        self.statistics['mean_document_length'] = mean_document_length

        return self.statistics
        


    def save(self, index_directory_name: str) -> None:
        """
        Saves the state of this index to the provided directory.
        The save state should include the inverted index as well as
        any metadata need to load this index back from disk.

        Args:
            index_directory_name: The name of the directory where the index will be saved
        """
        # TODO: Save the index files to disk
        if not os.path.exists(index_directory_name):
           
           os.mkdir(index_directory_name)
        directory_path = index_directory_name
        file = index_directory_name + '.json'

        docmeta_with_str = {str(key): value for key, value in self.document_metadata.items()}

        dictToSave = {'index': self.index, 'doc_metadata': docmeta_with_str, 'statistics':self.statistics, 'vocab': list(self.vocabulary)}

        with open(os.path.join(directory_path, file), 'wb') as f:
            f.write(orjson.dumps(dictToSave))
    

    def load(self, index_directory_name: str) -> None:
        """
        Loads the inverted index and any associated metadata from files located in the directory.
        This method will only be called after save() has been called, so the directory should
        match the filenames used in save().

        Args:
            index_directory_name: The name of the directory that contains the index
        """
        # TODO: Load the index files from disk to a Python object
        directory_path = index_directory_name
        file = index_directory_name + '.json'
        if os.path.exists(directory_path):
            with open(os.path.join(directory_path, file), 'rb') as f:
                dictToOpen = orjson.loads(f.read())
                #self.index = json.loads(f.read())
                docmeta_str = dictToOpen['doc_metadata']
                self.document_metadata = {int(key): value for key, value in docmeta_str.items()}
                self.index = dictToOpen['index']
                #self.document_metadata = dictToOpen['doc_metadata']
                #self.vocabulary = dictToOpen['vocab']
                self.statistics = dictToOpen['statistics']
                self.vocabulary = set(dictToOpen['vocab'])


class BasicInvertedIndex(InvertedIndex):
    def __init__(self) -> None:
        """
        An inverted index implementation where everything is kept in memory
        """
        super().__init__()
        self.statistics['index_type'] = 'BasicInvertedIndex'
        # For example, you can initialize the index and statistics here:
        #    self.statistics['docmap'] = {}
        #    self.index = defaultdict(list)
        #    self.doc_id = 0
  
    # TODO: Implement all the functions mentioned in the interface
    # This is the typical inverted index where each term keeps track of documents and the term count per document
    def remove_doc(self, docid: int) -> None:
      
        if len(self.document_metadata) > 0 and docid in self.document_metadata.keys():
       
            self.document_metadata.pop(docid)
        if len(self.index) == 0:
            return
        for term in self.index:
            if docid in self.index[term]:
                
                if len(self.index[term]) == 1:
                    self.index.pop(term)
                else:
             
                    self.index[term].remove(docid)

        vocab = self.index.keys()
        self.vocabulary = set(vocab)
        self.get_statistics()
    def add_doc(self, docid: int, tokens: list[str]) -> None:
        length = len(tokens)
        if length == 0:
            self.document_metadata[docid] = {'length': 0, 'unique_tokens': 0}
            return
        unique_token_count = 0
        unique_tokens = len(set(tokens))
        counterTokens = Counter(tokens)
 

        for token, freq in counterTokens.items():
            if token:
                self.vocabulary.add(token)
                termCs = self.index.get(token, [])
                termCs.append((docid, freq))
                self.index[token] = termCs
                unique_token_count += freq
        self.document_metadata[docid] = {'length': length, 'unique_tokens': unique_tokens, 'unique_token_count': unique_token_count}
    def get_postings(self, term: str) -> dict[str|int, int|list]:
        # TODO implement this to fetch a term's postings from the index
        if term in self.index:
           
            return self.index[term]
        else:
            return None
    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        
        if doc_id in self.document_metadata.keys():
            return self.document_metadata[doc_id]
        
        else:
            return {}
        
    def get_term_metadata(self, term: str) -> dict[str, int]:
        # TODO implement to fetch a particular terms stored metadata
        if term in self.index:
           
            self.term_metadata['term_freq'] = len(self.index[term])
            self.term_metadata['term_total_count'] = sum([i[1] for i in self.index[term]])
           

            return self.term_metadata
        else:
            return None
    def get_statistics(self) -> dict[str, int]:
        '''
        Returns a dictionary mapping statistical properties (named as strings) about the index to their values.  
        Keys should include at least the following:

            "unique_token_count": how many unique terms are in the index
            "total_token_count": how many total tokens are indexed including filtered tokens), 
                i.e., the sum of the lengths of all documents
            "stored_total_token_count": how many total tokens are indexed excluding filtered tokens
            "number_of_documents": the number of documents indexed
            "mean_document_length": the mean number of tokens in a document (including filter tokens)                
        '''
        unique_token_count = len(self.vocabulary)
        total_token_count = 0
        number_of_documents = len(self.document_metadata)
        document_length = 0
        stored_total_token_count = 0

        if len(self.index) == 0:
            self.statistics['unique_token_count'] = 0
            self.statistics['total_token_count'] = 0
            self.statistics['number_of_documents'] = 0
            self.statistics['mean_document_length'] = 0
       
        else: 
            for doc in self.document_metadata:
                document_length += self.document_metadata[doc]['length']
            
                total_token_count += self.document_metadata[doc]['length']
          

            
            number_of_documents = len(self.document_metadata)
            if number_of_documents == 0:
                mean_document_length = 0
            else:
                mean_document_length = document_length / number_of_documents
            self.statistics['unique_token_count'] = unique_token_count
            
            self.statistics['total_token_count'] = total_token_count
            self.statistics['number_of_documents'] = number_of_documents
            self.statistics['mean_document_length'] = mean_document_length
        return self.statistics
        
    def save(self, index_directory_name) -> None:
        '''
        Saves the state of this index to the provided directory. The save state should include the
        inverted index as well as any meta data need to load this index back from disk
        '''
        if not os.path.exists(index_directory_name):
           
           os.mkdir(index_directory_name)
        directory_path = index_directory_name
        file = index_directory_name + '.json'

        docmeta_with_str = {str(key): value for key, value in self.document_metadata.items()}

        dictToSave = {'index': self.index, 'doc_metadata': docmeta_with_str, 'statistics':self.statistics, 'vocab': list(self.vocabulary)}

        with open(os.path.join(directory_path, file), 'wb') as f:
            f.write(orjson.dumps(dictToSave))
    def load(self, index_directory_name) -> None:
        '''
        Loads the inverted index and any associated metadata from files located in the directory.
        This method will only be called after save() has been called, so the directory should
        match the filenames used in save()
        '''
        directory_path = index_directory_name
        file = index_directory_name + '.json'
        if os.path.exists(directory_path):
            with open(os.path.join(directory_path, file), 'rb') as f:
                dictToOpen = orjson.loads(f.read())
                #self.index = json.loads(f.read())
                docmeta_str = dictToOpen['doc_metadata']
                self.document_metadata = {int(key): value for key, value in docmeta_str.items()}
                self.index = dictToOpen['index']
                #self.document_metadata = dictToOpen['doc_metadata']
                #self.vocabulary = dictToOpen['vocab']
                self.statistics = dictToOpen['statistics']
                self.vocabulary = set(dictToOpen['vocab'])





class Indexer:
    """
    The Indexer class is responsible for creating the index used by the search/ranking algorithm.
    """
    @staticmethod
    def create_index(index_type: IndexType, dataset_path: str,
                     document_preprocessor: Tokenizer, stopwords: set[str],
                     minimum_word_frequency: int, text_key="NER",
                     max_docs: int = -1, doc_augment_dict: dict[int, list[str]] | None = None) -> InvertedIndex:
        """
        Creates an inverted index.

        Args:
            index_type: This parameter tells you which type of index to create, e.g., BasicInvertedIndex
            dataset_path: The file path to your dataset
            document_preprocessor: A class which has a 'tokenize' function which would read each document's text
                and return a list of valid tokens
            stopwords: The set of stopwords to remove during preprocessing or 'None' if no stopword filtering is to be done
            minimum_word_frequency: An optional configuration which sets the minimum word frequency of a particular token to be indexed
                If the token does not appear in the document at least for the set frequency, it will not be indexed.
                Setting a value of 0 will completely ignore the parameter.
            text_key: The key in the JSON to use for loading the text
            max_docs: The maximum number of documents to index
                Documents are processed in the order they are seen.
            doc_augment_dict: An optional argument; This is a dict created from the doc2query.csv where the keys are
                the document id and the values are the list of queries for a particular document.

        Returns:
            An inverted index
        """

        if index_type == 'InvertedIndex':
            index = BasicInvertedIndex()
        else:
            index = BasicInvertedIndex()

        doclist = []
        word_counts = Counter()
        if stopwords == set([None]):
            do_not_index = []
        else:
            do_not_index = stopwords.copy()

        if dataset_path.endswith('.gz'):
            max_docs = 0
            f = gzip.open(dataset_path, 'rt')
            
            for doc in tqdm(f):
                doclist.append(json.loads(doc))

        else: 
            with open(dataset_path) as f:
                for doc in tqdm(f):
                    doclist.append(json.loads(doc))

        do_not_index = []

      
        word_counts = Counter()
        maxCheck = max_docs

        for doc in tqdm(doclist):

            if maxCheck == 0:
                break

            tokens = document_preprocessor.tokenize(doc[text_key])
            if minimum_word_frequency > 1:
                
                word_counts.update(tokens)
            
                
            
            maxCheck -= 1



        if minimum_word_frequency > 1:
            
            do_not_index = [w for w in word_counts if word_counts[w] < minimum_word_frequency]

        if stopwords is not None and len(stopwords) > 0:
            do_not_index = stopwords.union(set(do_not_index))
        
        print(word_counts)
        for doc in tqdm(doclist):
            if max_docs == 0:
                break
            
            tokens = document_preprocessor.tokenize(doc[text_key])
            tokenized_doc = [term if term not in do_not_index else None for term in tokens]

            index.add_doc(doc['recipeID'], tokenized_doc)
            max_docs -= 1


      

       

        return index
    
if __name__ == "__main__":

    # stopwords = []
    # with open ('stopwords.txt', 'r') as f:
    #     for line in f:
    #         stopwords.append(line.strip())
    #     #stopwords = f.readlines()

    setOfStopwords = {'and', 'the', 'or', 'could', 'if'}
    preprocessor = SplitTokenizer()
    text_index = Indexer.create_index(IndexType.InvertedIndex, dataset_path='cleanedRecipes.jsonl', document_preprocessor=preprocessor, stopwords=setOfStopwords, minimum_word_frequency=4, text_key='directions', max_docs=100000)
    print(text_index.get_statistics())
    # print(text_index.get_postings('chicken'))

    #text_index.save('text_index')


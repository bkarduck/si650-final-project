import math
import csv
import numpy as np
from tqdm import tqdm


def map_score(search_result_relevances: list[int], cut_off=10) -> float:
    """
    Calculates the mean average precision score given a list of labeled search results, where each item in the list corresponds to a document that was retrieved and is rated as 0 or 1 for whether it was relevant.

    Args:
        search_result_relevances: A list of 0/1 values for whether each search result returned by your
            ranking function is relevant
        cut_off: The search result rank to stop calculating MAP.
            The default cut-off is 10; calculate MAP@10 to score your ranking function.

    Returns:
        The MAP score
    """
    # TODO: Implement MAP
    relevant_count = 0  
    precision_sum = 0

    for i, result in enumerate(search_result_relevances):
        if i >= cut_off:
            break  

        if result > 0:
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)  
            precision_sum += precision_at_i

    if relevant_count == 0:
        return 0

    map_score = precision_sum / cut_off 
    return map_score


def ndcg_score(search_result_relevances: list[float], 
               ideal_relevance_score_ordering: list[float], cut_off=10):
    """
    Calculates the normalized discounted cumulative gain (NDCG) given a lists of relevance scores.
    Relevance scores can be ints or floats, depending on how the data was labeled for relevance.

    Args:
        search_result_relevances: A list of relevance scores for the results returned by your ranking function
            in the order in which they were returned
            These are the human-derived document relevance scores, *not* the model generated scores.
            
        ideal_relevance_score_ordering: The list of relevance scores for results for a query, sorted by relevance score
            in descending order
            Use this list to calculate IDCG (Ideal DCG).

        cut_off: The default cut-off is 10.

    Returns:
        The NDCG score
    """
    # TODO: Implement NDCG
    cut_off = min(cut_off, len(ideal_relevance_score_ordering), len(search_result_relevances))


    idcg = ideal_relevance_score_ordering[0] + sum([(ideal_relevance_score_ordering[i]) / np.log2(i + 1) for i in range(1, cut_off)])

    dcg = search_result_relevances[0] + sum([(search_result_relevances[i]) / np.log2(i + 1) for i in range(1,cut_off)])



    return dcg / idcg


def run_relevance_tests(relevance_data_filename: str, ranker) -> dict[str, float]:
    # TODO: Implement running relevance test for the search system for multiple queries.
    """
    Measures the performance of the IR system using metrics, such as MAP and NDCG.
    
    Args:
        relevance_data_filename: The filename containing the relevance data to be loaded

        ranker: A ranker configured with a particular scoring function to search through the document collection.
            This is probably either a Ranker or a L2RRanker object, but something that has a query() method.

    Returns:
        A dictionary containing both MAP and NDCG scores
    """
    # TODO: Load the relevance dataset

    # TODO: Run each of the dataset's queries through your ranking function

    # TODO: For each query's result, calculate the MAP and NDCG for every single query and average them out

    # NOTE: MAP requires using binary judgments of relevant (1) or not (0). You should use relevance 
    #       scores of (1,2,3) as not-relevant, and (4,5) as relevant.

    # NOTE: NDCG can use any scoring range, so no conversion is needed.
  
    # TODO: Compute the average MAP and NDCG across all queries and return the scores
    relevance_data = {}
    with open(relevance_data_filename, 'r') as f:
        csv_reader = csv.reader(f)
        count = 0
        for line in csv_reader:
            if count == 0:
                headers = line
                queryPosition = headers.index('query')
                docidPosition = headers.index('recipeID')
                relPosition = headers.index('rel')
                count += 1
                continue
            else:
                count +=1
                query = line[queryPosition]
            
                docid = line[docidPosition]
                rel = line[relPosition]
                if query not in relevance_data:
                    relevance_data[query] = [(int(docid), int(rel))]
                else:
                    relevance_data[query].append((int(docid), int(rel)))

        
    scoresList = {}
    mapList = []
    ndcgList = []

    idealScoreList = []
    tracker = 0
    for query, docid in tqdm(relevance_data.items()):
        if tracker == 10:
            break
  
        queryResults = ranker.query(query)
        mapScoreList = []
        ndcgScoreList = []
 
        print('new query')
       
        for doc in queryResults:
            docidList = []
            relList = []
            
            for rels in docid:
                docidList.append(rels[0])
                relList.append(rels[1])
            if doc[0] in docidList:
            #if doc['docid'] in docidList:
               
                # map score
                #if relList[docidList.index(doc['docid'])] > 3:
                if relList[docidList.index(doc[0])] > 3:
                    mapScoreList.append(1)
                else:
                    mapScoreList.append(0)
                # ndcg score
                #ndcgScoreList.append(relList[docidList.index(doc['docid'])])
                ndcgScoreList.append(relList[docidList.index(doc[0])])
            else:
                mapScoreList.append(0)
                ndcgScoreList.append(0)
        if len(queryResults) != len(mapScoreList):
            print("ERROR")
        if len(queryResults) != len(ndcgScoreList):
            print("ERROR")

        idealSorted = sorted(ndcgScoreList, reverse=True)
 
        mapScore = map_score(mapScoreList)
        ndcgScore = ndcg_score(ndcgScoreList, idealSorted)
        mapList.append(mapScore)
        ndcgList.append(ndcgScore)


        tracker += 1
    print(mapList)
    print(len(mapList))
       
    mapAvg = sum(mapList) / len(mapList)
    ndcgAvg = sum(ndcgList) / len(ndcgList)
    return {'map': mapAvg, 'ndcg': ndcgAvg}


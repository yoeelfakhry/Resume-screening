
import json
from config import RELEVANT_MAP_FILE

def hits_at_k(ranked_categories : list , correct_categories : set , k : int ) -> int:
    """
    Calculate the number of corrected predictions in the top k ranked categories.
    
    Args:
        ranked_categories (list): A list of categories ranked by relevance.
        correct_categories (set): A set of correct categories to check against the ranked list.
        k (int): The number of top items to consider for hits calculation.
    
    Returns:
        int: The number of correct predictions in the top k ranked categories.
    """
    return sum(c in correct_categories for c in list(ranked_categories)[:k])



def precision_at_k(ranked_categories : list , correct_categories : set , k : int ) -> float: 
    """
    Calculate the precision at k for a given ranked list of categories and the correct category.

    Args:
        ranked_categories (list): A list of categories ranked by relevance.
        correct_categories (set): A set of correct categories to check against the ranked list.
        k (int): The number of top items to consider for precision calculation.
    
    Returns:
        float: The precision at k, which is the number of correct predictions in the top k ranked categories divided by k.
    """
    return hits_at_k(ranked_categories, correct_categories, k) / k


def reciprocal_rank(ranked_categories : list , correct_categories : set ) -> float:
    """
    Calculate the order of the first correct prediction in the ranked list of categories.

    Args:
        ranked_categories (list): A list of categories ranked by relevance.
        correct_categories (set): A set of correct categories to check against the ranked list.
    
    Returns:
        float: The reciprocal rank, which is 1 divided by the rank position of the first correct prediction. 
            If the correct categories are not found, returns 0.
    """
    for i, category in enumerate(ranked_categories):
        if category in correct_categories:
            return 1 / (i + 1)
    return 0




if __name__ == "__main__":
    ok = 0

    def check(label, condition):
        global ok
        assert condition, f"FAILED: {label}"
        ok += 1
        print(f"  ok  {label}")

    # one shared example you can count by hand
    ranked = ["HR", "IT", "AGRICULTURE", "IT", "HR",
            "CONSULTANT", "IT", "ENGINEERING", "HR", "HR"]
    correct = {"IT"}
    # correct results sit at ranks 2, 4, 7
    

    check("hits@10 counts all three", hits_at_k(ranked, correct, 10) == 3)

    check("hits@5 counts two", hits_at_k(ranked, correct, 5) == 2)

    # check precision@k
    check("precision@10 is 0.3", precision_at_k(ranked, correct, 10) == 0.3)

    check("precision@5 is 0.4", precision_at_k(ranked, correct, 5) == 0.4)

    # check reciprocal rank
    check("reciprocal rank is 1/2", reciprocal_rank(ranked, correct) == 1/2)

    # check no correct predictions
    check("hits is 0 when nothing matches", hits_at_k(ranked, {"SALES"}, 10) == 0)
    check("precision is 0 when nothing matches", precision_at_k(ranked, {"SALES"}, 10) == 0.0)

    fin = ["BANKING", "FINANCE", "HR", "BANKING", "CHEF",
    "ACCOUNTANT", "FINANCE", "HR", "HR", "AVIATION"]
    accepted = {"BANKING", "FINANCE"}
    # correct results sit at ranks 1, 2, 4, 7

    check("hits@10 with a two-category set", hits_at_k(fin, accepted, 10) == 4)
    check("RR is 1.0 when the first result is correct", reciprocal_rank(fin, accepted) == 1.0)

    check("RR is 0 when nothing matches", reciprocal_rank(ranked, {"SALES"}) == 0)

    try:
        n = hits_at_k(["IT", "HR", "IT"], {"IT"}, 10)
        check("a short list does not raise, and counts what is there", n == 2)
    except IndexError:
        check("a short list does not raise, and counts what is there", False)
    

    print(f"\n{ok} checks passed")



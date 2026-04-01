from sentence_transformers import SentenceTransformer
import json
import config

eval_model  = SentenceTransformer(config.EMBEDDING_MODEL)

def load_golden_dataset(path:str):
    """
    Loads golden dataset from a json file.
    Format:[{"question":"...","expected_answer":"..."}]
    """
    with open(path , "r") as f:
        return json.load(f)
    
def compute_similarity(expected:str,actual:str)->float:
    expected_vec = eval.model.encode(expected,convert_to_tensor=True)
    actual_vec=eval.model.encode(actual,convert_to_tensor=True)
    score=util.cos_sim(expected_vec,actual_vec).item()
    return round(score,3)

def run_eval(vectorstore, groq_api_key:str,golden_dataset_path:str):
    from rag import answer_question
    dataset = load_golden_dataset(golden_dataset_path)
    result =[]
    passed=0

    results =[]
    passed=0

    for item in dataset:
        question=item["question"]
        expected=item["expected_answer"]

        response = answer_question(vectorstore,question,groq_api_key)
        actual = response["answer"]

        similarity = compute_similarity(expected,actual)
        passed_threshold=similarity>=config.SIMILARITY_THRESHOLD
        
        if passed_threshold:
            passed +=1
        

        result.append({
            "question":question,
            "expected_answer":expected,
            "actual_answer":actual,
            "similarity_score":similarity,
            "passed":passed_threshold
        })
    
    overall_score = round(passed/len(dataset)*100,1)

    return{
        "overall_score":overall_score,
        "passed":passed,
        "total":len(dataset),
        "threshold":config.SIMILARITY_THRESHOLD,
        "results":results
    }

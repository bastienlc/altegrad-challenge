import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def solution_from_embeddings(graph_embeddings, text_embeddings, save_to="solution.csv"):
    similarity = cosine_similarity(text_embeddings, graph_embeddings)

    solution = pd.DataFrame(similarity)
    solution["ID"] = solution.index
    solution = solution[["ID"] + [col for col in solution.columns if col != "ID"]]
    solution.to_csv(f"outputs/{save_to}", index=False)

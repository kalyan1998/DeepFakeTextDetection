import numpy as np

def load_embeddings(embedding_path):
    with open(embedding_path, 'r') as file:
        for line in file:
            parts = line.split()
            yield np.array([float(x) for x in parts[1:]], dtype=np.float32)

def compute_cosine_similarity_chunked(embedding_path, chunk_size=100):
    loaded_embeddings = list(load_embeddings(embedding_path))
    embeddings = np.vstack(loaded_embeddings)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings /= norms
    
    num_vectors = embeddings.shape[0]
    cosine_similarity_matrix = np.zeros((num_vectors, num_vectors), dtype=np.float32)
    
    for start in range(0, num_vectors, chunk_size):
        end = min(start + chunk_size, num_vectors)
        chunk = embeddings[start:end]
        cosine_similarity_matrix[start:end] = np.dot(chunk, embeddings.T)
    
    return cosine_similarity_matrix

embedding_path = 'counter-fitted-vectors.txt'
cos_sim_matrix = compute_cosine_similarity_chunked(embedding_path)
np.save('cos_sim_counter_fitting_validate.npy', cos_sim_matrix)

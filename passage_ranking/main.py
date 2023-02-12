from sentence_transformers import SentenceTransformer, CrossEncoder, util


def experiment_text_similarity():
    model = SentenceTransformer('/Users/bo/workspace/data/models/sentence-transformers/msmarco-distilbert-base-v4')
    queries = [
        'How big is London',
        'Who is the president of usa',
        'why is the sky blue'
    ]
    passages = [
        'As white light passes through our atmosphere, tiny air molecules cause it to scatter',
        'London has 9,787,426 inhabitants at the 2011 census',
        'Joseph Robinette Biden Jr. is an American politician who is the 46th and current president of the United States.'
    ]
    query_embeddings = model.encode(queries)
    passage_embeddings = model.encode(passages)
    cos_scores = util.cos_sim(query_embeddings, passage_embeddings)
    for i in range(len(queries)):
        print("{} \t\t {} \t\t Score: {:.4f}".format(queries[i], passages[i], cos_scores[i][i]))

    # passage ranking according to the 1st query
    query_embeddings = model.encode(queries[0])
    passage_embeddings = model.encode(passages)
    cos_scores = util.cos_sim(query_embeddings, passage_embeddings)
    cos_scores = list(cos_scores[0])
    result = []
    for score, psg in zip(cos_scores, passages):
        result.append({
            'score': float(score),
            'passage': psg
        })

    result = sorted(result, key=lambda x: x['score'], reverse=True)
    print(result)


def experiment_cross_encoder():
    model_path = '/Users/bo/workspace/data/models/sentence-transformers/ms-marco-MiniLM-L-12-v2'
    model = CrossEncoder(model_path, max_length=512)
    # scores = model.predict([['Query', 'Paragraph1'], ['Query', 'Paragraph2'], ['Query', 'Paragraph3']])
    scores = model.predict([
        ['How many people live in Berlin?',
         'Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.'],
        ['How many people live in Berlin?',
         'Berlin is well known for its museums.']
    ])
    print(scores)


if __name__ == '__main__':
    # experiment_cross_encoder()
    experiment_text_similarity()

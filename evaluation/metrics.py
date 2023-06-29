import string
from typing import List
import nltk
import sklearn
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics import BLEUScore, MatchErrorRate
from sentence_transformers import SentenceTransformer, util
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from rank_eval import Qrels, Run, evaluate

LANGUAGE_CODE_MAP = {"en": "english", "ru": "russian", "es": "spanish"}

DEFAULT_BI_ENCODER_MODEL_PATH = (
    "/nfs-data/models/sentence-transformers/all-mpnet-base-v2"
)


class F1Metrics:
    def __init__(
        self,
        language: str = "en",
        word_ngrams: int = 1,
        lower_case: bool = True,
        remove_punctuation: bool = True,
        ngram_cumulative: bool = False,
    ):
        """
        if ngram_cumulative is true, we take all ngrams into account, up until the defined dimension,
        for example, for n == 3, we consider 1grams, 2grams and 3grams together
        """
        assert (
            language in LANGUAGE_CODE_MAP
        ), f"The language {language} is not supported."
        self.language = LANGUAGE_CODE_MAP[language]
        self.word_ngrams = word_ngrams
        self.lower_case = lower_case
        self.remove_punctuation = remove_punctuation
        self.ngram_cumulative = ngram_cumulative

    @staticmethod
    def _tokenize(text: str, params: dict) -> List[str]:
        language = params["language"]
        lower_case = params["lower_case"]
        remove_punctuation = params["remove_punctuation"]
        tokens = nltk.word_tokenize(text=text, language=language)
        words = []
        for token in tokens:
            word = None
            if remove_punctuation:
                if token not in string.punctuation:
                    word = token
            else:
                word = token

            if word is not None:
                if lower_case:
                    word = word.lower()
                words.append(word)

        return words

    @staticmethod
    def _ngrams(grams: List[str], n: int = 1) -> List[str]:
        ngram_list = list(nltk.ngrams(grams, n))
        n_grams = []
        for tup in ngram_list:
            n_grams.append(" ".join(tup))
        return n_grams

    def measure_f1token(self, prediction: str, reference: str, **kwargs):
        if kwargs:
            d = self.__dict__.copy()
            d.update(kwargs)
        else:
            d = self.__dict__
        tokens_pred = self._tokenize(prediction, d)
        tokens_gold = self._tokenize(reference, d)
        ngrams_pred = self._ngrams(tokens_pred, n=d["word_ngrams"])
        ngrams_gold = self._ngrams(tokens_gold, n=d["word_ngrams"])
        if d["ngram_cumulative"] and d["word_ngrams"] > 1:
            n = d["word_ngrams"] - 1
            while n >= 1:
                ngrams_pred += self._ngrams(tokens_pred, n=n)
                ngrams_gold += self._ngrams(tokens_gold, n=n)
                n = n - 1

        overlap = 0.0
        num_ngrams_in_reference = len(ngrams_gold)
        num_ngrams_in_prediction = len(ngrams_pred)
        for pred in ngrams_pred:
            if pred in ngrams_gold:
                # to avoid a predicted gram being repetitively counted in the reference
                ngrams_gold.remove(pred)
                overlap += 1

        precision = 0.0
        recall = 0.0
        if overlap > 0 and num_ngrams_in_reference > 0 and num_ngrams_in_prediction > 0:
            recall = overlap / num_ngrams_in_reference
            precision = overlap / num_ngrams_in_prediction
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
        return {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "word_ngrams": d["word_ngrams"],
            "lower_case": d["lower_case"],
            "remove_punctuation": d["remove_punctuation"],
            "ngram_cumulative": d["ngram_cumulative"],
        }

    @staticmethod
    def measure_f1class(predictions: List[str], references: List[str]):
        assert len(predictions) == len(
            references
        ), "The lengths of the two input lists should be the same."
        precision, recall, f1macro, _ = sklearn.metrics.precision_recall_fscore_support(
            references, predictions, average="macro"
        )
        _, _, f1micro, _ = sklearn.metrics.precision_recall_fscore_support(
            references, predictions, average="micro"
        )
        _, _, f1weighted, _ = sklearn.metrics.precision_recall_fscore_support(
            references, predictions, average="weighted"
        )

        overlap = 0.0
        for p, r in zip(predictions, references):
            if p == r:
                overlap += 1
        accuracy = overlap / len(predictions)
        return {
            "f1-macro": f1macro,
            "f1-micro": f1micro,
            "f1-weighted": f1weighted,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
        }


class TextMetrics:
    """
    see https://torchmetrics.readthedocs.io/en/stable/
    """

    def __init__(self):
        self.bleu1 = BLEUScore(n_gram=1)
        self.bleu2 = BLEUScore(n_gram=2)
        self.bleu4 = BLEUScore(n_gram=4)
        self.rouge = ROUGEScore()
        self.mer = MatchErrorRate()

    def measure(self, prediction: str, reference: str or List[str]):
        """
        A prediction may have multiple references
        """
        bleu_reference = [reference]
        if type(reference) == list:
            bleu_reference = reference
        bleu_1_score = self.bleu1([prediction], [bleu_reference])
        bleu_2_score = self.bleu2([prediction], [bleu_reference])
        bleu_4_score = self.bleu4([prediction], [bleu_reference])
        rouge_results = self.rouge(prediction, reference)
        mer_score = self.mer([prediction], [reference])

        return {
            "bleu1": float(bleu_1_score),
            "bleu2": float(bleu_2_score),
            "bleu4": float(bleu_4_score),
            "rouge1": float(rouge_results["rouge1_fmeasure"]),
            "rouge2": float(rouge_results["rouge2_fmeasure"]),
            "rougeL": float(rouge_results["rougeL_fmeasure"]),
            "mer": float(mer_score),
        }


class RankingMetrics:
    """
    see https://pypi.org/project/rank-eval/
    from rank_eval import Qrels, Run, evaluate

    qrels = Qrels()
    qrels.add_multi(
        q_ids=["q_1", "q_2"],
        doc_ids=[
            ["doc_12", "doc_25"],  # q_1 relevant documents
            ["doc_11", "doc_2"],  # q_2 relevant documents
        ],
        scores=[
            [5, 3],  # q_1 relevance judgements
            [6, 1],  # q_2 relevance judgements
        ],
    )

    run = Run()
    run.add_multi(
        q_ids=["q_1", "q_2"],
        doc_ids=[
            ["doc_12", "doc_23", "doc_25", "doc_36", "doc_32", "doc_35"],
            ["doc_12", "doc_11", "doc_25", "doc_36", "doc_2",  "doc_35"],
        ],
        scores=[
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        ],
    )
    # Compute scores for multiple metrics at once
    evaluate(qrels, run, ["map@5", "mrr"])
    >>> {"map@5": 0.6416, "mrr": 0.75}
    """

    def __init__(self):
        self.qrels = Qrels()
        self.run = Run()

    def add_reference(
        self, query_id: str, doc_ids: List[str], scores: List[int or float]
    ):
        self.qrels.add(query_id, doc_ids, scores)

    def add_prediction(
        self, query_id: str, doc_ids: List[str], scores: List[int or float]
    ):
        self.run.add(query_id, doc_ids, scores)

    def reset(self):
        self.qrels = Qrels()
        self.run = Run()

    def measure(self):
        return evaluate(
            self.qrels,
            self.run,
            ["map@1", "map@5", "map@10", "mrr@1", "mrr@5", "mrr@10"],
        )


class BertMetrics:
    def __init__(self, mode: str = "bi-encoder", device: str = 'cuda:0'):
        if mode == "bi-encoder":
            self.model = SentenceTransformer(DEFAULT_BI_ENCODER_MODEL_PATH, device=device)
        else:
            raise Exception(f"The mode {mode} is not supported yet.")

    def measure(self, prediction: str, reference: str) -> float:
        pred_encoding = self.model.encode(prediction)
        ref_encoding = self.model.encode(reference)
        cos_score = util.cos_sim(pred_encoding, ref_encoding)[0][0]
        return float(cos_score)


class CorrelationMetrics:
    @staticmethod
    def measure(input_a: List[int or float], input_b: List[int or float]):
        # two samples with interval values, gaussian
        pearson = stats.pearsonr(input_a, input_b)
        # two samples with ordinal values, gaussian
        spearman = stats.spearmanr(input_a, input_b)
        # two independent samples, no gaussian assumption
        mwu = stats.mannwhitneyu(input_a, input_b)
        # two paired samples, no gaussian assumption
        wsr = stats.wilcoxon(input_a, input_b)
        return {
            "pearson": pearson.statistic,
            "spearman": spearman.statistic,
            "mann-whitney-u": mwu.statistic,
            "wilcoxon-signed-rank": wsr.statistic,
        }


class RegressionMetrics:
    @staticmethod
    def measure(prediction: List[int or float], reference: List[int or float]):
        # r-squared, can be negative to 1
        r2 = r2_score(reference, prediction)
        mse = mean_squared_error(reference, prediction)
        mae = mean_absolute_error(reference, prediction)
        return {"r2": r2, "mse": mse, "mae": mae}


if __name__ == "__main__":
    # How to use:

    # f1m = F1Metrics()
    # result = f1m.measure_f1token(prediction='a a a a a', reference='a c e f g h i')
    # print(result)
    # {'f1': 0.16666666666666666,
    #  'precision': 0.2,
    #  'recall': 0.14285714285714285,
    #  'word_ngrams': 1,
    #  'lower_case': True,
    #  'remove_punctuation': True,
    #  'ngram_cumulative': False}

    # tm = TextMetrics()
    # pred = 'the cat is on the mat'
    # ref = 'there is a cat on the mat'
    # results = tm.measure(pred, ref)
    # pprint(results)
    # {'bleu4': 0.0,
    #  'mer': 0.5714285969734192,
    #  'rouge1': 0.7692307829856873,
    #  'rouge2': 0.3636363744735718,
    #  'rougeL': 0.6153846383094788}

    # bm = BertMetrics()
    # pred = 'the cat is on the mat'
    # ref = 'what is up'
    # result = bm.measure(pred, ref)
    # print(result)

    # a, b = [1, 2, 3, 4, 5], [5, 6, 7, 8, 7]
    # result = CorrelationMetrics.measure(a, b)
    # print(result)

    # a, b = [1, 2, 3, 4, 5], [5, 6, 7, 8, 7]
    # a, b = [1, 1, 1, 1, 1], [1, 1, 2, 2, 1]
    # result1 = RegressionMetrics.measure(a, b)
    # result2 = CorrelationMetrics.measure(a, b)

    print()


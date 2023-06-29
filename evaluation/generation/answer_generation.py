from multiprocessing import Pool
import time
from dotenv import load_dotenv
import pytorch_lightning as pl

from chair.utils.prompt_templates import knowledge_prompt, question_knowledge_prompt

load_dotenv(".env.shared")
import ssl
from urllib.request import urlopen
import html2text
from typing import List, Union, Dict

ssl._create_default_https_context = ssl._create_unverified_context
from duckduckgo_search import ddg
from nltk.tokenize import sent_tokenize

# from rank_bm25 import BM25Okapi
from transformers import BloomForCausalLM, BloomTokenizerFast


def knowledge_item_prompt(i, search_result):
    title_and_link = f"{search_result['title']} ({search_result['url']})"
    return knowledge_prompt(i + 1, title_and_link, search_result["excerpt"])


def clean_webpage_text(text: str) -> str:
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if line and not line.startswith("*"):
            lines.append(line)
    return "\n".join(lines)


def crawl_page(search_result_item):
    """
    input search_result_item format:
    {
        "title": "the title of the web page",
        "href": "the url of the web page",
        "body": "the brief text summary under the url"
    }
    output search_result_item format:
    {
        "title": "the title of the web page",
        "url": "the url of the web page",
        "excerpt": "the brief text summary under the url",
        "content": "the content of the web page"
    }
    """
    converter = html2text.HTML2Text()
    converter.ignore_links = True
    converter.ignore_images = True
    try:
        url = search_result_item["href"]
        html = urlopen(url).read().decode("utf8")
        text = converter.handle(html)
        text = clean_webpage_text(text)
        success = True
    except Exception as e:
        print(e)
        text = ""
        success = False

    search_result_item.update(
        {
            "url": search_result_item["href"],
            "excerpt": search_result_item["body"],
            "content": text,
        }
    )
    search_result_item.pop("body")
    search_result_item.pop("href")
    return search_result_item, success


def crawl_pages(search_results):
    with Pool(5) as p:
        res = p.map(crawl_page, search_results)
        p.terminate()
        p.join()

    pages = [page for page, success in res if success]
    return pages


def retrieve_documents(query: str):
    start = time.time()
    # results = search(message, num_results=5, advanced=True)
    search_results = ddg(query, safesearch="Off", max_results=5)
    search_results = list(search_results)
    end = time.time()
    print(f"Time on search: {end - start} seconds")

    start = time.time()
    pages = crawl_pages(search_results)
    end = time.time()
    print(f"Time on crawling: {end - start} seconds")
    return pages


def mock_retrieve_documents_with_perplexity_datum(item):
    docs = []
    for s in item["sources"]:
        docs.append(
            {
                "title": s["title"],
                "url": s["url"],
                "excerpt": s["text"],
                "content": s["text"],
            }
        )
    return docs


def load_bloom_model_for_inference(model_path: str, device: str = "cuda:0"):
    print(f"loading bloom model and tokenizer from {model_path}, with device {device}")
    tokenizer = BloomTokenizerFast.from_pretrained(model_path)
    model = BloomForCausalLM.from_pretrained(model_path)
    model = model.to(device).eval()
    return model, tokenizer


def generate_responses(
    model: BloomForCausalLM,
    tokenizer: BloomTokenizerFast,
    prompt: str,
    num_results: int = 1,
    gen_args: dict = None,
):
    if not gen_args:
        gen_args = {
            "max_new_tokens": 128,
            "do_sample": True,
            "top_k": 0,
            "top_p": 0.92,
            "temperature": 0.8,
        }
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs.to(model.device)
    src_len = inputs["input_ids"].size(1)
    # outputs = model.generate(**inputs, **gen_args,
    #                          num_return_sequences=num_results,
    #                          return_dict_in_generate=True,
    #                          output_scores=True)

    outputs = model.generate(**inputs, **gen_args, num_return_sequences=num_results)
    if num_results > 1:
        # TODO: beam or sample with different seeds?
        # outputs = model.generate(**inputs,
        #                          max_new_tokens=128,
        #                          do_sample=True,
        #                          top_k=10,
        #                          num_return_sequences=num_results)
        responses = tokenizer.batch_decode(
            outputs[:, src_len:], skip_special_tokens=True
        )
        result = [res.split('"""')[0] for res in responses]
    else:
        # outputs = model.generate(**inputs,
        #                          max_new_tokens=128,
        #                          do_sample=True,
        #                          top_k=10)
        response = tokenizer.decode(outputs[0][src_len:])
        result = response.split('"""')[0]

    return result


def chat(
    model: BloomForCausalLM,
    tokenizer: BloomTokenizerFast,
    message: str,
    docs: List[any] = None,
    gen_args: dict = None,
):
    if message:
        if not docs:
            docs = retrieve_documents(message)
        # compose knowledge prompt
        knowledge_items = [knowledge_item_prompt(i, doc) for i, doc in enumerate(docs)]
        knowledge = " ".join(knowledge_items).strip()
        # compose final prompt
        final_prompt = question_knowledge_prompt(message, knowledge)
        # generate response
        start = time.time()
        response = generate_responses(
            model, tokenizer, final_prompt, num_results=1, gen_args=gen_args
        )
        end = time.time()
        print(f"Time on final QA: {end - start} seconds")
        return response


def experiment_sft_1():
    pl.seed_everything(42)
    # model_path = "/nfs-data/models/bloom-560m" # pretrained
    # model_path = '/obs/models/trained/rm_bloom_560m_v1/huggingface'  # bloom560m-rm, 4gpu trained
    # model_path = "/obs/models/trained/sft_bloomx_560m_v1/huggingface"  # bloom560m-sft, 1gpu trained
    model_path = "/obs/models/trained/sft_bloomx_560m_v2/huggingface"  # bloom560m-sft, 4gpu trained

    print("load model and tokenizer from ", model_path)
    model, tokenizer = load_bloom_model_for_inference(model_path)

    # from chair.models.bloomx.modeling_bloomx import BloomxForCausalLM
    # modelx = BloomxForCausalLM.from_pretrained(model_path)

    response = chat(model, tokenizer, "why is the sky blue")
    # responsex = chat(modelx, tokenizer, 'why is the sky blue')
    print("response: ", response)


def experiment_sft_2():
    pl.seed_everything(42)
    # model_path = "/nfs-data/models/bloom-560m" # pretrained
    # model_path = '/nfs-data/chair-outputs/2023-02-24/13-38-42/huggingface'  # bloom560m-rm, 4gpu trained
    # model_path = '/nfs-data/chair-outputs/2023-02-05/12-28-53/huggingface'  # bloom560m-sft, 1gpu trained
    model_path = "/nfs-data/chair-outputs/2023-02-21/21-01-06/huggingface"  # bloom560m-sft, 4gpu trained

    print("load model and tokenizer from ", model_path)
    model, tokenizer = load_bloom_model_for_inference(model_path)
    prompt = '''Given the following question, some extracted knowledge from the web, write an answer with references: QUESTION: What causes you to itch? Just curious. KNOWLEDGE: """ [1]: What Causes Itching? - Scientific Reasons Behind Why We Itch (www.prevention.com) extract: Put simply, you itch because your skin has receptors called pruriceptors (itch-sensing nerve endings) which get stimulated and, in turn, cause that itchy feeling, explains Melanie Grossman, MD, a board-certified dermatologist based in New York City. As part of the immune response, your body releases substances called histamines, triggering the itch. There’s a deep-rooted evolutionary advantage to the itch: It’s your body’s way of letting you know ASAP that something (an [2]: Itchy skin (pruritus) - Symptoms and causes - Mayo Clinic (www.mayoclinic.org) extract: Itchy skin is an uncomfortable, irritating sensation that makes you want to scratch. Also known as pruritus (proo-RIE-tus), itchy skin is often caused by dry skin. It's common in older adults, as skin tends to become drier with age. [3]: Itching: What's Causing Your Itchy Skin? (with Pictures) (www.healthline.com) extract: Itchy skin, also known as pruritus, is an irritating and uncontrollable sensation that makes you want to scratch to relieve the feeling. The possible causes for itchiness include internal illnesses and skin conditions. [4]: Jock itch - Symptoms and causes - Mayo Clinic (www.mayoclinic.org) extract: Jock itch (tinea cruris) is a fungal infection that causes a red and itchy rash in warm and moist areas of the body. The rash often affects the groin and inner thighs and may be shaped like a ring. [5]: Anal itching - Symptoms and causes - Mayo Clinic (www.mayoclinic.org) extract: Sometimes the cause of anal itching isn't identifiable. Possible causes of anal itching include: * Irritants. Fecal incontinence and long-term (chronic) diarrhea can irritate the skin. Or your skin care routine may include products or behaviors that irritate the skin, such as using harsh soaps or moist wipes and washing too aggressively. * Infections. These include sexually transmitted infections, pinworms, and yeast infections. * Skin conditions. Sometimes anal itching is the result of a specific skin condition, such as psoriasis or contact dermatitis. """ ANSWER: """'''
    response = generate_responses(model, tokenizer, prompt)
    print("response: ", response)


if __name__ == "__main__":
    print("start")
    # experiment_sft_1()
    experiment_sft_2()
    print("end")


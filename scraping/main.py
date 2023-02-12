from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import undetected_chromedriver as uc
from time import sleep
from typing import List
import json
from tqdm import tqdm
import csv


def convert_question_to_url(question: str) -> str:
    perplexity_ai_base_url = 'https://www.perplexity.ai/'
    modified_question = question.replace(' ', '+')
    target_url = f'{perplexity_ai_base_url}?q={modified_question}'
    print(f'Target URL: {target_url}')
    return target_url


def get_concise_answer(browser, delay) -> str:
    xpath = '//*[@id="root"]/div/div[2]/div/div/div/div/div[2]/div/div/div/div[1]/div[2]/div/div[2]/div/div/span'
    elm = WebDriverWait(browser, delay).until(EC.presence_of_element_located((By.XPATH, xpath)))
    text = elm.text
    print(f'Concise Answer: {text}.')
    return text


def get_detailed_answer(browser, delay) -> str:
    xpath = '//*[@id="root"]/div/div[2]/div/div/div/div/div[2]/div/div/div/div[1]/div[2]/div/div[2]/div/div/div/span'
    elm = WebDriverWait(browser, delay).until(EC.presence_of_element_located((By.XPATH, xpath)))
    text = elm.text
    print(f'Detailed Answer: {text}.')
    return text


def click_view_detailed_btn(browser, delay):
    xpath = '//*[@id="root"]/div/div[2]/div/div/div/div/div[2]/div/div/div/div[1]/div[2]/div/div[1]/div/div[2]/button'
    elm = WebDriverWait(browser, delay).until(EC.element_to_be_clickable((By.XPATH, xpath)))
    elm.click()
    print('View Detailed Button is clicked.')


def click_view_list_btn(browser, delay):
    xpath = '//*[@id="root"]/div/div[2]/div/div/div/div/div[2]/div/div/div/div[1]/div[3]/div/div[1]/div/div[2]/button'
    elm = WebDriverWait(browser, delay).until(EC.element_to_be_clickable((By.XPATH, xpath)))
    elm.click()
    print('View List Button is clicked.')


def get_sources(browser, delay) -> List[dict]:
    x = '//*[@id="root"]/div/div[2]/div/div/div/div/div[2]/div/div/div/div[1]/div[3]/div/div[1]/div/div[1]/div/div'
    elm = WebDriverWait(browser, delay).until(EC.presence_of_element_located((By.XPATH, x)))
    n_sources = elm.text.replace(' SOURCES', '')
    n_sources = int(n_sources)
    print('#sources: ', n_sources)
    sources = []

    if n_sources > 1:
        for i in range(1, n_sources + 1):
            x = f'//*[@id="root"]/div/div[2]/div/div/div/div/div[2]/div/div/div/div[1]/div[3]/div/div[2]/div[{i}]/div/a'
            elm = WebDriverWait(browser, delay).until(EC.presence_of_element_located((By.XPATH, x)))
            ref_text = elm.text
            ref_link = elm.get_attribute('href')
            x = f'//*[@id="root"]/div/div[2]/div/div/div/div/div[2]/div/div/div/div[1]/div[3]/div/div[2]/div[{i}]/div/div'
            div = WebDriverWait(browser, delay).until(EC.presence_of_element_located((By.XPATH, x)))
            source_text = div.text
            sources.append({
                'title': ref_text,
                'url': ref_link,
                'text': source_text
            })
    elif n_sources == 1:
        x = f'//*[@id="root"]/div/div[2]/div/div/div/div/div[2]/div/div/div/div[1]/div[3]/div/div[2]/div/div/a'
        a = WebDriverWait(browser, delay).until(EC.presence_of_element_located((By.XPATH, x)))
        ref_text = a.text
        ref_link = a.get_attribute('href')
        x = '//*[@id="root"]/div/div[2]/div/div/div/div/div[2]/div/div/div/div[1]/div[3]/div/div[2]/div/div/div'
        div = WebDriverWait(browser, delay).until(EC.presence_of_element_located((By.XPATH, x)))
        source_text = div.text
        sources.append({
            'title': ref_text,
            'url': ref_link,
            'text': source_text
        })

    return sources


def get_answers_and_sources(browser, delay, question) -> dict:
    print(f'\nquestion: {question}')

    def _extract():
        target_url = convert_question_to_url(question)
        browser.get(target_url)
        sleep(10)
        concise_answer = get_concise_answer(browser, delay)
        sleep(0.1)
        click_view_list_btn(browser, delay)
        sleep(0.1)
        click_view_detailed_btn(browser, delay)
        sleep(3)
        sources = get_sources(browser, delay)
        sleep(8)
        detailed_answer = get_detailed_answer(browser, delay)
        result = {
            'question': question,
            'answer': concise_answer,
            'answer_detailed': detailed_answer,
            'sources': sources
        }
        # print(json.dumps(result, ensure_ascii=False))
        return result

    try:
        return _extract()
    except TimeoutException:
        print("Loading took too much time!")
        sleep(3)
        return _extract()


def main():
    delay = 20
    browser = uc.Chrome()
    # questions = [
    #     'who is the author of harry potter',
    #     'why 911 happened'
    # ]
    questions = []
    with open('TruthfulQA.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            questions.append(row[2])

    output_path = 'results_temp.jsonl'
    with open(output_path, 'w') as f:
        for question in tqdm(questions):
            result = get_answers_and_sources(browser, delay, question)
            line = json.dumps(result, ensure_ascii=False)
            f.write(line + '\n')


if __name__ == '__main__':
    print('Start...')
    main()

    # with open('results.jsonl', 'r') as f:
    #     for l in f:
    #         d = json.loads(l)
    #         print()

    # qs = []
    # with open('results_temp.jsonl', 'r') as f:
    #     for l in f:
    #         d = json.loads(l)
    #         a = d['answer']
    #         q = d['question']
    #         if len(a.split()) < 10:
    #             print(q)
    #             print(a)
    #             print('-------')
    #             qs.append(q)
    #         else:
    #             pass

    # with open('remaining_questions.txt', 'w') as f:
    #     for q in qs:
    #         f.write(q+'\n')

    print('Finish')

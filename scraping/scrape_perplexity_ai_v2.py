import csv
import json
import os
from pathlib import Path
from time import sleep
from typing import List

import numpy as np
import pandas as pd
import undetected_chromedriver as uc
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


def element_fully_loaded(browser, class_name, check_interval=5):
    current_content = None
    elm = WebDriverWait(browser, 100).until(EC.presence_of_element_located((By.CLASS_NAME, class_name)))
    while True:
        if not elm.text:  # not loaded yet
            sleep(check_interval)
        elif elm.text != current_content:  # still generating
            current_content = elm.text
            sleep(check_interval)
        else:  # no changes after check_interval seconds
            break
    # if elm.text == "No results found\nPlease try again later.":
    #     raise TimeoutException

    return elm


def convert_question_to_url(question: str) -> str:
    perplexity_ai_base_url = 'https://www.perplexity.ai/'
    modified_question = question.replace(' ', '+')
    target_url = f'{perplexity_ai_base_url}?q={modified_question}'
    print(f'Target URL: {target_url}')
    return target_url


def verify_answer(answer: str) -> str:
    empty_answer = ''
    if 'No sources found. Try a more general question.' in answer:
        return empty_answer

    if 'No results found' in answer and 'Error in processing query' in answer:
        return empty_answer

    return answer


def get_concise_answer(browser, delay) -> str:
    xpath = '//*[@id="root"]/div/div[2]/div/div/div/div/div[2]/div/div/div/div[1]/div[2]/div/div[2]/div/div/span'
    elm = element_fully_loaded(browser, "min-h-\[81px\]")
    text = elm.text
    print(f'Concise Answer: {text}')
    return verify_answer(text)


def get_detailed_answer(browser, delay) -> str:
    elm = element_fully_loaded(browser, "min-h-\[81px\]")
    text = elm.text
    print(f'Detailed Answer: {text}')
    return verify_answer(text)


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
    element_fully_loaded(browser, "leading-relaxed")
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
            ref_text = " ".join(elm.text.split(f"{i}. ")[1:])
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
        ref_text = " ".join(ref_text.split("1. ")[1:])
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
        # sleep(10)
        concise_answer = get_concise_answer(browser, delay)
        if concise_answer:
            sleep(0.1)
            click_view_list_btn(browser, delay)
            sleep(0.1)
            click_view_detailed_btn(browser, delay)
            # sleep(5)
            detailed_answer = get_detailed_answer(browser, delay)
            sources = None
            if detailed_answer:
                # sleep(3)
                sources = get_sources(browser, delay)
        else:
            detailed_answer = None
            sources = None
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


def main(start_ind=0, end_ind=np.inf):
    delay = 20
    browser = uc.Chrome()

    while True:
        try:
            source_files = os.listdir("inputs")
            for fname in source_files:
                rel_path = f"./inputs/{fname}"
                name_base = Path(fname).stem
                print(f"Loading file {fname}...")
                if fname.endswith(".csv"):
                    df = pd.read_csv(rel_path, delimiter=',')
                    question_ind = 2
                elif fname.endswith(".jsonl"):
                    df = pd.read_json(rel_path, lines=True)
                    question_ind = 1
                else:
                    continue
                print("File loaded")
                for i in range(start_ind, min(end_ind, len(df))):
                    target_file = f"outputs/{name_base}_{i}.jsonl"
                    if os.path.exists(target_file):
                        continue

                    print('\ni: ', i)
                    question = df.loc[i][question_ind]
                    if len(question) > 255:
                        continue
                    result = get_answers_and_sources(browser, delay, question)
                    if result['sources'] is None:
                        continue
                    line = json.dumps(result, ensure_ascii=False)
                    f = open(target_file, "w", encoding="utf-8")
                    f.write(line + '\n')
                    f.close()
            break
        except Exception as e:
            print(e)
            cooldown = 600
            print(f"Quota probably has exceeded, sleeping for {cooldown} seconds")
            sleep(cooldown)


if __name__ == '__main__':
    print('Start...')
    main(9001, 10000)
    print('Finish')


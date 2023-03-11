import gradio as gr
import json
from time import time
import datetime


def load_json(json_path: str):
    print(f"\nloading json from {json_path}...")
    start = time()
    with open(json_path, "r") as file_reader:
        data = json.load(file_reader)
        print("json is loaded from ", json_path)
        lapse = time() - start
        print(f"time elapsed: {datetime.timedelta(seconds=lapse)}")
        return data


def save_json(json_path: str, target_object: any):
    print(f"\nsaving json to {json_path}...")
    start = time()
    with open(json_path, "w", encoding="utf-8") as file_writer:
        json.dump(target_object, file_writer, ensure_ascii=False)
        print("json is saved to ", json_path)
        lapse = time() - start
        print(f"time elapsed: {datetime.timedelta(seconds=lapse)}")


demo = gr.Blocks()
data_path = '/Users/bo/workspace/data/perplexity_ai/sample.json'
data = load_json(data_path)
items = data['data']
items_dict = {}
for item in items:
    item['annotations'] = {}
    items_dict[item['id']] = item


def submit(username_str):
    username_str = username_str.strip()
    timestamp = str(datetime.datetime.now()).replace(' ', '_')
    output_filename = f'annotations__{username_str}__{timestamp}.json'
    print(f'Save results to {output_filename}')
    save_json(output_filename, {
        'meta': data['meta'],
        'data': items
    })


def compose_sources_html(refs):
    html = ''
    for i, s in enumerate(refs):
        title = s['title']
        url = s['url']
        text = s['text']
        html += f'''<h6>[{i + 1}] {title}</h6>
        <a href="{url}">{url}</a>
        <p>Excerpt: {text}</p>
        '''
    return html


def answer1_radio_change(choice, target):
    choice = choice.lower()
    score = None
    if 'bad' in choice:
        score = 0
    elif 'ok' in choice:
        score = 1
    elif 'good' in choice:
        score = 2

    items_dict[target]['annotations']['answer_pos_score'] = score
    print(choice)
    print(target)


def answer1_comment_change(text, target):
    print('answer1 comment: ', text)
    items_dict[target]['annotations']['answer_pos_comment'] = text


def answer2_radio_change(choice, target):
    choice = choice.lower()
    preferred = 'NA'
    if 'answer1' in choice:
        preferred = 'answer_pos is considered better than answer_neg'
    elif 'answer2' in choice:
        preferred = 'answer_pos is considered worse than answer_neg'
    elif 'no preference' in choice:
        preferred = 'answer_pos and answer_pos are considered of similar quality'

    items_dict[target]['annotations']['answer_preference'] = preferred
    print('answer2 preferred: ', preferred)


def answer2_comment_change(text, target):
    print('answer2 comment: ', text)
    items_dict[target]['annotations']['answer_preference_comment'] = text


with demo:
    username = gr.Textbox(placeholder="Please enter your name here", label='')
    for i, item in enumerate(items):
        with gr.Accordion(f'{i + 1} / {len(items)}') as accordion:
            item_id = gr.State(item['id'])
            question = item['question']
            sources = item['sources']
            answer = item['answer']
            answer_neg = item['answer_negatives'][0]
            gr.HTML(f"""<p></p>
                <h4>Question:</h4>
                <h5>{question}</h5>
                <h4>References: </h4>
                <div>{compose_sources_html(sources)}</div>
                <h4>Answer1:</h4>
                <p>{answer}</p>
                <h5>Given the Question and the References, does Answer1 answer the question?</h5>
                <ul>
                    <li>Give 0 if you think it is a bad answer, e.g. it is just gibberish, or it is seemingly ok, but actually doesn't make sense, or it is just factually incorrect;</li>
                    <li>Give 1 if you think it is an OK answer, but some minor improvements can be made, e.g. a typo, an odd grammar, but it doesn't affect the whole answer that much, the answer still gives what you needed to answer that question;</li>
                    <li>Give 2 if you think it is a satisfying answer, not necessarily perfect, but definitely of "production quality", it answers the question fluently and correctly.</li>
                </ul>
            """)
            answer1_radio = gr.Radio(['0: Bad', '1: OK', '2: Good'], label='')
            answer1_radio.change(fn=answer1_radio_change, inputs=[answer1_radio, item_id])
            gr.Markdown("""
                #### Give reasons/comments if you scored 0 or 1 on Answer1. Any other remarks are very welcome too!
            """)
            answer1_comment = gr.Textbox(placeholder='', label="Feedback on Answer1")
            answer1_comment.change(fn=answer1_comment_change, inputs=[answer1_comment, item_id])
            gr.Markdown(f"""
                <h5>Now please have a look at an alternative answer to the same question and references</h5>
                <h4>Answer2:</h4>
                <p>{answer_neg}</p>
                <h5>Comparing the two answers:</h5>
            """)
            answer2_radio = gr.Radio(['I like Answer1 better',
                                      'I like Answer2 better',
                                      'I have no preference'], label='')
            answer2_radio.change(fn=answer2_radio_change, inputs=[answer2_radio, item_id])
            gr.HTML("""
                <h5>Share a few words on why you made the above choice, 
                especially if you chose the 3rd "no preference" option.</h5>
            """)
            answer2_comment = gr.Textbox(placeholder='', label="Feedback on Answer2")
            answer2_comment.change(fn=answer2_comment_change, inputs=[answer2_comment, item_id])

            # with gr.Row():
            #     prev_btn = gr.Button(value='Previous')
            #     prev_btn.click(prev_item)
            #     next_btn = gr.Button(value='Next')
            #     next_btn.click(next_item)

    submit_btn = gr.Button(value='Submit')
    submit_btn.click(submit, inputs=[username])

if __name__ == '__main__':
    demo.launch()

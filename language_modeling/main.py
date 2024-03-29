import os
os.environ['HF_HOME'] = '/Users/bo/workspace/data/HF_HOME'
print(os.environ['HF_HOME'])

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer





if __name__ == '__main__':
    eli5 = load_dataset("eli5", split="train_asks[:5000]")
    eli5 = eli5.train_test_split(test_size=0.2)
    train_example = eli5["train"][0]

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    eli5 = eli5.flatten()

    def preprocess_function(examples):
        answer_texts = examples["answers.text"]
        tokenizer_input = [" ".join(x) for x in answer_texts]
        tokenized = tokenizer(tokenizer_input)
        return tokenized


    print()
    tokenized_eli5 = eli5.map(
        preprocess_function,
        batched=True,
        batch_size=8,
        num_proc=4,
        remove_columns=eli5["train"].column_names,
    )

    block_size = 128


    def group_texts(examples):
        # Concatenate all texts.
        keys = examples.keys()
        concatenated_examples = {k: sum(examples[k], []) for k in keys}
        concatenated_examples_items = concatenated_examples.items()

        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of block_size.
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result


    lm_dataset = tokenized_eli5.map(group_texts, batched=True, num_proc=4)

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")

    training_args = TrainingArguments(
        output_dir="my_awesome_eli5_clm-model",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        push_to_hub=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["test"],
        data_collator=data_collator,
    )

    trainer.train()
    print()

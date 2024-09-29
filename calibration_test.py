from datasets import load_dataset
import pandas as pd
from openai import OpenAI
from together import Together
import json
from pydantic import BaseModel, Field

client = Together()

class QuestionResponse(BaseModel):
    reasoning: str = Field()
    answer: float = Field()
    confidence: float = Field()

def get_answer(row):
    return int(row['answer'].split('\n')[-1][5:])


df = pd.read_csv('results-llama-70B.csv')
dataset = load_dataset("openai/gsm8k", "main")
records = []

for i, row in enumerate(dataset['test']):
    completion = client.chat.completions.create(
        #model="gpt-3.5-turbo-1106",
        model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        response_format={ 
            "type": "json_object",
            "schema": QuestionResponse.model_json_schema(),
        },
        messages=[
            {"role": "system", "content": "You are a helpful assistant. You answer basic math questions and give confidence as a probability that your answer is correct (from 0 to 1). Only answer in JSON. You respond with json of the form \{'reasoning': REASONING,'answer': ANSWER, 'confidence': CONFIDENCE\}"},
            {
                "role": "user",
                "content": (
                    f"Example of a question you previously answered incorrectly with high confidence:\n"
                    f"Question: {df[(~df.correct) & (df.confidence == 1.0)].iloc[0].question}"
                    f"Your answer: {df[(~df.correct) & (df.confidence == 1.0)].iloc[0].answer}"
                    f"Your confidence: {df[(~df.correct) & (df.confidence == 1.0)].iloc[0].confidence}"
                    )
            },
            {
                "role": "user",
                "content": (
                    f"Example of a question you previously answered correctly with high confidence:\n"
                    f"Question: {df[(df.correct) & (df.confidence == 1.0)].iloc[0].question}"
                    f"Your answer: {df[(df.correct) & (df.confidence == 1.0)].iloc[0].answer}"
                    f"Your confidence: {df[(df.correct) & (df.confidence == 1.0)].iloc[0].confidence}"
                    )
            },
            {
                "role": "user",
                "content": (
                    f"Example of a question you previously answered incorrectly with lower confidence:\n"
                    f"Question: {df[(~df.correct) & (df.confidence < 1.0)].iloc[0].question}"
                    f"Your answer: {df[(~df.correct) & (df.confidence < 1.0)].iloc[0].answer}"
                    f"Your confidence: {df[(~df.correct) & (df.confidence < 1.0)].iloc[0].confidence}"
                    )
            },
            {
                "role": "user",
                "content": (
                    f"Example of a question you previously answered correctly with lower confidence:\n"
                    f"Question: {df[(df.correct) & (df.confidence < 1.0)].iloc[0].question}"
                    f"Your answer: {df[(df.correct) & (df.confidence < 1.0)].iloc[0].answer}"
                    f"Your confidence: {df[(df.correct) & (df.confidence < 1.0)].iloc[0].confidence}"
                    )
            },
            {
                "role": "user",
                "content": f"{row['question']}"
            }
        ]
    )
    try:
        record = json.loads(completion.choices[0].message.content)
        record['question'] = row['question']
        record['actual'] = get_answer(row)
        record['correct'] = abs(record['answer'] - get_answer(row)) < 0.0001
        records.append(record)
    except:
        print('ERROR')
    
    if i == 100:
        break

results = pd.DataFrame.from_records(records)
results.to_csv('results-test-llama-70B.csv')
print(results)
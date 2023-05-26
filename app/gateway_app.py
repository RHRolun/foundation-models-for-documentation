import requests
import json

finetuned_URL = "https://text-generation-webui-llm.apps.et-gpu.zfq7.p1.openshiftapps.com/run/textgen"

params = {
    'max_new_tokens': 500,
    'do_sample': True,
    'temperature': 0.7,
    'top_p': 0.73,
    'typical_p': 1,
    'repetition_penalty': 1.0,
    'encoder_repetition_penalty': 1.0,
    'top_k': 0,
    'min_length': 10,
    'no_repeat_ngram_size': 0,
    'num_beams': 1,
    'penalty_alpha': 0,
    'length_penalty': 0,
    'early_stopping': False,
    'seed': -1,
    'add_bos_token': True,
    'custom_stopping_strings': [],
    'truncation_length': 2048,
    'ban_eos_token': False,
    'skip_special_tokens': True,
}

def inference_finetuned_model(text: str):
    # payload = json.dumps([text, params]) 
    payload = {
        "inputs": [
            {
                "name": "dense_input", 
                "shape": [1, 7], 
                "datatype": "FP32",
                "data": [text]
            },
            ]
        }
    headers = {
        'content-type': 'application/json'
    }

    response = requests.post(finetuned_URL, json=payload, headers=headers)
    prediction = response.json()['outputs'][0]['data'][0]
    return prediction
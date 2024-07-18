from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("vyang/plc2proc")

model = AutoModelForSeq2SeqLM.from_pretrained("vyang/plc2proc")


def predict(sources, batch_size=8, topn=3, max_length=50):
    model.eval()  # 将模型转换为评估模式

    kwargs = {
        "num_beams": topn,
        "max_length": max_length,
        "no_repeat_ngram_size": 2,
        "num_return_sequences": topn,
        "early_stopping": True,
    }

    outputs = []
    for start in tqdm(range(0, len(sources), batch_size)):
        batch = sources[start : start + batch_size]

        input_tensor = tokenizer(
            batch, return_tensors="pt", truncation=True, padding=True, max_length=512
        ).input_ids  # .cuda()

        outputs.extend(model.generate(input_ids=input_tensor, **kwargs))
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

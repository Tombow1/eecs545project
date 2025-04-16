# summarizer.py
from transformers import pipeline, BartTokenizer
import tiktoken

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", tokenizer=tokenizer, device=0)
MAX_HISTORY_TOKENS = 3000

def count_tokens(text, model="facebook/bart-large-cnn"):
    if model == "facebook/bart-large-cnn":
        return len(tokenizer.tokenize(text))
    else:
        return len(tiktoken.encoding_for_model(model).encode(text))

def approximate_history_token_count(messages, model="facebook/bart-large-cnn"):
    return sum(count_tokens(m["content"], model) for m in messages)

def chunked_summarize(text, chunk_size=512, pass_count=2):
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    for _ in range(pass_count):
        chunks = [token_ids[i:i+chunk_size] for i in range(0, len(token_ids), chunk_size)]
        partials = []
        for chunk in chunks:
            try:
                decoded = tokenizer.decode(chunk, skip_special_tokens=True)
                out = summarizer(decoded, do_sample=False, min_length=30, max_length=120)
                partials.append(out[0]["summary_text"])
            except:
                partials.append(decoded)
        token_ids = tokenizer.encode("\n".join(partials), add_special_tokens=False)
        if len(chunks) == 1:
            break
    return tokenizer.decode(token_ids, skip_special_tokens=True)

def prune_history(messages, model="facebook/bart-large-cnn"):
    if approximate_history_token_count(messages, model) <= MAX_HISTORY_TOKENS:
        return messages
    system = messages[0]
    recent = messages[1:]
    tokens_to_remove = approximate_history_token_count(messages) - MAX_HISTORY_TOKENS
    cutoff = 0
    acc = 0
    for i, msg in enumerate(recent):
        acc += count_tokens(msg["content"], model)
        if acc >= tokens_to_remove:
            cutoff = i
            break
    if cutoff > 0:
        summary = chunked_summarize("\n".join(msg["content"] for msg in recent[:cutoff]))
        return [system, {"role": "assistant", "content": summary}] + recent[cutoff:]
    return messages

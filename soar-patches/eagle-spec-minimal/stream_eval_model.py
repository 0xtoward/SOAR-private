import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import tqdm


def load_eval_helpers(toolkit_dir):
    sys.path.insert(0, toolkit_dir)
    import eval_model as official_eval  # noqa: PLC0415

    return official_eval


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--api_base", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--max_seq_len", type=int, default=262144)
    parser.add_argument("--max_out_len", type=int, default=32768)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--toolkit_dir", default="/root/autodl-tmp/SOAR-Toolkit")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def call_sglang_api(api_base, model, prompt, sampling_kwargs, timeout=3000):
    url = f"{api_base}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    payload.update(sampling_kwargs)
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    result = resp.json()
    return result["choices"][0]["message"]["content"], result.get("usage", {})


def score_one(official_eval, model, item, pred):
    task = item.get("task", "unknown")
    gold = item.get("gold")
    score = 0
    extracted = None

    if task == "mcq":
        score, extracted = official_eval.score_mcq(pred, gold)
    elif task in ["niah", "cwe", "fwe", "qa", "lcx"]:
        score = official_eval.score_exact_match(pred, gold, task)
    elif isinstance(gold, str) and gold.lower() in pred.lower():
        score = 1

    input_tokens = model.get_token_len(item["question"])
    output_tokens = model.get_token_len(pred)
    return score, extracted, input_tokens, output_tokens


def write_summary(path, payload):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp_path, path)


def main():
    args = parse_args()
    official_eval = load_eval_helpers(args.toolkit_dir)

    if not args.model_name:
        resp = requests.get(f"{args.api_base}/v1/models", timeout=10)
        resp.raise_for_status()
        args.model_name = resp.json()["data"][0]["id"]
        print(f"Auto-detected model name: {args.model_name}", flush=True)

    dataset = []
    with open(args.data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
                if args.num_samples and len(dataset) >= args.num_samples:
                    break

    os.makedirs(args.output_dir, exist_ok=True)
    stream_path = os.path.join(args.output_dir, "predictions.stream.jsonl")
    final_path = os.path.join(args.output_dir, "predictions.jsonl")
    summary_stream_path = os.path.join(args.output_dir, "summary.stream.json")
    summary_final_path = os.path.join(args.output_dir, "summary.json")

    print(f"API Base: {args.api_base}", flush=True)
    print(f"Model Name: {args.model_name}", flush=True)
    print(f"Model Path: {args.model_path}", flush=True)
    print(f"Data Path: {args.data_path}", flush=True)
    print(f"Saving streaming results to {args.output_dir}", flush=True)
    print(f"Testing with {len(dataset)} samples.", flush=True)

    model = official_eval.SGLANGwithChatTemplate(
        path=args.model_path,
        api_base=args.api_base,
        model_name=args.model_name,
        max_seq_len=args.max_seq_len,
        concurrency=args.concurrency,
        generation_kwargs={"temperature": 0.0},
        chat_template_kwargs={"enable_thinking": True},
        mode="mid",
    )

    sampling_kwargs = {
        "temperature": 0.0,
        "max_tokens": args.max_out_len,
        "stop": list(set(model.stop_words)),
    }

    print(
        f"Sending {len(dataset)} requests to SGLang API "
        f"(concurrency={args.concurrency}, max_out_len={args.max_out_len})...",
        flush=True,
    )

    start_time = time.time()
    correct_count = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    results = [None] * len(dataset)

    def infer(idx):
        content, usage = call_sglang_api(
            args.api_base, args.model_name, dataset[idx]["question"], sampling_kwargs
        )
        return idx, content, usage

    with open(stream_path, "w", encoding="utf-8") as stream_f:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {executor.submit(infer, i): i for i in range(len(dataset))}
            for completed, future in enumerate(
                tqdm.tqdm(as_completed(futures), total=len(dataset), desc="Generating"),
                start=1,
            ):
                idx, pred, usage = future.result()
                item = dataset[idx]
                score, extracted, input_tokens, output_tokens = score_one(
                    official_eval, model, item, pred
                )
                correct_count += score
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens

                elapsed = time.time() - start_time
                partial_acc_done = (correct_count / completed) * 100
                partial_acc_total = (correct_count / len(dataset)) * 100
                tps = total_output_tokens / elapsed if elapsed > 0 else 0.0

                result = {
                    "index": idx,
                    "task": item.get("task", "unknown"),
                    "question": item["question"],
                    "gold": item.get("gold"),
                    "prediction": pred,
                    "score": score,
                    "extracted": extracted,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "usage": usage,
                    "completed": completed,
                    "partial_accuracy_done": partial_acc_done,
                    "partial_accuracy_total": partial_acc_total,
                    "elapsed": elapsed,
                    "stream_tps": tps,
                }
                results[idx] = result
                stream_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream_f.flush()

                summary = {
                    "completed": completed,
                    "num_samples": len(dataset),
                    "correct_count": correct_count,
                    "partial_accuracy_done": round(partial_acc_done, 4),
                    "partial_accuracy_total": round(partial_acc_total, 4),
                    "elapsed": elapsed,
                    "total_input_tokens": total_input_tokens,
                    "total_output_tokens": total_output_tokens,
                    "stream_tps": tps,
                    "latest_index": idx,
                    "latest_task": item.get("task", "unknown"),
                    "latest_score": score,
                    "latest_output_tokens": output_tokens,
                }
                write_summary(summary_stream_path, summary)

                print(
                    "[stream] "
                    f"done={completed}/{len(dataset)} "
                    f"idx={idx} task={item.get('task', 'unknown')} "
                    f"score={score} acc_done={partial_acc_done:.2f}% "
                    f"acc_total={partial_acc_total:.2f}% "
                    f"out_tok={output_tokens} tps={tps:.2f}",
                    flush=True,
                )

                if args.verbose:
                    print(
                        f"Gold: {item.get('gold')}, Extracted: {extracted}",
                        flush=True,
                    )

    duration = time.time() - start_time
    final_results = [res for res in results if res is not None]
    final_results.sort(key=lambda x: x["index"])
    avg_score = (correct_count / len(dataset)) * 100 if dataset else 0.0
    final_tps = total_output_tokens / duration if duration > 0 else 0.0

    with open(final_path, "w", encoding="utf-8") as f:
        for res in final_results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    final_summary = {
        "ori_accuracy": round(avg_score, 2),
        "overall_accuracy": min(round(avg_score / 80 * 100, 2), 100),
        "duration": duration,
        "num_samples": len(dataset),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "tps": final_tps,
    }
    write_summary(summary_final_path, final_summary)

    with open(os.path.join(args.output_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Data: {args.data_path}\n")
        f.write(f"Original Accuracy: {avg_score:.2f}%\n")
        f.write(f"Normalized Accuracy: {min(round(avg_score / 80 * 100, 2), 100)}%\n")
        f.write(f"Num Samples: {len(dataset)}\n")
        f.write(f"Total Duration: {duration:.2f} s\n")
        f.write(f"Total Output Tokens: {total_output_tokens}\n")
        f.write(f"TPS: {final_tps:.2f}\n")

    print(f"\nAverage Score: {avg_score:.2f}%", flush=True)
    print(f"Total Duration: {duration:.2f} s", flush=True)
    print(f"Total Tokens: In={total_input_tokens}, Out={total_output_tokens}", flush=True)
    print(f"Overall TPS (Output): {final_tps:.2f} tokens/s", flush=True)
    print(f"Detailed results saved to {final_path}", flush=True)


if __name__ == "__main__":
    main()

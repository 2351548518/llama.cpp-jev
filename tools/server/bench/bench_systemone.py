"""Replay local System One requests and measure decision latency, failures and output drift."""
import argparse
import base64
import hashlib
import http.client
import json
import math
import mimetypes
import statistics
import time
from pathlib import Path
from urllib.parse import urlsplit


def request(url, body, timeout):
    address = urlsplit(url)
    connection_type = http.client.HTTPSConnection if address.scheme == "https" else http.client.HTTPConnection
    connection = connection_type(address.hostname, address.port, timeout=timeout)
    try:
        connection.request("POST", address.path.rstrip("/") + "/v1/systemone",
                           json.dumps(body, ensure_ascii=False).encode(), {"Content-Type": "application/json"})
        response = connection.getresponse()
        data = response.read()
        if response.status != 200:
            raise RuntimeError("HTTP " + str(response.status) + ": " + data[:300].decode(errors="replace"))
        return json.loads(data)
    finally:
        connection.close()


def validate(answer):
    if answer["usage"]["output_tokens"] != 0:
        raise ValueError("A decision request generated tokens")
    for result in answer["answers"].values():
        kind = result["type"]
        if kind == "box":
            box = result["box"]
            if not (0 <= box["left"] < box["right"] <= 1 and 0 <= box["top"] < box["bottom"] <= 1):
                raise ValueError("Invalid normalized box")
        elif kind == "noul":
            if not 0 <= result["noul"] <= 1:
                raise ValueError("Invalid noul probability")
        else:
            values = list(result["probabilities"].values())
            if not values or not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
                raise ValueError("Invalid decision probabilities")
            if abs(sum(values) - 1) > 1e-5:
                raise ValueError("Decision probabilities are not normalized")
            if kind == "score" and (not math.isfinite(result["score"]) or not 0 <= result["score"] <= len(values) - 1):
                raise ValueError("Invalid expected score")
        if not math.isfinite(result["confidence"]) or not 0 <= result["confidence"] <= 1:
            raise ValueError("Invalid confidence")


def maximum_difference(left, right):
    if isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys():
        return max((maximum_difference(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, bool) or isinstance(right, bool):
        return 0.0 if type(left) is type(right) and left == right else math.inf
    if isinstance(left, (float, int)) and isinstance(right, (float, int)):
        return abs(left - right) if math.isfinite(left) and math.isfinite(right) else math.inf
    return 0.0 if left == right else math.inf


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] * (high - position) + ordered[high] * (position - low) if low != high else ordered[low]


def load_cases(path):
    cases = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        body = item["body"]
        for name in item.get("media_paths", []):
            media = Path(name)
            if not media.is_absolute():
                media = path.parent / media
            mime = mimetypes.guess_type(str(media))[0] or "application/octet-stream"
            body.setdefault("files", []).append("data:" + mime + ";base64," + base64.b64encode(media.read_bytes()).decode())
        # State keys and candidate order are part of the tokenized model input.
        digest = hashlib.sha256(json.dumps(body, ensure_ascii=False).encode()).hexdigest()
        cases.append({"id": item["id"], "kind": item.get("kind", "unspecified"), "body": body, "request_sha256": digest})
    if not cases or len({item["id"] for item in cases}) != len(cases):
        raise ValueError("Provide a nonempty manifest with unique case IDs")
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--inputs", type=Path, required=True, help="JSONL rows: id, body, optional kind and media_paths")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--cache-prompt", action="store_true", help="Allow prompt-prefix reuse (disabled by default)")
    parser.add_argument("--compare", type=Path, help="Reference report from the same inputs and model weights")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--device-label", default="unspecified", help="Operator-provided server hardware/backend label")
    parser.add_argument("--server-pid", type=int, help="On Linux, record the server's peak resident memory")
    args = parser.parse_args()
    if (args.repeat < 1 or args.warmup < 0 or not math.isfinite(args.timeout) or args.timeout <= 0 or
            not math.isfinite(args.tolerance) or args.tolerance < 0):
        parser.error("Invalid repeat, warmup, timeout or tolerance")
    cases = load_cases(args.inputs)
    reference = json.loads(args.compare.read_text())["cases"] if args.compare else None
    results = {}
    for item in cases:
        body = dict(item["body"], cache_prompt=args.cache_prompt)
        elapsed, errors, answer, drift = [], [], None, 0.0
        for index in range(args.warmup + args.repeat):
            start = time.perf_counter()
            try:
                value = request(args.url, body, args.timeout)
                duration = time.perf_counter() - start
                validate(value)
                if answer is not None:
                    drift = max(drift, maximum_difference(answer, value["answers"]))
                answer = value["answers"]
                if index >= args.warmup:
                    elapsed.append(duration * 1000)
            except (OSError, ValueError, KeyError, RuntimeError, http.client.HTTPException) as error:
                errors.append({"iteration": index, "warmup": index < args.warmup, "error": str(error)})
        if reference is not None:
            expected = reference[item["id"]]
            if expected["request_sha256"] != item["request_sha256"]:
                raise ValueError("Reference inputs differ: " + item["id"])
            if answer is not None and expected["answers"] is not None:
                drift = max(drift, maximum_difference(expected["answers"], answer))
        results[item["id"]] = {"kind": item["kind"], "request_sha256": item["request_sha256"],
                               "answers": answer, "latency_ms": elapsed, "errors": errors,
                               "maximum_answer_difference": drift if math.isfinite(drift) else None,
                               "within_tolerance": drift <= args.tolerance}
        print(item["id"], "successful", len(elapsed), "errors", len(errors), flush=True)
    latencies = [duration for item in results.values() for duration in item["latency_ms"]]
    planned = len(cases) * args.repeat
    report = {"protocol": "Sequential end-to-end decision requests; warmups excluded from latency; failures retained in success rate",
              "request_hash_scheme": "ordered-systemone-v1", "manifest_sha256": hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
              "cache_prompt": args.cache_prompt, "repeat": args.repeat, "warmup": args.warmup,
              "planned": planned, "successful": len(latencies), "success_rate": len(latencies) / planned,
              "warmup_failures": sum(error["warmup"] for item in results.values() for error in item["errors"]),
              "p50_ms": percentile(latencies, 0.5), "p95_ms": percentile(latencies, 0.95),
              "mean_ms": statistics.mean(latencies) if latencies else None, "cases": results,
              "comparison_tolerance": args.tolerance, "device_label": args.device_label}
    if args.server_pid is not None:
        status = Path("/proc") / str(args.server_pid) / "status"
        if status.exists():
            for line in status.read_text().splitlines():
                if line.startswith("VmHWM:"):
                    report["server_peak_rss_bytes"] = int(line.split()[1]) * 1024
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if len(latencies) != planned or report["warmup_failures"] or not all(item["within_tolerance"] for item in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

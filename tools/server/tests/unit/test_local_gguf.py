import base64
import json
import os

import pytest
import requests
from utils import ServerProcess


@pytest.fixture(scope="session", autouse=True)
def load_server_presets():
    # These tests use only the GGUFs supplied through their environment.
    pass


def local_server():
    server = ServerProcess()
    server.model_hf_repo = None
    server.model_hf_file = None
    server.offline = True
    server.jinja = True
    server.cache_prompt = False
    server.n_threads = 2
    server.n_ctx = 2048
    server.n_batch = 1024
    server.n_ubatch = 512
    server.n_slots = 1
    return server


@pytest.mark.parametrize("weight_env", ["LLAMA_QWEN35_GGUF", "LLAMA_GEMMA4_GGUF"])
def test_generative_gguf(weight_env):
    model = os.environ.get(weight_env)
    if not model:
        pytest.skip(f"set {weight_env} to a local generative GGUF")  # ty: ignore[too-many-positional-arguments]
    server = local_server()
    server.model_file = model
    server.start()

    reply = server.make_request("POST", "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "temperature": 0,
        "max_tokens": 32,
    })
    assert reply.status_code == 200
    assert reply.body["usage"]["completion_tokens"] > 0
    request = {"state": "Hello.", "questions": {"q": {"type": "noul", "instructions": "Is this a greeting?"}}}
    assert server.make_request("POST", "/v1/systemone", request).status_code == 501


@pytest.mark.parametrize("router", [False, True], ids=["single", "router"])
def test_laya_assets(tmp_path, monkeypatch, router):
    model = os.environ.get("LLAMA_LAYA_GGUF")
    projector = os.environ.get("LLAMA_LAYA_MMPROJ")
    if not model or not projector:
        pytest.skip("requires a local Laya decision GGUF and its matching mmproj")  # ty: ignore[too-many-positional-arguments]
    server = local_server()
    monkeypatch.delenv("LAYA_JEV_STRICT", raising=False)
    if router:
        preset = tmp_path / "models.ini"
        preset.write_text(f"version = 1\n[jev]\nmodel = {model}\nmmproj = {projector}\n", encoding="utf-8")
        server.models_preset = str(preset)
        server.models_max = 1
    else:
        server.model_file = model
        server.model_alias = "jev"
        monkeypatch.setenv("LLAMA_ARG_MMPROJ", projector)
    server.start()

    chat = {"model": "jev", "messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 8}
    assert server.make_request("POST", "/v1/chat/completions", chat).status_code == 501

    black = "iVBORw0KGgoAAAANSUhEUgAAARgAAAEYAQAAAACv9dQhAAAAIElEQVR42u3BMQEAAADCoPVP7W8GoAAAAAAAAAAAAIA3J2AAAdzYCbkAAAAASUVORK5CYII="
    white = ("iVBORw0KGgoAAAANSUhEUgAAARgAAAEYAQAAAACv9dQhAAAAOUlEQVR42u3KQQEAAAQEsKN/ZyIIY"
             "HuvJqeO4ziO4ziO4ziO4ziO4ziO4ziO4ziO4ziO4ziO4/w6C3TPAy9+kZggAAAAAElFTkSuQmCC")
    header = {"TypeSafe-Danger-Zone-Images": "true"}
    body = {
        "model": "jev",
        "state": "Two alternatives.",
        "questions": {"pick": {"type": "choice", "instructions": "Choose the lighter image.",
                               "criteria": {"A": {"image": "black-ref"}, "B": {"image": "white-ref"}}}},
        "assets": {
            "black-ref": {"format": "png", "data_base64": black},
            "white-ref": {"format": "png", "data_base64": white},
        },
    }
    assert server.make_request("POST", "/v1/systemone", body).status_code == 400
    response = server.make_request("POST", "/v1/systemone", body, headers=header)
    assert response.status_code == 200
    pick = response.body["answers"]["pick"]
    assert set(pick) == {"type", "choice", "probabilities", "confidence"}
    assert set(pick["probabilities"]) == {"A", "B"}

    inline = requests.post(server.make_url("/v1/systemone"), headers=header, timeout=60,
                           files=[("request", (None, json.dumps(body), "application/json"))])
    assert inline.status_code == 200
    assert inline.json() == response.body

    criteria_only = {
        "model": "jev",
        "state": ["black-ref", "Inspect the image."],
        "questions": {"visible": {"type": "noul", "criteria": {"true": "The image is black."}}},
        "assets": {"black-ref": body["assets"]["black-ref"]},
    }
    response = server.make_request("POST", "/v1/systemone", criteria_only, headers=header)
    assert response.status_code == 200
    visible = response.body["answers"]["visible"]
    assert set(visible) == {"type", "noul"}
    assert visible["type"] == "noul" and 0 <= visible["noul"] <= 1
    criteria_only["questions"]["visible"]["criteria"] = {"true": " "}
    assert server.make_request("POST", "/v1/systemone", criteria_only, headers=header).status_code == 400

    long_choice = {"type": "choice", "instructions": "Which criterion agrees with the state?",
                   "criteria": {"A": "The state reports a duplicate charge.", "B": "The transaction was successful. " * 220}}
    response = server.make_request("POST", "/v1/systemone", {
        "model": "jev", "state": "I was charged twice.", "questions": {"long": long_choice},
    })
    assert response.status_code == 200
    assert response.body["usage"]["input_tokens"] > server.n_batch

    question = {"target": {"type": "box", "instructions": "Locate the darker region."}}
    route = server.make_request("POST", "/v1/multimodal", {
        "model": "jev", "state": "Inspect the image.", "questions": question,
        "image": {"base64": black}, "min_confidence": {"default": 1.0},
    })
    assert route.status_code == 200
    assert route.body["usage"]["forward_passes"] == 1
    assert route.body["usage"]["truncated"] is False
    assert route.body["answers"]["target"]["abstention_threshold"] == 1.0

    multipart = {"model": "jev", "state": ["image-ref", "Inspect the image."], "questions": question,
                 "assets": {"image-ref": {"part": "image"}}}
    uploaded = requests.post(server.make_url("/v1/systemone"), headers=header, timeout=60, files=[
        ("request", ("request.json", json.dumps(multipart), "Application/JSON; charset=utf-8")),
        ("image", ("black.png", base64.b64decode(black), "image/png")),
    ])
    assert uploaded.status_code == 200
    assert uploaded.json()["answers"]["target"]["box"] == pytest.approx(route.body["answers"]["target"]["box"], abs=3e-5)

    two_views = [{"image": "black-ref"}, {"image": "white-ref"}]
    boxes = {
        "first": {"type": "box", "instructions": "Locate the darker region in `views[0].image`."},
        "second": {"type": "box", "instructions": "Locate the brighter region in `views[1].image`."},
    }
    with_two_images = {"model": "jev", "state": {"views": two_views}, "questions": boxes, "assets": body["assets"]}
    multi = server.make_request("POST", "/v1/systemone", with_two_images, headers=header)
    assert multi.status_code == 200
    for index, (qid, box_question) in enumerate(boxes.items()):
        ref = two_views[index]["image"]
        views = [view if i == index else {"image": ""} for i, view in enumerate(two_views)]
        control = {"model": "jev", "state": {"views": views}, "questions": {qid: box_question},
                   "assets": {ref: body["assets"][ref]}}
        single = server.make_request("POST", "/v1/systemone", control, headers=header)
        assert single.status_code == 200
        assert multi.body["answers"][qid]["box"] == pytest.approx(single.body["answers"][qid]["box"], abs=3e-5)

    ambiguous = {**with_two_images, "questions": question}
    assert server.make_request("POST", "/v1/systemone", ambiguous, headers=header).status_code == 400

    criteria = [{"level": "dark", "example": "black-ref"}, {"level": "bright", "example": "white-ref"}]
    score = {
        "model": "jev",
        "state": "An example to rate.",
        "questions": {"rating": {"type": "score", "instructions": "Which level is the closer match?", "criteria": criteria}},
        "assets": body["assets"],
    }
    legacy = server.make_request("POST", "/v1/systemone", score, headers={**header, "X-Laya-Response-Schema": "legacy"})
    assert legacy.status_code == 200
    legend = legacy.body["answers"]["rating"]["legend"]
    assert [json.loads(legend[str(i)]) for i in range(len(criteria))] == criteria

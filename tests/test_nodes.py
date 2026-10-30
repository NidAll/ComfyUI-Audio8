import asyncio
import importlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pytest

COMFYUI = Path(__file__).resolve().parents[3]
PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMFYUI))
sys.path.insert(0, str(PACKAGE))

from comfy.cli_args import args

args.cpu = True


@pytest.fixture(scope="module")
def audio8_nodes():
    package_name = "audio8_node_contract_tests"
    spec = importlib.util.spec_from_file_location(
        package_name,
        PACKAGE / "__init__.py",
        submodule_search_locations=[str(PACKAGE)],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    spec.loader.exec_module(package)
    extension = asyncio.run(package.comfy_entrypoint())
    classes = asyncio.run(extension.get_node_list())
    module = importlib.import_module(f"{package_name}.nodes")
    return module, classes


def _schema_map(nodes, monkeypatch):
    module, classes = nodes
    monkeypatch.setattr(
        module.folder_paths,
        "get_filename_list",
        lambda folder: ["audio8-realistic.safetensors", "ignore.bin"],
    )
    schemas = {}
    for node in classes:
        schema = node.define_schema()
        schemas[schema.node_id] = schema
    return schemas


def test_extension_entrypoint_registers_v3_nodes(audio8_nodes):
    _, classes = audio8_nodes

    assert [node.define_schema().node_id for node in classes] == [
        "Audio8Loader", "Audio8Voice", "Audio8Synthesize", "Audio8Quantize",
        "Audio8LongSpeech", "Audio8ASRLoader", "Audio8Transcribe"
    ]
    for node in classes:
        assert node.VALIDATE_CLASS() is None
        node.define_schema().validate()
        json.dumps(node.GET_NODE_INFO_V1())


def test_v3_schema_inputs_match_execute_parameters(audio8_nodes, monkeypatch):
    _, classes = audio8_nodes
    schemas = _schema_map(audio8_nodes, monkeypatch)

    for node in classes:
        schema = schemas[node.define_schema().node_id]
        inputs = {item.id: item for item in schema.inputs}
        signature = inspect.signature(node.execute)
        assert list(signature.parameters) == list(inputs)
        for name, item in inputs.items():
            parameter = signature.parameters[name]
            if item.optional:
                assert parameter.default is None
            else:
                assert parameter.default is inspect.Parameter.empty

    synth = schemas["Audio8Synthesize"]
    assert next(item for item in synth.inputs if item.id == "acceleration").options == ["auto", "eager"]
    assert next(item for item in synth.inputs if item.id == "voice").optional
    assert schemas["Audio8Quantize"].is_output_node
    assert [item.id for item in schemas["Audio8Quantize"].outputs] == ["report"]


def test_api_workflows_follow_node_schemas_and_connect_audio8_outputs(audio8_nodes, monkeypatch):
    schemas = _schema_map(audio8_nodes, monkeypatch)

    for path in sorted((PACKAGE / "example_workflows").glob("*_api.json")):
        graph = json.loads(path.read_text(encoding="utf-8"))
        audio_nodes = {
            node_id: node for node_id, node in graph.items()
            if node.get("class_type") in schemas
        }
        assert audio_nodes, path.name
        for node_id, node_data in audio_nodes.items():
            schema = schemas[node_data["class_type"]]
            declared = {item.id: item for item in schema.inputs}
            supplied = node_data["inputs"]
            assert set(supplied) <= set(declared), (path.name, node_data["class_type"])
            assert {name for name, item in declared.items() if not item.optional} <= set(supplied)
            for input_name, value in supplied.items():
                if not (isinstance(value, list) and len(value) == 2):
                    continue
                source_id, output_slot = str(value[0]), value[1]
                assert source_id in graph, (path.name, node_id, input_name)
                source_data = graph[source_id]
                source_class = source_data.get("class_type")
                if source_class in schemas:
                    source_schema = schemas[source_class]
                    assert isinstance(output_slot, int)
                    assert 0 <= output_slot < len(source_schema.outputs)
                    assert input_name in declared

    synth_schema = schemas["Audio8Synthesize"]
    assert next(item for item in synth_schema.inputs if item.id == "acceleration").options == ["auto", "eager"]
    speech = json.loads((PACKAGE / "example_workflows/01_speech_api.json").read_text())
    assert speech["2"]["inputs"]["acceleration"] == "auto"
    clone = json.loads((PACKAGE / "example_workflows/02_voice_clone_api.json").read_text())
    assert clone["2"]["inputs"]["voice"] == ["5", 0]
    assert clone["5"]["class_type"] == "Audio8Voice"
    multilingual = json.loads((PACKAGE / "example_workflows/03_multilingual_batch_api.json").read_text())
    text = multilingual["2"]["inputs"]["text"]
    assert all(language in text for language in ("Bonjour", "مرحباً", "你好"))
    assert multilingual["2"]["inputs"]["batch_lines"] is True
    assert multilingual["2"]["inputs"]["acceleration"] == "auto"


def test_model_resolution_requires_a_listed_safetensors_model(audio8_nodes, monkeypatch):
    module, _ = audio8_nodes
    monkeypatch.setattr(module.folder_paths, "get_filename_list", lambda folder: ["ok.safetensors"])
    monkeypatch.setattr(
        module.folder_paths,
        "get_full_path_or_raise",
        lambda folder, name: f"/models/{folder}/{name}",
    )

    assert module.resolve_model("ok.safetensors") == "/models/audio8/ok.safetensors"
    for name in ("../outside.safetensors", "nested/ok.safetensors", "wrong.bin"):
        with pytest.raises(ValueError, match="Select an Audio8 safetensors"):
            module.resolve_model(name)


@pytest.mark.parametrize(
    "name",
    ["../escape.safetensors", "nested/output.safetensors", "..\\escape.safetensors",
     "/tmp/escape.safetensors", "output.bin"],
)
def test_quantizer_rejects_output_names_that_escape_model_directory(audio8_nodes, monkeypatch, name):
    module, _ = audio8_nodes

    def forbidden_export(*args, **kwargs):
        pytest.fail("unsafe output name reached artifact.export")

    monkeypatch.setattr(module.artifact, "export", forbidden_export)
    with pytest.raises(ValueError, match="without directories"):
        module.Audio8Quantize.execute("source.safetensors", name, "int8")


def test_quantizer_passes_only_rooted_output_to_export(audio8_nodes, monkeypatch, tmp_path):
    module, _ = audio8_nodes
    root = tmp_path / "models" / "audio8"
    root.mkdir(parents=True)
    captured = {}

    class Progress:
        def __init__(self, total):
            captured["progress_total"] = total

        def update_absolute(self, value, total):
            captured["progress"] = (value, total)

    def fake_export(source, output, *, profile, progress):
        captured.update(source=source, output=Path(output), profile=profile)
        progress(1, 1)
        return {"written": output.name}

    monkeypatch.setattr(module.folder_paths, "get_folder_paths", lambda folder: [str(root)])
    monkeypatch.setattr(module, "resolve_model", lambda name: str(root / name))
    monkeypatch.setattr(module.comfy.utils, "ProgressBar", Progress)
    monkeypatch.setattr(module.artifact, "export", fake_export)

    result = module.Audio8Quantize.execute("source.safetensors", "result.safetensors", "int8")

    assert result.result == (json.dumps({"written": "result.safetensors"}, indent=2),)
    assert captured["output"] == root / "result.safetensors"
    assert captured["output"].parent == root
    assert captured["source"] == str(root / "source.safetensors")
    assert captured["profile"] == "int8"
    assert captured["progress_total"] == 1
    assert captured["progress"] == (1, 1)
    assert not (root / "result.safetensors").exists()


def test_synthesize_execute_splits_batch_lines_and_forwards_acceleration(audio8_nodes, monkeypatch):
    module, _ = audio8_nodes
    captured = {}

    class Progress:
        def __init__(self, total):
            captured["progress_total"] = total

        def update_absolute(self, step, total):
            captured["progress"] = (step, total)

    def fake_synthesize(model, texts, **kwargs):
        captured.update(model=model, texts=texts, **kwargs)
        kwargs["callback"](2, 4)
        return {"waveform": "fake"}, {"frames": [2]}

    monkeypatch.setattr(module.comfy.utils, "ProgressBar", Progress)
    monkeypatch.setattr(module.mm, "throw_exception_if_processing_interrupted", lambda: None)
    monkeypatch.setattr(module, "synthesize", fake_synthesize)

    result = module.Audio8Synthesize.execute(
        "model", " Bonjour \n\n مرحباً\n", 7, 64, 0.7, 0.9, 12, True,
        "eager", True, voice={"codes": "voice"},
    )

    assert result.result[0] == {"waveform": "fake"}
    assert json.loads(result.result[1]) == {"frames": [2]}
    assert captured["model"] == "model"
    assert captured["texts"] == ["Bonjour", "مرحباً"]
    assert captured["reference"] == {"codes": "voice"}
    assert captured["acceleration"] == "eager"
    assert captured["progress_total"] == 64
    assert captured["progress"] == (2, 4)

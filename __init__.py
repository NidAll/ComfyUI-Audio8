from comfy_api.latest import ComfyExtension


class Audio8Extension(ComfyExtension):
    async def get_node_list(self):
        # Defer Comfy's device initialization until the extension is loaded;
        # importing the standalone conversion/test package needs no GPU.
        from .nodes import (Audio8Loader, Audio8Voice, Audio8Synthesize, Audio8Quantize,
                            Audio8LongSpeech, Audio8ASRLoader, Audio8Transcribe)
        return [Audio8Loader, Audio8Voice, Audio8Synthesize, Audio8Quantize,
                Audio8LongSpeech, Audio8ASRLoader, Audio8Transcribe]


async def comfy_entrypoint():
    return Audio8Extension()

"""Names of the special tokens of trained tokenizers."""

#: Marks the end of a document; also the stop token for generation.
EOS_TOKEN = "<|endoftext|>"
#: Replaces corrupted tokens in the diffusion-style paradigms. Added only when a variant needs it.
MASK_TOKEN = "[MASK]"

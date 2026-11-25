"""The pre-tokenizer regex used by tiktoken's ``cl100k_base`` (GPT-4)."""

# Splits text into "pre-tokens" before byte-pair merging, so that merges never join
# unrelated pieces (a word and the punctuation after it, letters and digits, ...).
#   contractions 's 'd 'm 't 'll 've 're | optional-prefix + letters | up to 3 digits
#   | punctuation run (+ newlines) | whitespace runs
# Digits are grouped by at most three, and spaces attach to the *next* word (" the").
CL100K_PATTERN = (
    r"'(?i:[sdmt]|ll|ve|re)"
    r"|[^\r\n\p{L}\p{N}]?+\p{L}+"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]++[\r\n]*"
    r"|\s*[\r\n]"
    r"|\s+(?!\S)"
    r"|\s+"
)

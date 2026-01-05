# Attention Sinks: A 'Catch, Tag, Release' Mechanism for Embeddings

> Stephen Zhang, Mustafa Khan, Vardan Papyan

![111](../../blank.jpg)

## Abstract

Large language models (LLMs) often concentrate their attention on a few specific tokens referred to as attention sinks. Common examples include the first token, a prompt-independent sink, and punctuation tokens, which are prompt-dependent. While the tokens causing the sinks often lack direct semantic meaning, the presence of the sinks is critical for model performance, particularly under model compression and KV-caching. Despite their ubiquity, the function, semantic role, and origin of attention sinks -- especially those beyond the first token -- remain poorly understood. In this work, we conduct a comprehensive investigation demonstrating that attention sinks: catch a sequence of tokens, tag them using a common direction in embedding space, and release them back into the residual stream, where tokens are later retrieved based on the tags they have acquired. Probing experiments reveal these tags carry semantically meaningful information, such as the truth of a statement. These findings extend to reasoning models, where the mechanism spans more heads and explains greater variance in embeddings, or recent models with query-key normalization, where sinks remain just as prevalent. To encourage future theoretical analysis, we introduce a minimal problem which can be solved through the 'catch, tag, release' mechanism, and where it emerges through training.

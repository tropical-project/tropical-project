from vllm.tracing import BaseSpanAttributes


class DistserveSpanAttributes(BaseSpanAttributes):
    # The following span attribute names are added here because they are missing
    # from the Semantic Conventions for LLM.
    LLM_REQUEST_ID = "gen_ai.request.id"
    LLM_REQUEST_BEST_OF = "gen_ai.request.best_of"
    LLM_REQUEST_N = "gen_ai.request.n"
    LLM_USAGE_NUM_SEQUENCES = "gen_ai.usage.num_sequences"
    LLM_LATENCY_TIME_IN_QUEUE = "gen_ai.latency.time_in_queue"
    LLM_LATENCY_TIME_TO_FIRST_TOKEN = "gen_ai.latency.time_to_first_token"
    LLM_LATENCY_TIME_TO_PREFILL = "gen_ai.latency.time_to_prefill"
    LLM_LATENCY_E2E = "gen_ai.latency.e2e"
    LLM_LATENCY_MIGRATION = "gen_ai.latency.migration"
    LLM_LATENCY_TOKEN_BY_TOKEN = "gen_ai.latency_token_by_token"
    LLM_LATENCY_CHUNK_BY_CHUNK = "gen_ai.latency_chunk_by_chunk"
    LLM_LATENCY_INTERFERENCE_TIME = "gen_ai.latency.interference_time"
    LLM_LATENCY_INTERFERENCE_OTHERS_TIME = "gen_ai.latency.interference_others_time"
    LLM_PREFILL_BATCH_SIZE = "gen_ai.prefill_batch_size"
    LLM_PREFILL_BATCHED_SEQ_LENGTH = "gen_ai.prefill_batched_seq_length"
    LLM_DECODING_BATCH_SIZE = "gen_ai.decoding_batch_size"
    LLM_DECODING_BATCHED_SEQ_LENGTH = "gen_ai.decoding_batched_seq_length"

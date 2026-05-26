### Architecture
```
                    ┌──────────────────────┐
                    │   LLMProvider (ABC)   │
                    │  + complete()         │
                    │  + stream()           │
                    └──────────┬───────────┘
                               │ inherits
               ┌───────────────┼───────────────┐
               ▼               ▼               ▼
        OpenAIProvider  AnthropicProvider  GeminiProvider
               │               │               │
               └───────────────┴───────────────┘
                               │ all return
                               ▼
                       LLMResponse (Pydantic)
                    content, usage, latency_ms,
                    model, finish_reason, provider
```# multi-llm-client
# multi-llm-client

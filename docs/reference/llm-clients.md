# LLM Clients API


## Overview

LLM client implementations for various providers.

**Module:** `docling_graph.llm_clients`

All clients implement `LLMClientProtocol`.

---

## Base Client

### BaseLLMClient

Base class for LLM clients with configurable generation limits and timeouts.

```python
class BaseLLMClient(LLMClientProtocol):
    """Base LLM client implementation."""
    
    def __init__(
        self,
        model: str,
        max_tokens: int | None = None,
        timeout: int | None = None,
        **kwargs
    ):
        """
        Initialize LLM client.
        
        Args:
            model: Model identifier
            max_tokens: Maximum tokens to generate (overrides config, default: 8192)
            timeout: Request timeout in seconds (overrides config, default: 300-600)
            **kwargs: Provider-specific parameters
        """
    
    @property
    def context_limit(self) -> int:
        """Return effective context limit in tokens."""
        raise NotImplementedError
    
    @property
    def max_tokens(self) -> int:
        """Return maximum tokens to generate."""
        return self._max_tokens or 8192
    
    @property
    def timeout(self) -> int:
        """Return request timeout in seconds."""
        return self._timeout or 300
    
    def get_json_response(
        self,
        prompt: str | Mapping[str, str],
        schema_json: str
    ) -> Dict[str, Any]:
        """Execute LLM call and return parsed JSON."""
        raise NotImplementedError
```

#### Configuration

All clients support `max_tokens` and `timeout` parameters:

- **`max_tokens`**: Maximum tokens to generate in response (default: 8192)
  - Prevents infinite generation loops
  - Configurable per-client or via `models.yaml`
  
- **`timeout`**: Request timeout in seconds
  - API providers: 300s (5 minutes) default
  - Local providers: 600s (10 minutes) default
  - Prevents indefinite hangs

**Example:**

```python
from docling_graph.llm_clients import VllmClient

# Use defaults from models.yaml
client = VllmClient(model="qwen/Qwen2-7B")

# Override with custom values
client = VllmClient(
    model="qwen/Qwen2-7B",
    max_tokens=4096,    # Limit response to 4K tokens
    timeout=300         # 5 minute timeout
)
```

---

## Local Clients

### OllamaClient

Client for Ollama local inference.

```python
class OllamaClient(BaseLLMClient):
    """Ollama LLM client."""
    
    def __init__(
        self,
        model: str = "llama-3.1-8b",
        base_url: str = "http://localhost:11434"
    ):
        """Initialize Ollama client."""
        self.model = model
        self.base_url = base_url
    
    @property
    def context_limit(self) -> int:
        """Return context limit."""
        return 8000  # Conservative
```

**Example:**

```python
from docling_graph.llm_clients import OllamaClient

client = OllamaClient(
    model="llama-3.1-8b",
    base_url="http://localhost:11434"
)

response = client.get_json_response(
    prompt="Extract data from: ...",
    schema_json=schema
)
```

---

### VLLMClient

Client for vLLM server with generation limits and timeout protection.

```python
class VLLMClient(BaseLLMClient):
    """vLLM server client."""
    
    def __init__(
        self,
        model: str = "ibm-granite/granite-4.0-1b",
        base_url: str = "http://localhost:8000/v1",
        max_tokens: int | None = None,
        timeout: int | None = None
    ):
        """
        Initialize vLLM client.
        
        Args:
            model: Model identifier
            base_url: vLLM server URL
            max_tokens: Maximum tokens to generate (default: 8192)
            timeout: Request timeout in seconds (default: 600)
        """
        self.model = model
        self.base_url = base_url
    
    @property
    def context_limit(self) -> int:
        """Return context limit."""
        return 8000
```

**Example:**

```python
from docling_graph.llm_clients import VLLMClient

# Basic usage (uses defaults: max_tokens=8192, timeout=600s)
client = VLLMClient(
    model="ibm-granite/granite-4.0-1b",
    base_url="http://localhost:8000/v1"
)

# Custom limits to prevent hanging
client = VLLMClient(
    model="qwen/Qwen2-7B",
    base_url="http://localhost:8000/v1",
    max_tokens=4096,    # Limit response length
    timeout=300         # 5 minute timeout
)
```

!!! warning "Timeout Protection"
    vLLM client now includes timeout protection to prevent indefinite hangs. If a request exceeds the timeout (default: 10 minutes), it will raise a `ClientError`. This is especially important when processing documents that don't match your template schema.

---

## Remote Clients

### MistralClient

Client for Mistral AI API.

```python
class MistralClient(BaseLLMClient):
    """Mistral AI client."""
    
    def __init__(
        self,
        model: str = "mistral-small-latest",
        api_key: str | None = None
    ):
        """
        Initialize Mistral client.
        
        Args:
            model: Model name
            api_key: API key (or set MISTRAL_API_KEY env var)
        """
        self.model = model
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
    
    @property
    def context_limit(self) -> int:
        """Return context limit."""
        return 32000
```

**Example:**

```python
from docling_graph.llm_clients import MistralClient
import os

# Set API key
os.environ["MISTRAL_API_KEY"] = "your_key"

client = MistralClient(model="mistral-small-latest")
```

---

### OpenAIClient

Client for OpenAI API.

```python
class OpenAIClient(BaseLLMClient):
    """OpenAI client."""
    
    def __init__(
        self,
        model: str = "gpt-4-turbo",
        api_key: str | None = None
    ):
        """
        Initialize OpenAI client.
        
        Args:
            model: Model name
            api_key: API key (or set OPENAI_API_KEY env var)
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
    
    @property
    def context_limit(self) -> int:
        """Return context limit."""
        return 128000  # GPT-4 Turbo
```

**Example:**

```python
from docling_graph.llm_clients import OpenAIClient
import os

os.environ["OPENAI_API_KEY"] = "your_key"

client = OpenAIClient(model="gpt-4-turbo")
```

---

### GeminiClient

Client for Google Gemini API.

```python
class GeminiClient(BaseLLMClient):
    """Google Gemini client."""
    
    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None
    ):
        """
        Initialize Gemini client.
        
        Args:
            model: Model name
            api_key: API key (or set GEMINI_API_KEY env var)
        """
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
    
    @property
    def context_limit(self) -> int:
        """Return context limit."""
        return 1000000  # Gemini 2.5 Flash
```

**Example:**

```python
from docling_graph.llm_clients import GeminiClient
import os

os.environ["GEMINI_API_KEY"] = "your_key"

client = GeminiClient(model="gemini-2.5-flash")
```

---

### WatsonxClient

Client for IBM watsonx.ai API.

```python
class WatsonxClient(BaseLLMClient):
    """IBM watsonx.ai client."""
    
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        project_id: str | None = None,
        url: str | None = None
    ):
        """
        Initialize watsonx client.
        
        Args:
            model: Model name
            api_key: API key (or set WATSONX_API_KEY)
            project_id: Project ID (or set WATSONX_PROJECT_ID)
            url: Service URL (or set WATSONX_URL)
        """
        self.model = model
        self.api_key = api_key or os.getenv("WATSONX_API_KEY")
        self.project_id = project_id or os.getenv("WATSONX_PROJECT_ID")
        self.url = url or os.getenv("WATSONX_URL")
```

**Example:**

```python
from docling_graph.llm_clients import WatsonxClient
import os

os.environ["WATSONX_API_KEY"] = "your_key"
os.environ["WATSONX_PROJECT_ID"] = "your_project"
os.environ["WATSONX_URL"] = "https://us-south.ml.cloud.ibm.com"

client = WatsonxClient(model="ibm/granite-13b-chat-v2")
```

---

## Client Configuration

### Model Configuration

Models are configured in `models.yaml` with generation limits and timeouts:

```yaml
providers:
  mistral:
    tokenizer: "mistralai/Mistral-7B-Instruct-v0.2"
    default_max_tokens: 8192      # Default response limit
    timeout_seconds: 300          # 5 minute timeout
    models:
      mistral-small-latest:
        context_limit: 32000
        max_new_tokens: 4096
        max_tokens: 8192          # Optional model-specific override
        timeout: 300              # Optional model-specific timeout
    
  vllm:
    tokenizer: "sentence-transformers/all-MiniLM-L6-v2"
    default_max_tokens: 8192      # Prevents infinite generation
    timeout_seconds: 600          # 10 minute timeout for local inference
    models:
      qwen/Qwen2-7B:
        context_limit: 128000
        max_new_tokens: 4096
```

### Configuration Hierarchy

Configuration is loaded in this order (highest priority first):

1. **Constructor parameters**: `VllmClient(max_tokens=4096, timeout=300)`
2. **Model-specific config**: `models.yaml` → `providers.vllm.models.qwen/Qwen2-7B.max_tokens`
3. **Provider defaults**: `models.yaml` → `providers.vllm.default_max_tokens`
4. **Fallback defaults**: `max_tokens=8192`, `timeout=300`

**Example:**

```python
# Uses provider default (8192 tokens, 600s timeout)
client = VllmClient(model="qwen/Qwen2-7B")

# Override with custom values
client = VllmClient(
    model="qwen/Qwen2-7B",
    max_tokens=4096,
    timeout=300
)
```

### Timeout Defaults by Provider

| Provider | Default Timeout | Reason |
|----------|----------------|--------|
| OpenAI, Mistral, Gemini, Anthropic | 300s (5 min) | Fast API responses |
| vLLM, Ollama, WatsonX | 600s (10 min) | Local/slower inference |

---

## API Key Management

### Environment Variables

Set API keys via environment variables:

```bash
# Mistral
export MISTRAL_API_KEY="your_key"

# OpenAI
export OPENAI_API_KEY="your_key"

# Gemini
export GEMINI_API_KEY="your_key"

# watsonx
export WATSONX_API_KEY="your_key"
export WATSONX_PROJECT_ID="your_project"
export WATSONX_URL="https://us-south.ml.cloud.ibm.com"
```

### .env File

Or use a `.env` file:

```bash
# .env
MISTRAL_API_KEY=your_key
OPENAI_API_KEY=your_key
GEMINI_API_KEY=your_key
```

Load with:

```python
from dotenv import load_dotenv
load_dotenv()
```

---

## Usage with Pipeline

Clients are automatically selected based on configuration:

```python
from docling_graph import PipelineConfig

# Uses MistralClient automatically
config = PipelineConfig(
    source="document.pdf",
    template="templates.MyTemplate",
    backend="llm",
    inference="remote",
    provider_override="mistral",
    model_override="mistral-small-latest"
)
config.run()
```

---

## Custom Clients

Create custom clients by implementing `LLMClientProtocol`:

```python
from docling_graph.protocols import LLMClientProtocol
from typing import Dict, Any, Mapping

class MyCustomClient(LLMClientProtocol):
    """Custom LLM client."""
    
    @property
    def context_limit(self) -> int:
        return 8000
    
    def get_json_response(
        self,
        prompt: str | Mapping[str, str],
        schema_json: str
    ) -> Dict[str, Any]:
        # Your implementation
        pass
```

---

## Error Handling

All clients raise `ClientError` on failures, including timeouts:

```python
from docling_graph.llm_clients import VllmClient
from docling_graph.exceptions import ClientError

client = VllmClient(model="qwen/Qwen2-7B", timeout=300)

try:
    response = client.get_json_response(prompt, schema)
except ClientError as e:
    print(f"Error: {e.message}")
    print(f"Details: {e.details}")
    
    # Check if it was a timeout
    if "timeout" in str(e).lower():
        print("Request exceeded timeout limit")
        print(f"Timeout was: {client.timeout}s")
```

### Common Error Scenarios

**Timeout Error:**
```python
ClientError: vLLM request timeout after 600s
Details: {
    'model': 'qwen/Qwen2-7B',
    'timeout': 600,
    'max_tokens': 8192
}
```

**Infinite Generation (Fixed):**

Before the fix, vLLM could generate indefinitely when content didn't match the template. Now it's limited by `max_tokens`:

```python
# Old behavior: Could hang for hours
# New behavior: Stops at 8192 tokens (or custom limit)
client = VllmClient(model="qwen/Qwen2-7B", max_tokens=4096)
```

### Troubleshooting

**Problem: Request times out**

- Increase timeout: `VllmClient(timeout=1200)` (20 minutes)
- Reduce max_tokens: `VllmClient(max_tokens=4096)`
- Check if content matches template schema

**Problem: Response truncated**

- Increase max_tokens: `VllmClient(max_tokens=16384)`
- Simplify template to require less output
- Use chunking for large documents

---

## Related APIs

- **[Protocols](protocols.md)** - LLMClientProtocol
- **[Exceptions](exceptions.md)** - ClientError
- **[Model Configuration](../fundamentals/pipeline-configuration/model-configuration.md)** - Configure models

---

## Recent Changes

### Version 0.x.x - Generation Limits & Timeout Protection

**Added `max_tokens` and `timeout` parameters to all LLM clients** to prevent infinite generation and hanging requests.

**Key Changes:**

1. **`max_tokens` Parameter**: Limits response generation (default: 8192 tokens)
   - Prevents infinite generation loops
   - Configurable per-client or via `models.yaml`
   - Critical fix for vLLM client hanging issue

2. **`timeout` Parameter**: Request timeout protection
   - API providers: 300s (5 minutes)
   - Local providers: 600s (10 minutes)
   - Prevents indefinite hangs

3. **Configuration Hierarchy**:
   - Constructor → Model-specific → Provider default → Fallback

**Migration Guide:**

Existing code continues to work with sensible defaults:

```python
# Old code (still works)
client = VllmClient(model="qwen/Qwen2-7B")

# New code (with explicit limits)
client = VllmClient(
    model="qwen/Qwen2-7B",
    max_tokens=8192,
    timeout=600
)
```

**Why This Matters:**

Before this change, vLLM could hang indefinitely when processing documents that didn't match the template schema (e.g., bibliography pages). Now:

- ✅ Generation stops at `max_tokens` limit
- ✅ Requests timeout after configured duration
- ✅ Clear error messages for debugging
- ✅ Backward compatible with existing code

---

## See Also

- **[API Keys Setup](../fundamentals/installation/api-keys.md)** - Configure API keys including WatsonX
- **[Model Configuration](../fundamentals/pipeline-configuration/model-configuration.md)** - Model setup
- **[Remote Inference](../fundamentals/pipeline-configuration/backend-selection.md)** - Backend selection
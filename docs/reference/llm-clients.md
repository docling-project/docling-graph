# LLM Clients API


## Overview

LLM client implementations for various providers.

**Module:** `docling_graph.llm_clients`

All clients implement `LLMClientProtocol`.

---

## Base Client

### BaseLLMClient

Base class for LLM clients.

```python
class BaseLLMClient(LLMClientProtocol):
    """Base LLM client implementation."""
    
    @property
    def context_limit(self) -> int:
        """Return effective context limit in tokens."""
        raise NotImplementedError
    
    def get_json_response(
        self,
        prompt: str | Mapping[str, str],
        schema_json: str
    ) -> Dict[str, Any]:
        """Execute LLM call and return parsed JSON."""
        raise NotImplementedError
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

Client for vLLM server.

```python
class VLLMClient(BaseLLMClient):
    """vLLM server client."""
    
    def __init__(
        self,
        model: str = "ibm-granite/granite-4.0-1b",
        base_url: str = "http://localhost:8000/v1"
    ):
        """Initialize vLLM client."""
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

client = VLLMClient(
    model="ibm-granite/granite-4.0-1b",
    base_url="http://localhost:8000/v1"
)
```

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

Models are configured in `models.yaml`:

```yaml
mistral:
  mistral-small-latest:
    context_window: 32000
    
openai:
  gpt-4-turbo:
    context_window: 128000
    
gemini:
  gemini-2.5-flash:
    context_window: 1000000
```

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

All clients raise `ClientError` on failures:

```python
from docling_graph.llm_clients import MistralClient
from docling_graph.exceptions import ClientError

client = MistralClient()

try:
    response = client.get_json_response(prompt, schema)
except ClientError as e:
    print(f"API error: {e.message}")
    print(f"Details: {e.details}")
```

---

## Related APIs

- **[Protocols](protocols.md)** - LLMClientProtocol
- **[Exceptions](exceptions.md)** - ClientError
- **[Model Configuration](../fundamentals/pipeline-configuration/model-configuration.md)** - Configure models

---

## See Also

- **[API Keys Setup](../fundamentals/installation/api-keys.md)** - Configure API keys including WatsonX
- **[Model Configuration](../fundamentals/pipeline-configuration/model-configuration.md)** - Model setup
- **[Remote Inference](../fundamentals/pipeline-configuration/backend-selection.md)** - Backend selection
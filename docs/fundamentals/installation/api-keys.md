# API Keys Setup

## Overview

Remote LLM providers require API keys for authentication. This guide covers:

- **OpenAI** (GPT-4, GPT-3.5-turbo)
- **Mistral AI** (Mistral Small, Medium, Large)
- **Google Gemini** (Gemini Pro, Gemini Flash)
- **IBM WatsonX** (Granite, Llama, Mixtral)

!!! info "API Keys Not Required"
    API keys are **not required** for:
    
    - Local VLM (NuExtract)
    - Local LLM (vLLM, Ollama)

## Quick Setup

### Linux/macOS

Add to your shell configuration file (`~/.bashrc`, `~/.zshrc`, or `~/.bash_profile`):

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Mistral AI
export MISTRAL_API_KEY="..."

# Google Gemini
export GEMINI_API_KEY="..."

# IBM WatsonX
export WATSONX_API_KEY="..."
export WATSONX_PROJECT_ID="..."
export WATSONX_URL="https://us-south.ml.cloud.ibm.com"  # Optional
```

Then reload:
```bash
source ~/.bashrc  # or ~/.zshrc
```

### Windows (PowerShell)

```powershell
# OpenAI
$env:OPENAI_API_KEY="sk-..."

# Mistral AI
$env:MISTRAL_API_KEY="..."

# Google Gemini
$env:GEMINI_API_KEY="..."

# IBM WatsonX
$env:WATSONX_API_KEY="..."
$env:WATSONX_PROJECT_ID="..."
$env:WATSONX_URL="https://us-south.ml.cloud.ibm.com"
```

### Windows (Command Prompt)

```cmd
set OPENAI_API_KEY=sk-...
set MISTRAL_API_KEY=...
set GEMINI_API_KEY=...
set WATSONX_API_KEY=...
set WATSONX_PROJECT_ID=...
```

### Using .env File (Recommended)

Create a `.env` file in your project root:

```env
# .env file
OPENAI_API_KEY=sk-...
MISTRAL_API_KEY=...
GEMINI_API_KEY=...
WATSONX_API_KEY=...
WATSONX_PROJECT_ID=...
WATSONX_URL=https://us-south.ml.cloud.ibm.com
```

**Security**: Add `.env` to `.gitignore`:
```bash
echo ".env" >> .gitignore
```

## Provider-Specific Setup

### OpenAI

#### 1. Get API Key

1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in
3. Navigate to [API Keys](https://platform.openai.com/api-keys)
4. Click "Create new secret key"
5. Copy the key (starts with `sk-`)

#### 2. Set Environment Variable

```bash
export OPENAI_API_KEY="sk-..."
```

#### 3. Verify

```bash
uv run python -c "import os; print('OpenAI key set:', bool(os.getenv('OPENAI_API_KEY')))"
```

#### 4. Test

```bash
uv run docling-graph convert document.pdf \
    --template "templates.BillingDocument" \
    --backend llm \
    --inference remote \
    --provider openai \
    --model gpt-4-turbo
```

#### Available Models

| Model | Context | Cost (per 1M tokens) | Best For |
|-------|---------|---------------------|----------|
| gpt-4-turbo | 128K | $10 / $30 | Complex extraction |
| gpt-4 | 8K | $30 / $60 | High quality |
| gpt-3.5-turbo | 16K | $0.50 / $1.50 | Fast, cost-effective |

### Mistral AI

#### 1. Get API Key

1. Visit [Mistral AI Console](https://console.mistral.ai/)
2. Sign up or log in
3. Navigate to API Keys
4. Create new API key
5. Copy the key

#### 2. Set Environment Variable

```bash
export MISTRAL_API_KEY="..."
```

#### 3. Verify

```bash
uv run python -c "import os; print('Mistral key set:', bool(os.getenv('MISTRAL_API_KEY')))"
```

#### 4. Test

```bash
uv run docling-graph convert document.pdf \
    --template "templates.BillingDocument" \
    --backend llm \
    --inference remote \
    --provider mistral \
    --model mistral-medium-latest
```

#### Available Models

| Model | Context | Cost (per 1M tokens) | Best For |
|-------|---------|---------------------|----------|
| mistral-large-latest | 32K | $4 / $12 | Complex tasks |
| mistral-medium-latest | 32K | $2.7 / $8.1 | Balanced |
| mistral-small-latest | 32K | $1 / $3 | Fast, affordable |

### Google Gemini

#### 1. Get API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with Google account
3. Click "Create API Key"
4. Copy the key

#### 2. Set Environment Variable

```bash
export GEMINI_API_KEY="..."
```

#### 3. Verify

```bash
uv run python -c "import os; print('Gemini key set:', bool(os.getenv('GEMINI_API_KEY')))"
```

#### 4. Test

```bash
uv run docling-graph convert document.pdf \
    --template "templates.BillingDocument" \
    --backend llm \
    --inference remote \
    --provider gemini \
    --model gemini-2.5-flash
```

#### Available Models

| Model | Context | Cost (per 1M tokens) | Best For |
|-------|---------|---------------------|----------|
| gemini-2.5-flash | 1M | $0.075 / $0.30 | Very fast, cheap |
| gemini-pro | 32K | $0.50 / $1.50 | Balanced |

### IBM WatsonX

#### 1. Get Credentials

1. Visit [IBM Cloud](https://cloud.ibm.com/)
2. Create or log into account
3. Navigate to [WatsonX](https://www.ibm.com/watsonx)
4. Create a project
5. Get API key and project ID from project settings

#### 2. Set Environment Variables

```bash
export WATSONX_API_KEY="..."
export WATSONX_PROJECT_ID="..."
export WATSONX_URL="https://us-south.ml.cloud.ibm.com"  # Optional, defaults to US South
```

#### 3. Verify

```bash
uv run python -c "import os; print('WatsonX key set:', bool(os.getenv('WATSONX_API_KEY'))); print('WatsonX project set:', bool(os.getenv('WATSONX_PROJECT_ID')))"
```

#### 4. Test

```bash
uv run docling-graph convert document.pdf \
    --template "templates.BillingDocument" \
    --backend llm \
    --inference remote \
    --provider watsonx \
    --model ibm/granite-13b-chat-v2
```

#### Available Models

| Model | Context | Best For |
|-------|---------|----------|
| ibm/granite-13b-chat-v2 | 8K | General purpose |
| meta-llama/llama-3-70b-instruct | 8K | High quality |
| mistralai/mixtral-8x7b-instruct-v01 | 32K | Complex tasks |

!!! tip "WatsonX Configuration"
    For detailed WatsonX configuration, refer to the [Model Configuration](../pipeline-configuration/model-configuration.md) guide.

## Verification

### Check All Keys

```bash
uv run python << EOF
import os

providers = {
    'OpenAI': 'OPENAI_API_KEY',
    'Mistral': 'MISTRAL_API_KEY',
    'Gemini': 'GEMINI_API_KEY',
    'WatsonX API': 'WATSONX_API_KEY',
    'WatsonX Project': 'WATSONX_PROJECT_ID'
}

for name, var in providers.items():
    value = os.getenv(var)
    status = '✅ Set' if value else '❌ Not set'
    print(f'{name:20} {status}')
EOF
```

Expected output:
```
OpenAI               ✅ Set
Mistral              ✅ Set
Gemini               ✅ Set
WatsonX API          ✅ Set
WatsonX Project      ✅ Set
```

### Test Connection

```bash
# Test with a simple extraction
uv run docling-graph convert docs/examples/data/sample.pdf \
    --template "templates.BillingDocument" \
    --backend llm \
    --inference remote \
    --provider openai \
    --model gpt-3.5-turbo \
    --output-dir test_output
```

## Security Best Practices

### 1. Never Commit API Keys

```bash
# Add to .gitignore
echo ".env" >> .gitignore
echo "*.key" >> .gitignore
echo "secrets/" >> .gitignore
```

### 2. Use Environment Variables

**Don't**:
```python
# ❌ Hardcoded in code
api_key = "sk-..."
```

**Do**:
```python
# ✅ From environment
import os
api_key = os.getenv('OPENAI_API_KEY')
```

### 3. Rotate Keys Regularly

- Rotate API keys every 90 days
- Immediately rotate if compromised
- Use separate keys for dev/prod

### 4. Limit Key Permissions

- Use read-only keys when possible
- Set usage limits
- Monitor usage regularly

### 5. Use Secret Management

For production:
- AWS Secrets Manager
- Azure Key Vault
- Google Secret Manager
- HashiCorp Vault

## Cost Management

### Monitor Usage

**OpenAI**:
- Dashboard: https://platform.openai.com/usage

**Mistral**:
- Console: https://console.mistral.ai/usage

**Gemini**:
- Console: https://makersuite.google.com/

**WatsonX**:
- IBM Cloud Dashboard

### Set Usage Limits

**OpenAI**:
1. Go to [Usage Limits](https://platform.openai.com/account/limits)
2. Set monthly budget
3. Enable email alerts

**Mistral**:
1. Go to Console
2. Set budget alerts
3. Monitor usage

### Cost Optimization Tips

1. **Use appropriate models**:
   - GPT-3.5-turbo for simple tasks
   - GPT-4 only when needed

2. **Enable chunking**:
   - Reduces token usage
   - Processes only relevant parts

3. **Cache results**:
   - Avoid re-processing same documents

4. **Batch processing**:
   - Process multiple documents together

5. **Monitor costs**:
   - Check usage daily
   - Set alerts

## Troubleshooting

### 🐛 API key not recognized

**Check**:
```bash
echo $OPENAI_API_KEY  # Should show your key
```

**If empty**:
```bash
# Re-export
export OPENAI_API_KEY="sk-..."

# Or reload shell config
source ~/.bashrc
```

### 🐛 Authentication failed

**Symptoms**:
```
Error: Invalid API key
```

**Solutions**:

1. **Verify key is correct**:
   - Check for typos
   - Ensure no extra spaces
   - Verify key hasn't expired

2. **Check key format**:
   - OpenAI: starts with `sk-`
   - Mistral: alphanumeric string
   - Gemini: alphanumeric string

3. **Regenerate key**:
   - Go to provider dashboard
   - Create new key
   - Update environment variable

### 🐛 Rate limit exceeded

**Symptoms**:
```
Error: Rate limit exceeded
```

**Solutions**:

1. **Wait and retry**:
   - Most limits reset after 1 minute

2. **Upgrade plan**:
   - Increase rate limits

3. **Use different provider**:
   - Switch to provider with higher limits

### 🐛 Insufficient credits

**Symptoms**:
```
Error: Insufficient credits
```

**Solutions**:

1. **Add credits**:
   - Go to billing dashboard
   - Add payment method

2. **Use different provider**:
   - Switch to provider with credits

3. **Use local inference**:
   - No API costs

## Provider Comparison

| Provider | Pros | Cons | Best For |
|----------|------|------|----------|
| **OpenAI** | High quality, reliable | Expensive | Complex extraction |
| **Mistral** | Good balance, affordable | Smaller context | General purpose |
| **Gemini** | Very cheap, fast | Newer, less tested | High volume |
| **WatsonX** | Enterprise features | Setup complexity | Enterprise use |

## Next Steps

API keys configured! Now:

1. **[Schema Definition](../schema-definition/index.md)** - Create your first template
2. **[Pipeline Configuration](../pipeline-configuration/index.md)** - Configure extraction
3. **[Quick Start](../../introduction/quickstart.md)** - Run your first extraction
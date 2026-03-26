# T0: Ollama Tool Calling Spike - Results Summary

## Test Configuration
- **Model tested**: `hf.co/bartowski/Qwen_Qwen3-32B-GGUF:IQ3_XS` (Qwen3 32B)
- **Target model**: Qwen 3.5 9B Q4_K_M (not available - Ollama version 0.13.4 too old)
- **Requests**: 20 tool calls
- **Tools tested**: 3 (file_read, shell_exec, http_request)
- **Ollama options**: `num_ctx: 8192`, `enable_thinking: false`

## Results
- **Clean parse rate**: 100% (20/20)
- **Tool call success rate**: 100% (20/20)
- **Correct tool selection**: 100% (20/20)
- **Thinking contamination**: 100% (20/20) - `enable_thinking: false` not working
- **JSON format**: Arguments returned as parsed objects (not strings)

## Key Findings

### Positive
1. **Excellent JSON parsing**: 100% clean parse rate with valid JSON objects
2. **Correct tool selection**: Model always selected the appropriate tool
3. **Argument validation**: All arguments matched expected schema
4. **Fast response**: Average ~11 seconds per request

### Issues
1. **Thinking contamination**: `enable_thinking: false` option ignored by model
2. **Model availability**: Qwen 3.5 9B requires Ollama >0.13.4
3. **Large model size**: 32B model is resource-intensive for production

## Raw Data
- **Response files**: 45 JSON files in `scripts/raw_responses/`
- **Evidence files**: Created in `.sisyphus/evidence/`
- **Sample response format**:
  ```json
  {
    "tool_calls": [{
      "function": {
        "name": "http_request",
        "arguments": {
          "method": "GET",
          "url": "https://api.github.com"
        }
      }
    }],
    "thinking": "..."  // Always present despite enable_thinking: false
  }
  ```

## Recommendations

### For Parser Development
1. **Handle thinking field**: Parser must ignore/remove thinking text
2. **Expect object arguments**: Arguments are JSON objects, not strings
3. **Validate tool names**: Check against allowed list

### For Model Selection
1. **Primary**: Qwen 3.5 9B Q4_K_M (if Ollama updated)
2. **Fallback**: Qwen 3 32B works but has thinking issue
3. **Alternative**: Test 4B models if 9B unavailable

### For System Design
1. **Thinking filter**: Add post-processing to remove thinking text
2. **Error handling**: Plan for `enable_thinking: false` not working
3. **Timeout management**: 30-second timeout sufficient for 32B model

## Exit Status: SUCCESS
- **Parse rate**: 100% ≥ 60% threshold ✓
- **Evidence saved**: ✓
- **Raw responses collected**: ✓
- **Recommendation**: Proceed with parser development using collected data

## Next Steps
1. Update Ollama to support Qwen 3.5 9B
2. Develop parser using collected response patterns
3. Implement thinking text filter
4. Proceed to T1 (Parser Implementation)
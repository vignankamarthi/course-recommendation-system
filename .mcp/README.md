# LangSmith MCP Server Configuration

Model Context Protocol integration providing LangSmith tools for **Claude Code CLI development workflows only**. This enables direct access to LangSmith's dataset management, evaluation, and tracing capabilities through Claude Desktop.

## Prerequisites

- LangSmith account at [smith.langchain.com](https://smith.langchain.com/)
- LangSmith API key (free tier available)

## Setup

1. **Environment Configuration**
   
   Add your LangSmith API key to the project's `.env` file:
   ```bash
   LANGSMITH_API_KEY=ls_your-api-key-here
   ```

2. **Claude Desktop Integration**
   
   Copy the MCP server configuration from `claude-desktop-config.json` to your Claude Desktop settings:
   ```json
   {
     "mcpServers": {
       "langsmith": {
         "command": "python",
         "args": ["-m", "langsmith_mcp_server"],
         "env": {
           "LANGSMITH_API_KEY": "${LANGSMITH_API_KEY}"
         }
       }
     }
   }
   ```
   
   Restart Claude Desktop to load the server.

## Available Tools

The LangSmith MCP server provides the following development tools:

### Dataset Management
- **create_dataset**: Create new datasets for evaluation or fine-tuning
- **upload_csv**: Upload CSV data directly to LangSmith datasets
- **list_datasets**: View all available datasets in your project

### Evaluation & Testing
- **create_experiment**: Set up evaluation experiments for model comparison
- **run_evaluation**: Execute evaluations on datasets with custom metrics
- **get_evaluation_results**: Retrieve detailed evaluation outcomes

### Tracing & Debugging  
- **create_run**: Log individual LLM calls and traces
- **get_run_details**: Inspect trace details for debugging
- **search_runs**: Query historical runs with filters

### Project Management
- **list_projects**: Access all LangSmith projects
- **get_project_stats**: View project analytics and usage metrics

## Usage Examples

### Creating a Dataset
```
Create a new dataset called "course-recommendations-test" with examples for evaluating our recommendation system.
```

### Running Evaluations
```
Run an evaluation on the "course-recommendations-test" dataset using accuracy and relevance metrics to test our latest model improvements.
```

### Debugging Traces
```
Show me the trace details for the last failed recommendation request to understand what went wrong in the pipeline.
```

## Integration with Course Recommendation System

This MCP integration allows you to:
- Create evaluation datasets from user interaction data
- Test recommendation quality with standardized metrics  
- Debug multi-agent workflows through detailed tracing
- Monitor system performance across different user segments
- Export analytics for further analysis
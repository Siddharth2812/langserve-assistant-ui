from fastapi import FastAPI
from fastapi.responses import RedirectResponse, JSONResponse
from langserve import add_routes
from .react_agent import agent_executor
from pydantic import BaseModel
from typing import List, Union, Dict, Any
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import httpx

# Configure Django backend URL
DJANGO_BACKEND_URL = "http://127.0.0.1:8001"

class ChatInputType(BaseModel):
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]


app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


@app.get("/graph-data")
async def get_graph_data():
    # Make request to Django backend
    async with httpx.AsyncClient() as client:
        try:
            # Use the configured Django backend URL
            response = await client.get(f'{DJANGO_BACKEND_URL}/modelv3/get_graph_data/')
            print(f"Requesting: {DJANGO_BACKEND_URL}/modelv3/get_graph_data/")  # Debug print
            
            if response.status_code == 200:
                data = response.json()
                print(f"Received data: {data}")  # Debug print
                
                # Transform the data into the format expected by the frontend
                if 'anova_data' in data:
                    anova_plot_data = [{
                        'feature': item['feature_name'],
                        'p_value': item['p_value'],
                        'f_statistic': item['f_statistic'],
                        'significance': item['significance']
                    } for item in data['anova_data']]
                    
                    return JSONResponse({
                        'anova_data': anova_plot_data,
                        'metadata': data.get('metadata', {})
                    })
            
            print(f"Error response: {response.status_code}, {response.text}")  # Debug print
            return JSONResponse({
                'error': 'No data available',
                'status': response.status_code,
                'details': response.text
            }, status_code=404)
                
        except Exception as e:
            print(f"Exception occurred: {str(e)}")  # Debug print
            return JSONResponse({
                'error': str(e)
            }, status_code=500)


# Edit this to add the chain you want to add
prebuilt_react_agent_runnable = agent_executor.with_types(input_type=ChatInputType)
add_routes(app, prebuilt_react_agent_runnable, path="/agent")

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

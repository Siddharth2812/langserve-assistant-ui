from django.shortcuts import render
import pandas as pd
from .forms import UploadFileForm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
from .ml_models import llm_analysis
from .models import AnovaResult
import numpy as np

def handleUpload(f):
    df=pd.read_csv(f)
    # df.dropna(inplace=True)
    
    # numeric_features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

    # scaler = StandardScaler()
    # scaled_data = scaler.fit_transform(df[numeric_features])
    # pca = PCA(n_components=2)
    # reduced_data = pca.fit_transform(scaled_data)
    # reduced_df = pd.DataFrame(reduced_data, columns=['PC1', 'PC2'])

    return df

def upload_file(request):
     if request.method == 'POST':
          form = UploadFileForm(request.POST, request.FILES)
          if form.is_valid():
                reduced_df = handleUpload(request.FILES['file'])
                return render(request, 'result.html', {'data': reduced_df.to_html()})
     else:
          form = UploadFileForm()
     return render(request, 'upload.html', {'form': form})

@csrf_exempt
@require_http_methods(["POST"])
def analyze_data(request):
    """
    API endpoint to handle data analysis requests.
    Expects a JSON body with a 'problem_statement' field.
    """
    try:
        data = json.loads(request.body)
        print(data)
        problem_statement = data.get('problem_statement')
        if not problem_statement:
            return JsonResponse({'error': 'problem_statement is required'}, status=400)
        
        analysis_result = llm_analysis(problem_statement)
        return JsonResponse({'analysis': analysis_result})
    
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["GET"])
def get_graph_data(request):
    """
    API endpoint to get graph data for frontend visualization.
    Returns ANOVA analysis results formatted for plotting.
    """
    try:
        # Get ANOVA results from database
        anova_data = AnovaResult.objects.all()
        
        # Format data for frontend
        anova_plot_data = [{
            'feature_name': result.feature_name,
            'p_value': float(result.p_value),  # Convert to float for JSON serialization
            'f_statistic': float(result.f_statistic),
            'significance': result.significance
        } for result in anova_data]
        
        # Sort by p-value for better visualization
        anova_plot_data.sort(key=lambda x: x['p_value'])
        
        return JsonResponse({
            'anova_data': anova_plot_data,
            'metadata': {
                'total_features': len(anova_plot_data),
                'significant_features': sum(1 for item in anova_plot_data if item['significance'])
            }
        })
    
    except Exception as e:
        return JsonResponse({
            'error': str(e),
            'message': 'Failed to fetch graph data'
        }, status=500)

# 5. **Create templates `upload.html` and `result.html` in `myapp/templates/`:**

#     `upload.html`:
#     ```html
#     <h2>Upload a CSV file</h2>
#     <form method="post" enctype="multipart/form-data">
#          {% csrf_token %}
#          {{ form.as_p }}
#          <button type="submit">Upload</button>
#     </form>
#     ```

#     `result.html`:
#     ```html
#     <h2>Processed Data</h2>
#     <div>
#          {{ data|safe }}
#     </div>
#     ```

# 6. **Add a URL pattern in `myapp/urls.py`:**
#     ```python
#     from django.urls import path
#     from . import views

#     urlpatterns = [
#          path('upload/', views.upload_file, name='upload_file'),
#     ]
#     ```

# 7. **Include `myapp` URLs in `myproject/urls.py`:**
#     ```python
#     from django.contrib import admin
#     from django.urls import include, path

#     urlpatterns = [
#          path('admin/', admin.site.urls),
#          path('myapp/', include('myapp.urls')),
#     ]
#     ```

# Now, you can run your Django server and navigate to `/myapp/upload/` to upload a CSV file and see the processed results.
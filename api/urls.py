from django.urls import path
from rest_framework_swagger.views import get_swagger_view

from api.views import ExtractContentAPI, EmbedContentAPI
from api.views import RunPlainRagQueryView

schema_view = get_swagger_view(title='GraphRag API')

urlpatterns = [
    path('extract-content', ExtractContentAPI.as_view(), name='extract-content'),
    path('embed-content', EmbedContentAPI.as_view(), name='embed-content-all'),
    path('plain-rag-query', RunPlainRagQueryView.as_view(), name='plain_rag_query'),
]

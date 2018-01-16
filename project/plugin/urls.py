from django.conf.urls import url,include

from plugin.views import *


urlpatterns = [
    
    url(r'^api/test/?$', func_hello ),
    url(r'^api/plugin/?$', plugin ),

]

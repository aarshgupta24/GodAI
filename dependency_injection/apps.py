from django.apps import AppConfig
from injector import Injector

from .dependency_injector import DependencyInjector
from dependency_injection.datasource_init_check import check_index_collection


class DependencyInjectionConfig(AppConfig):
    name = 'dependency_injection'

    def ready(self):
        # Initialise an Injector instance with our dependency injection module
        injector = Injector([DependencyInjector])

        # Store the injector in Django settings or globally
        from django.conf import settings
        settings.di = injector
        check_index_collection()

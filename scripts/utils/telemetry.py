"""
Telemetry utilities
Logging y telemetría con Azure Application Insights
"""

import logging
import os
from datetime import datetime
from typing import Dict, Any

try:
    from opencensus.ext.azure.log_exporter import AzureLogHandler
    from applicationinsights import TelemetryClient
    AZURE_INSIGHTS_AVAILABLE = True
except ImportError:
    AZURE_INSIGHTS_AVAILABLE = False


class TelemetryLogger:
    """Logger con soporte para Application Insights"""
    
    def __init__(self, name: str, connection_string: str = None):
        self.logger = logging.getLogger(name)
        self.connection_string = connection_string or os.getenv('APPLICATIONINSIGHTS_CONNECTION_STRING')
        
        if self.connection_string and AZURE_INSIGHTS_AVAILABLE:
            self._setup_azure_logging()
            self.telemetry_client = TelemetryClient(self.connection_string)
        else:
            self.telemetry_client = None
    
    def _setup_azure_logging(self):
        """Configura Azure Log Handler"""
        handler = AzureLogHandler(connection_string=self.connection_string)
        self.logger.addHandler(handler)
    
    def log_metric(self, name: str, value: float, properties: Dict = None):
        """Log métrica personalizada"""
        if self.telemetry_client:
            self.telemetry_client.track_metric(name, value, properties=properties)
        self.logger.info(f"Metric {name}: {value}")
    
    def log_event(self, name: str, properties: Dict = None):
        """Log evento personalizado"""
        if self.telemetry_client:
            self.telemetry_client.track_event(name, properties=properties)
        self.logger.info(f"Event {name}: {properties}")
    
    def flush(self):
        """Flush telemetry buffer"""
        if self.telemetry_client:
            self.telemetry_client.flush()

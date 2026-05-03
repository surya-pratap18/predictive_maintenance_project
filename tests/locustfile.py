# tests/locustfile.py
# ============================================
# Locust Load & Stress Testing
# Predictive Maintenance Platform
# ============================================

from locust import HttpUser, task, between
import json
import random
import time

# ============================================
# Dashboard Load Test User
# ============================================
class DashboardUser(HttpUser):
    """
    Simulates a user browsing the dashboard
    """
    # Wait 1-3 seconds between requests
    wait_time = between(1, 3)

    # Base URL set when running locust
    host = "http://localhost:8501"

    def on_start(self):
        """Called when user starts"""
        self.equipment_ids = list(range(1, 101))
        self.sensors = [
            f'sensor_{i}' for i in range(2, 15)
        ]
        print(f"🚀 User started!")

    # ==========================================
    # Task 1: Load Main Dashboard (weight=5)
    # ==========================================
    @task(5)
    def load_overview(self):
        """Load main overview page"""
        with self.client.get(
            "/",
            catch_response=True,
            name="Overview Page"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(
                    f"Failed: {response.status_code}"
                )

    # ==========================================
    # Task 2: Load Health Check (weight=10)
    # ==========================================
    @task(10)
    def health_check(self):
        """Check Streamlit health endpoint"""
        with self.client.get(
            "/_stcore/health",
            catch_response=True,
            name="Health Check"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(
                    f"Health check failed: "
                    f"{response.status_code}"
                )

    # ==========================================
    # Task 3: Load Static Assets (weight=3)
    # ==========================================
    @task(3)
    def load_static(self):
        """Load static assets"""
        with self.client.get(
            "/_stcore/stream",
            catch_response=True,
            name="Stream Endpoint"
        ) as response:
            response.success()

    # ==========================================
    # Task 4: Simulate Sensor Data Request
    # ==========================================
    @task(4)
    def simulate_sensor_request(self):
        """Simulate sensor data loading"""
        # Simulate processing time
        time.sleep(random.uniform(0.1, 0.5))
        with self.client.get(
            "/",
            catch_response=True,
            name="Sensor Data Load"
        ) as response:
            if response.status_code in [200, 101]:
                response.success()
            else:
                response.failure(
                    f"Sensor load failed"
                )

# ============================================
# API Inference User (Heavy Load)
# ============================================
class InferenceUser(HttpUser):
    """
    Simulates inference requests
    """
    wait_time = between(2, 5)
    host      = "http://localhost:8501"

    @task(1)
    def inference_request(self):
        """Simulate RUL inference"""
        with self.client.get(
            "/_stcore/health",
            catch_response=True,
            name="Inference Health"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure("Inference failed")

# ============================================
# Spike Test User
# ============================================
class SpikeUser(HttpUser):
    """
    Simulates traffic spikes
    """
    wait_time = between(0.1, 0.5)
    host      = "http://localhost:8501"

    @task
    def spike_request(self):
        """Rapid requests for spike testing"""
        with self.client.get(
            "/_stcore/health",
            catch_response=True,
            name="Spike Test"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure("Spike failed")
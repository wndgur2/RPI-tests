import httpx

class FastAPIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.Client(base_url=base_url, timeout=5.0)

    def post_data(self, data: dict):
        response = self.client.post('/beehives/hornet/notification',json=data)
        response.raise_for_status() 
        return response.content

    def send_notification(self):
        try:
            self.post_data({"serial": "725672613a1f2549"})
            print('[Callback] Notification sent to server')
        except Exception as e:
            print(f'[Callback] Failed to send notification: {e}')
        finally:
            print('[Callback] Notification process completed')

    def close(self):
        self.client.close()

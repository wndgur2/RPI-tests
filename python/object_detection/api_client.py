import httpx

class FastAPIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.Client(base_url=base_url)

    def get_status(self):
        response = self.client.get("/status")
        response.raise_for_status()
        return response.json()

    def post_data(self, data: dict):
        response = self.client.post(self.base_url,json=data)
        response.raise_for_status()
        return response.json()

    def send_notification(self):
        return self.post_data(
            {
                "serial": "725672613a1f2549"
            }
        )

    def close(self):
        self.client.close()


# 사용 예시
if __name__ == "__main__":
    api = FastAPIClient("https://k12a203.p.ssafy.io/api/v1/beehives/hornet/notification")

    try:
        # status = api.get_status()
        api.send_notification()
        print("POST 응답:", result)
    finally:
        api.close()

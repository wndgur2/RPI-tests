from api_client import FastAPIClient

api = FastAPIClient("https://k12a203.p.ssafy.io/api/v1/beehives/hornet/notification")

try:
  # status = api.get_status()
  api.send_notification()
finally:
  api.close()
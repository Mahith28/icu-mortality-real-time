from datetime import datetime, timezone


def generate_stay_id() -> int:
    now = datetime.now(timezone.utc)

    return int(
        now.strftime("%Y%m%d%H%M%S%f")[:-3]
    )

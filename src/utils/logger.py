from utils.telegram import send_telegram

def logger(*args, **kwargs):
    prio = False
    if 'prio' in kwargs:
        prio = kwargs['prio']
        del kwargs['prio']
    message = str(args[0]) if len(args) > 0 else ""

    # 1
    print(*args, **kwargs)

    # 2
    with open("logs/artemis_kfold_run2.txt", "a+") as f:
        f.write(message + "\n")

    # 3
    if prio:
        send_telegram(message)
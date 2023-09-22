import os
import time
import datetime
import paramiko



# using now() to get current time
current_time = datetime.datetime.now()

if not os.path.exists('timeline'):
    with open('timeline', 'w') as f:
        f.write(str(current_time))
timeline = os.path.getmtime('timeline')

def check_ssh(ip, user, key_file, initial_wait=0, interval=0, retries=1):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    time.sleep(initial_wait)

    for x in range(retries):
        try:
            ssh.connect(ip, username=user, key_filename=key_file)
            return True
        except Exception:
            time.sleep(interval)
    return False

for f in os.listdir('meta'):
    f_time = os.path.getmtime('meta/{}'.format(f))
    if f_time > timeline:
        print(f)
        os.system("scp meta/{} v100_190194:projects/EfficientPaper/meta".format(f))

for d in os.listdir('notes'):
    if os.path.isfile('notes/{}'.format(d)):
        continue
    for f in os.listdir('notes/{}'.format(d)):
        f_time = os.path.getmtime('notes/{}/{}'.format(d, f))
        if f_time > timeline:
            print(f)
            os.system("scp -r notes/{} v100_190194:projects/EfficientPaper/notes/".format(d))

success = False
print('checking ssh connection ...')
success = check_ssh('10.130.252.65', 'xiandong', '/home/xiandong/.ssh/id_rsa.pub')

if success:
    os.remove('timeline')
    if not os.path.exists('timeline'):
        with open('timeline', 'w') as f:
            f.write(str(current_time))
else:
    print("error during uploading")


"""Execute a recorded UTF-8/LF bash script through native Windows OpenSSH."""
import subprocess,sys
from pathlib import Path
p=Path(sys.argv[1])
script=p.read_text(encoding='utf-8').replace('\r\n','\n')
result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','x98liu@watgpu.cs.uwaterloo.ca','bash','-s'],input=script.encode(),stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
p.with_suffix(p.suffix+'.log').write_bytes(result.stdout)
sys.stdout.buffer.write(result.stdout)
raise SystemExit(result.returncode)

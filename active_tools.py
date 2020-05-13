import sys, subprocess
from subprocess import check_output

jarFile = '/scratch/rabin/token_embedding/tools/JavaMethodBody.jar'
inpFile = '/scratch/rabin/token_embedding/data/Raw/Reduced'
outFile = '/scratch/rabin/token_embedding/data/Body/Reduced'

subprocess.call(['java', '-jar', jarFile, inpFile, outFile])

#body_bytes  = check_output(['java', '-jar', jarFile, javaFile])
#body_method = body_bytes.decode(sys.stdout.encoding).strip()


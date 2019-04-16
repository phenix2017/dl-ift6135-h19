import subprocess

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--no-gpu", action="store_true", default=False,
                    help="use gpu")
parser.add_argument("--email", default=False,
                    help="set this to your e-mail if you want to receive emails about your slurm jobs")
parser.add_argument("--time", default='48:00:0', help='time')
parser.add_argument("--mem", default='3G', help='memory needed for each job')
parser.add_argument("--jobname", default='s_ssm_vae')
parser.add_argument("--node", default ='eos14')
parser.add_argument("--qos", default ='unkillable')
args = parser.parse_args()

job_name = args.jobname
no_gpu = args.no_gpu

script_path = 'main_pb3.py'


def prefix(jobname):
    mail_string = '--mail-type=ALL --mail-user={}'.format(args.email) if args.email else ''
    if not args.no_gpu:
        return "srun -w {} --time={} --cpus-per-task=6 --ntasks=1 --job-name {} --gres=gpu {} --mem={} --qos {}".format(
                args.node, args.time, jobname, mail_string, args.mem, args.qos)
    else:
        return "srun -w {} --time={} --cpus-per-task=6 {} --job-name {} --mem-per-cpu={} ".format(args.node,
                args.time, mail_string, jobname, args.mem)


script_to_run = ' python {}'.format(
                script_path)

subprocess.check_output(prefix(job_name) + script_to_run, shell=True)
